"""Step 1 of additiona_followup_plan2: vanilla answer-correctness for `N+3=`.

The v1 script (`plus3_followup_check.py`) reported 4/12 correct, but the
"completion" column shows the model was getting cut off mid-echo of the
chat-templated prompt before it ever emitted the answer. So that signal was a
token-budget artifact, not a real arithmetic-failure signal.

This script fixes the budget bug:

  * generates ≥30 new tokens with greedy decoding
  * parses the answer out of the *full* completion via regex `=\\s*([0-9]+)`
  * marks correct / incorrect against `int(N)+3`

Vanilla Gemma3-4b-it only — no MoLT splice. Step 2 of the plan handles the
full-MoLT-spliced version.

Outputs `feature_dash/addition_followup/v2/answer_check.json` and a
companion `results.md` with a 12-row table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS: list[tuple[str, str]] = [
    ("1d", "2+3="),
    ("1d", "4+3="),
    ("1d", "0+3="),
    ("1d", "3+3="),
    ("2d", "52+3="),
    ("2d", "54+3="),
    ("2d", "50+3="),
    ("2d", "53+3="),
    ("3d", "852+3="),
    ("3d", "854+3="),
    ("3d", "850+3="),
    ("3d", "853+3="),
]


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


_ANSWER_RE_ECHO = re.compile(r"=\s*([0-9]+)")
_ANSWER_RE_FALLBACK = re.compile(r"([0-9]+)")


def expected_answer(prompt: str) -> int:
    return int(prompt.split("+")[0]) + 3


def apply_chat_template(tokenizer, user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    s = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    bos = getattr(tokenizer, "bos_token", None)
    if bos and s.startswith(bos):
        s = s[len(bos):]
    return s


@torch.no_grad()
def greedy_complete(model, tokenizer, input_ids: torch.Tensor, n: int) -> str:
    """Greedy-decode up to `n` new tokens; stop on EOS / <end_of_turn>."""
    eos = tokenizer.eos_token_id
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    cur = input_ids
    new_ids: list[int] = []
    for _ in range(n):
        nxt = int(model(input_ids=cur).logits[0, -1].argmax().item())
        new_ids.append(nxt)
        if eos is not None and nxt == eos:
            break
        if end_of_turn is not None and nxt == end_of_turn:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def parse_answer(completion: str) -> int | None:
    """Pull the answer integer out of the completion.

    First tries `= <digits>` (chat-template echo style: '... 2 + 3 = 5 ...').
    Falls back to the first contiguous digit run anywhere in the completion
    (answer-only style: '5\\n<end_of_turn>'). Returns None if neither matches.
    """
    m = _ANSWER_RE_ECHO.search(completion)
    if m is not None:
        return int(m.group(1))
    m = _ANSWER_RE_FALLBACK.search(completion)
    if m is not None:
        return int(m.group(1))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="google/gemma-3-4b-it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE))
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument(
        "--prompt-template",
        default="{p}",
        help="format string with `{p}` placeholder substituted with each `N+3=` prompt; "
             "e.g. 'please respond with only the answer, {p}' to suppress the echo",
    )
    p.add_argument(
        "--space-operators",
        action="store_true",
        help="convert `N+3=` to `N + 3 =` in the prompts before applying --prompt-template",
    )
    p.add_argument(
        "--out-dir",
        default="feature_dash/addition_followup/v2",
        help="output dir; writes answer_check.json and results.md inside",
    )
    return p.parse_args()


def write_results_md(payload: dict, path: Path) -> None:
    md: list[str] = []
    md.append("# +3 follow-up v2 — Step 1 (vanilla answer-correctness)")
    md.append("")
    md.append(
        "Re-run of [v1](../results.md) with `max_new_tokens="
        f"{payload['max_new_tokens']}` and answer parsed out of the completion "
        "via regex `=\\s*([0-9]+)`. v1's 4/12 was a token-budget artifact — "
        "echoes longer than the budget got cut off before the answer emitted."
    )
    md.append("")
    md.append(f"- base model: `{payload['base_model']}`")
    md.append(f"- dtype: `{payload['dtype']}` device: `{payload['device']}`")
    md.append(f"- prompt template: `{payload['prompt_template']}`")
    md.append(f"- chat template applied; greedy decoding")
    md.append("")
    md.append("## Per-prompt results")
    md.append("")
    md.append("| group | prompt | expected | parsed | greedy completion | correct? |")
    md.append("|---|---|---:|---:|---|:--:|")

    n_correct = 0
    for r in payload["results"]:
        check = "✓" if r["correct"] else "✗"
        if r["correct"]:
            n_correct += 1
        completion_inline = r["completion"].replace("\n", "\\n").replace("|", "\\|")
        parsed_str = "—" if r["parsed_answer"] is None else str(r["parsed_answer"])
        md.append(
            f"| {r['group']} | `{r['prompt']}` | {r['expected_answer']} | "
            f"{parsed_str} | `{completion_inline}` | {check} |"
        )
    md.append("")
    md.append(f"**Correct: {n_correct} / {len(payload['results'])}**")
    md.append("")
    md.append(
        "## Step-1 stop condition (per "
        "[plan v2](../../../notes/additiona_followup_plan2.md))"
    )
    md.append("")
    if n_correct >= 11:
        md.append(
            f"> ✅ {n_correct}/12 correct — proceed to Step 2 "
            "(full-MoLT-splice answer-correctness)."
        )
    else:
        md.append(
            f"> ❌ {n_correct}/12 correct — stop and figure out why. "
            "There's a real arithmetic-failure regime to investigate "
            "before the MoLT story makes sense."
        )
    md.append("")
    path.write_text("\n".join(md) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=> loading {args.base_model} ({args.dtype}) on {args.device}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=_DTYPE[args.dtype])
        .to(args.device)
        .eval()
    )

    if "{p}" not in args.prompt_template:
        print(
            f"WARN: --prompt-template={args.prompt_template!r} has no `{{p}}` placeholder; "
            "every prompt will be identical.",
            file=sys.stderr,
        )

    results: list[dict] = []
    for group, raw_prompt in PROMPTS:
        # Optionally pretty-print operators ("2+3=" -> "2 + 3 =") so an IT model
        # that early-stops on dense single-token "N+3=" gets a clearer parse.
        prompt_for_eval = (
            raw_prompt.replace("+", " + ").replace("=", " =")
            if args.space_operators
            else raw_prompt
        )
        templated_user_msg = args.prompt_template.format(p=prompt_for_eval)
        prompt_text = apply_chat_template(tokenizer, templated_user_msg)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc.input_ids.to(args.device)

        completion = greedy_complete(
            model, tokenizer, input_ids, args.max_new_tokens
        )
        parsed = parse_answer(completion)
        expected = expected_answer(raw_prompt)
        correct = parsed is not None and parsed == expected

        print(
            f"[{group}] {raw_prompt!s:<10s} expected={expected:>4d}  "
            f"parsed={parsed!s:<5s}  ok={correct}",
            file=sys.stderr,
        )
        results.append({
            "group": group,
            "prompt": raw_prompt,
            "user_message": templated_user_msg,
            "templated_prompt": prompt_text,
            "expected_answer": expected,
            "parsed_answer": parsed,
            "correct": correct,
            "completion": completion,
            "n_input_tokens": int(input_ids.shape[1]),
        })

    payload = {
        "base_model": args.base_model,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": args.prompt_template,
        "results": results,
    }

    out_json = out_dir / "answer_check.json"
    out_md = out_dir / "results.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_results_md(payload, out_md)

    n_correct = sum(int(r["correct"]) for r in results)
    print(
        f"\nWrote {out_json}\nWrote {out_md}\nVanilla: {n_correct}/{len(results)} correct",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
