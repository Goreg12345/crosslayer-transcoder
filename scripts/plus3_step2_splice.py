"""Step 2 of additiona_followup_plan2: vanilla vs full-MoLT-splice answer-correctness.

For each `N+3=` prompt, runs Gemma3-4b-it twice — once vanilla, once with
MoLT spliced into every block (full transcoder substitution) — and reports:

  * vanilla parsed answer, ✓/✗ vs ground truth
  * spliced parsed answer, ✓/✗ vs ground truth
  * top-1 token agreement at the answer position (last prompt token)
  * KL(vanilla logits || spliced logits) at the answer position

Reuses `splice_hooks`, `_resolve_blocks`, and `_molt_layer_forward` from
`scripts/verify_molt.py` so the splice point exactly matches the sanity-check
script.

Outputs `feature_dash/addition_followup/v3/step2_splice.{json,md}`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse splice machinery + the answer parser from step 1.
sys.path.insert(0, str(Path(__file__).parent))
from verify_molt import (  # type: ignore
    _resolve_blocks,
    splice_hooks,
)
from plus3_answer_check import (  # type: ignore
    PROMPTS,
    apply_chat_template,
    expected_answer,
    greedy_complete,
    parse_answer,
)

from crosslayer_transcoder.feature_dash.multilayer import (
    download_multilayer_from_hf,
    load_multilayer_molt,
)


_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@torch.no_grad()
def last_pos_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids).logits[0, -1].float().softmax(-1).cpu()


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    return float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())


def top1(probs: torch.Tensor, tokenizer) -> tuple[int, str, float]:
    p, i = probs.max(0)
    tid = int(i)
    return tid, tokenizer.decode([tid]), float(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="google/gemma-3-4b-it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE))
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument(
        "--prompt-template",
        default="please respond with only the answer, {p}",
        help="format string with `{p}` placeholder substituted with each `N+3=` prompt",
    )
    p.add_argument(
        "--ckpt-dir",
        default="checkpoints/molt-gemma3-4b-it",
        help="local dir holding per-layer .pt files (default: pre-built symlinks "
             "in checkpoints/molt-gemma3-4b-it that point at the HF cache); "
             "set to None to use --hf-folder instead",
    )
    p.add_argument("--hf-repo", default="kylelovesllms/molt-sweeps")
    p.add_argument(
        "--hf-folder",
        default=None,
        help="HF folder under --hf-repo, used only if --ckpt-dir is None. "
             "Original upload was molt-multilayer-gemma3-4b-it-N50-100M-2gpu-b200-fp32weights",
    )
    p.add_argument(
        "--run-name",
        default="molt-gemma3-4b-it",
        help="filename prefix shared by every per-layer .pt. Default matches "
             "the symlinks in checkpoints/molt-gemma3-4b-it/. The original HF "
             "upload's run_name is the dummy-logger repr; use that with --hf-folder.",
    )
    p.add_argument("--step", default=None, help="default: latest in folder")
    p.add_argument(
        "--out-dir",
        default="feature_dash/addition_followup/v3",
        help="output dir for step2_splice.json / .md",
    )
    return p.parse_args()


def write_md(payload: dict, path: Path) -> None:
    md: list[str] = []
    md.append("# +3 follow-up v3 — Step 2 (full-MoLT-splice answer-correctness)")
    md.append("")
    md.append(
        "Same prompts and chat template as [Step 1](results.md), now also run "
        "with MoLT spliced into every block (full transcoder substitution). "
        "Reports vanilla vs spliced answer parses, plus next-token KL and "
        "top-1 agreement at the last prompt position."
    )
    md.append("")
    md.append(f"- base model: `{payload['base_model']}` ({payload['dtype']})")
    md.append(f"- MoLT: `{payload['hf_folder']}` @ `{payload['step']}`")
    md.append(f"- prompt template: `{payload['prompt_template']}`")
    md.append("- splice: forward-hook overwrite of `post_feedforward_layernorm` at all 34 blocks")
    md.append("")
    md.append("## Per-prompt comparison")
    md.append("")
    md.append("| group | prompt | expected | vanilla | spliced | top1 agree? | KL(v‖s) |")
    md.append("|---|---|---:|---|---|:--:|---:|")

    n_v = n_s = n_match = 0
    for r in payload["results"]:
        v_ok = "✓" if r["vanilla_correct"] else "✗"
        s_ok = "✓" if r["spliced_correct"] else "✗"
        match = "✓" if r["top1_agree"] else "✗"
        n_v += int(r["vanilla_correct"])
        n_s += int(r["spliced_correct"])
        n_match += int(r["top1_agree"])
        v_str = f"{r['vanilla_parsed']}" if r["vanilla_parsed"] is not None else "—"
        s_str = f"{r['spliced_parsed']}" if r["spliced_parsed"] is not None else "—"
        md.append(
            f"| {r['group']} | `{r['prompt']}` | {r['expected_answer']} | "
            f"{v_str} {v_ok} | {s_str} {s_ok} | {match} | {r['kl_vanilla_spliced']:.3f} |"
        )

    md.append("")
    md.append(f"**Vanilla: {n_v}/{len(payload['results'])} correct.**  "
              f"**Spliced: {n_s}/{len(payload['results'])} correct.**  "
              f"**Top-1 agreement at answer position: {n_match}/{len(payload['results'])}.**")
    md.append("")
    md.append("## Sample completions")
    md.append("")
    for r in payload["results"]:
        v_c = r["vanilla_completion"].replace("\n", "\\n")
        s_c = r["spliced_completion"].replace("\n", "\\n")
        md.append(f"- `{r['prompt']}`")
        md.append(f"    - vanilla: `{v_c}`")
        md.append(f"    - spliced: `{s_c}`")
    md.append("")
    md.append("## Step-2 stop condition (per "
              "[plan v2](../../../notes/additiona_followup_plan2.md))")
    md.append("")
    if n_v - n_s <= 1 and n_match >= 11:
        md.append(
            f"> ✅ Spliced model loses ≤1 prompt vs vanilla and agrees on "
            f"top-1 in {n_match}/12 cases — proceed to Step 3 "
            "(relocate answer position; collect gates there)."
        )
    else:
        md.append(
            f"> ❌ Spliced model differs from vanilla on too many prompts "
            f"(vanilla={n_v}/12, spliced={n_s}/12, top1-agree={n_match}/12). "
            "Transcoder isn't faithful enough on this task — report and revisit."
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
    arch, blocks = _resolve_blocks(model)
    print(f"   detected arch={arch} n_blocks={len(blocks)}", file=sys.stderr)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"   GPU after base load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated", file=sys.stderr)

    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"--ckpt-dir {ckpt_dir} does not exist")
        # find_latest_step is happy to inspect a local dir; pass the user's run_name.
        from crosslayer_transcoder.feature_dash.multilayer import find_latest_step
        step = args.step or find_latest_step(ckpt_dir, args.run_name)
        print(f"=> using local checkpoints {ckpt_dir} run_name={args.run_name!r} "
              f"step={step!r}", file=sys.stderr)
    else:
        if not args.hf_folder:
            raise ValueError("either --ckpt-dir or --hf-folder must be set")
        print(f"=> downloading MoLT from {args.hf_repo}/{args.hf_folder} …", file=sys.stderr)
        ckpt_dir, step = download_multilayer_from_hf(
            repo_id=args.hf_repo,
            folder=args.hf_folder,
            run_name=args.run_name,
            step=args.step,
        )
        print(f"   step={step!r}, cache={ckpt_dir}", file=sys.stderr)
    print(f"=> loading MoLT (cpu staging, then move to {args.device})…", file=sys.stderr)
    molt, meta = load_multilayer_molt(
        ckpt_dir=ckpt_dir,
        run_name=args.run_name,
        step=step,
        device=args.device,
    )
    print(
        f"   d_acts={meta.d_acts} n_features={meta.n_features} "
        f"n_layers={meta.n_layers} ranks={meta.ranks} N={meta.N}",
        file=sys.stderr,
    )
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"   GPU after MoLT load: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated", file=sys.stderr)

    splice_layers = list(range(len(blocks)))

    results: list[dict] = []
    for group, raw_prompt in PROMPTS:
        user_msg = args.prompt_template.format(p=raw_prompt)
        prompt_text = apply_chat_template(tokenizer, user_msg)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc.input_ids.to(args.device)

        # Vanilla
        v_probs = last_pos_probs(model, input_ids)
        v_completion = greedy_complete(model, tokenizer, input_ids, args.max_new_tokens)
        v_parsed = parse_answer(v_completion)
        v_top1_id, v_top1_tok, v_top1_p = top1(v_probs, tokenizer)

        # Spliced
        with splice_hooks(model, molt, arch, splice_layers):
            s_probs = last_pos_probs(model, input_ids)
            s_completion = greedy_complete(model, tokenizer, input_ids, args.max_new_tokens)
        s_parsed = parse_answer(s_completion)
        s_top1_id, s_top1_tok, s_top1_p = top1(s_probs, tokenizer)

        expected = expected_answer(raw_prompt)
        v_correct = v_parsed is not None and v_parsed == expected
        s_correct = s_parsed is not None and s_parsed == expected
        kl_vs = kl(v_probs, s_probs)
        agree = v_top1_id == s_top1_id

        print(
            f"[{group}] {raw_prompt:<10s} "
            f"v={v_parsed!s:<5s}({'✓' if v_correct else '✗'}) "
            f"s={s_parsed!s:<5s}({'✓' if s_correct else '✗'}) "
            f"top1=({v_top1_tok!r}|{s_top1_tok!r}) agree={agree} "
            f"kl={kl_vs:.3f}",
            file=sys.stderr,
        )
        results.append({
            "group": group,
            "prompt": raw_prompt,
            "user_message": user_msg,
            "templated_prompt": prompt_text,
            "expected_answer": expected,
            "vanilla_parsed": v_parsed,
            "vanilla_correct": v_correct,
            "vanilla_completion": v_completion,
            "vanilla_top1": {"id": v_top1_id, "token": v_top1_tok, "prob": v_top1_p},
            "spliced_parsed": s_parsed,
            "spliced_correct": s_correct,
            "spliced_completion": s_completion,
            "spliced_top1": {"id": s_top1_id, "token": s_top1_tok, "prob": s_top1_p},
            "top1_agree": agree,
            "kl_vanilla_spliced": kl_vs,
            "n_input_tokens": int(input_ids.shape[1]),
        })

    payload = {
        "base_model": args.base_model,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": args.prompt_template,
        "hf_repo": args.hf_repo,
        "hf_folder": args.hf_folder,
        "step": step,
        "n_layers": meta.n_layers,
        "n_features": meta.n_features,
        "results": results,
    }

    out_json = out_dir / "step2_splice.json"
    out_md = out_dir / "step2_splice.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_md(payload, out_md)

    n_v = sum(int(r["vanilla_correct"]) for r in results)
    n_s = sum(int(r["spliced_correct"]) for r in results)
    n_match = sum(int(r["top1_agree"]) for r in results)
    print(
        f"\nWrote {out_json}\nWrote {out_md}\n"
        f"Vanilla: {n_v}/{len(results)}  Spliced: {n_s}/{len(results)}  "
        f"top1-agree: {n_match}/{len(results)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
