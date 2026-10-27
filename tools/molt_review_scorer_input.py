"""Find saved scorer messages for review; never synthesize a claimed model input."""
import json

from crosslayer_transcoder.dashboard.token_display import escape_added_tokens

PREFIX = '\n\nText examples:\n\nExample 0: '
SUFFIX = '\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'


def audit_index(path):
    index = {}
    if not path.exists():
        return index
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        event = json.loads(line)
        messages = event.get('messages', [])
        if event.get('task') == 'binary' and messages:
            index.setdefault(messages[-1]['content'], []).append((line_number, event))
    return index


def saved_input(index, explanation, case):
    if case.get('prediction') is None:
        return None
    text = case.get('shown_text', ''.join(case.get('str_tokens', [])))
    expected = 'Latent explanation: ' + explanation + PREFIX + text + SUFFIX
    candidates = []
    for key in dict.fromkeys((expected, escape_added_tokens(expected))):
        for line, event in index.get(key, []):
            try:
                answer = json.loads(event.get('response', ''))
            except (ValueError, TypeError):
                continue
            if event.get('finish_reason') == 'length' or answer != [int(case['prediction'])]:
                continue
            candidates.append((line, event))
    if not candidates:
        return None
    line, event = max(candidates, key=lambda pair: pair[0])
    content = event['messages'][-1]['content']
    # Extract the literal saved substring after matching the complete user message.
    prefix = 'Latent explanation: ' + explanation + PREFIX
    if not content.startswith(prefix):
        prefix = escape_added_tokens(prefix)
    return dict(line=line, example_text=content[len(prefix):-len(SUFFIX)],
                messages=event['messages'], response=event.get('response'),
                usage=event.get('usage'))
