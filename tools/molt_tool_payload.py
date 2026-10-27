"""Inspect tool-message text using the explainer's actual delivery transforms."""
import hashlib
import json
from pathlib import Path

from tools.run_molt_agentic_full import compact_tool_result, safe_messages


TOKENIZER_PATH = Path('/home/georg/Code/vllm-utils/models/Qwen3.8-27B-NVFP4/tokenizer.json')
TOKENIZER_NOTE = (
    'Qwen3.8-27B tokenizer, add_special_tokens=False. Counts cover the displayed '
    'text only, excluding chat-template wrappers, tool schemas, and repeated history. '
    'They are not inference usage or Qwen3-4B subject-model token counts.'
)


def raw_tool_text(result):
    """Match the agent loop: join MCP text blocks; do not substitute structuredContent."""
    content = (result or {}).get('content', [])
    if isinstance(content, str):
        return content  # Exception records in the pilot use a string here.
    return '\n'.join(block.get('text', '') for block in content
                     if isinstance(block, dict) and block.get('type') == 'text')


def payload_preview(result, tokenizer, error=None):
    raw = raw_tool_text(result)
    if error is not None and not result:
        raw = json.dumps({'error': error})
    model = safe_messages([{'role': 'tool', 'content': compact_tool_result(raw)}])[0]['content']
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids) if tokenizer is not None else None
    return {
        'raw_text': raw, 'raw_tokens': count(raw), 'raw_bytes': len(raw.encode('utf-8')),
        'model_text': model, 'model_tokens': count(model), 'model_bytes': len(model.encode('utf-8')),
        'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'model_sha256': hashlib.sha256(model.encode('utf-8')).hexdigest(),
        'tokenizer_note': TOKENIZER_NOTE,
        'provenance': 'Preview reconstructed from the manual tool return using the current '
                      'agent delivery rules: join MCP text blocks, compact JSON when valid, '
                      'and escape special control tokens. This manual call was not sent to an explainer.',
        'sent_to_explainer': False,
        'transform': 'compact_tool_result followed by safe_messages from run_molt_agentic_full.py',
    }
