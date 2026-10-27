"""Expose saved training-scorer API messages without generating replacement prompts."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crosslayer_transcoder.dashboard.token_display import escape_added_tokens
from render_molt_agentic import esc

def literal(value):return esc(value).replace('\r','&#13;')


PREFIX='\n\nText examples:\n\nExample 0: '
SUFFIX='\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'


class TrainingScorerInputs:
    def __init__(self,root,output):self.root=Path(root).resolve();self.output=Path(output);self.cache={}
    def source(self,data):
        sha=data.get('diagnostic_sha256');recorded=data.get('audit')
        if not str(data.get('feature','')).isdecimal():return None
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha) or not isinstance(recorded,str):return None
        given=Path(recorded)
        candidates=[given.resolve(),(self.root/given).resolve(),(self.root/'training-scores'/str(data.get('feature'))/(sha+'.jsonl')).resolve()]
        for path in candidates:
            if path.is_file() and path.is_relative_to(self.root/'training-scores') and path.name==sha+'.jsonl':return path
        return None
    def prepare(self,data):
        if not isinstance(data,dict) or not isinstance(data.get('methods'),dict):return {}
        source=self.source(data)
        if source is None:return {}
        key=str(source)
        if key not in self.cache:
            raw=source.read_bytes();records=[]
            for line_no,line in enumerate(raw.decode().splitlines(),1):
                try:row=json.loads(line)
                except ValueError:continue
                messages=row.get('messages',row.get('request_body',{}).get('messages',[]))
                if messages and 'messages' not in row:row=dict(row,messages=messages,endpoint_url=row.get('base_url'),max_tokens=row.get('request_body',{}).get('max_tokens'))
                if messages and messages[-1].get('role')=='user' and isinstance(messages[-1].get('content'),str):records.append((line_no,row))
            source_hash=hashlib.sha256(raw).hexdigest()
            relative=f'scorer-inputs/{data["feature"]}/{data["diagnostic_sha256"]}.html'
            self.write_page(relative,source,source_hash,records)
            self.cache[key]=(records,source_hash,relative)
        records,source_hash,relative=self.cache[key]
        # Escaping is used ONLY to identify the saved request. Every displayed
        # character is subsequently sliced from the original audited message.
        prefix='Latent explanation: '+escape_added_tokens(str(data.get('explanation','')))+PREFIX
        by_text={}
        for line,row in records:
            actual=row['messages'][-1]['content']
            if actual.startswith(prefix) and actual.endswith(SUFFIX):
                text=actual[len(prefix):-len(SUFFIX)]
                by_text.setdefault(text,[]).append((line,row))
        methods={}
        for name,result in data['methods'].items():
            methods[name]={}
            for index,row in enumerate(result.get('rows',[])):
                text=row.get('shown_text')
                if text is None and name=='detection':text=''.join(row.get('str_tokens',[]))
                if not isinstance(text,str):continue
                native_matches=[]
                for line,audit in records:
                    if not audit.get('request_body') or audit.get('task')!=name:continue
                    actual=audit['messages'][-1]['content']
                    if not actual.startswith('Latent explanation: '+escape_added_tokens(str(data.get('explanation','')))+'\n'):continue
                    target=escape_added_tokens(text)
                    for slot in range(5):
                        marker=f'Example {slot}: '
                        location=actual.find(marker+target)
                        if location<0:continue
                        start=location+len(marker);answer=(audit.get('response',{}).get('choices') or [{}])[0].get('message',{}).get('content','')
                        prediction_matches=False
                        try:
                            import re
                            parsed=json.loads(re.search(r'\[[^\]]*\]',answer).group())
                            prediction_matches=len(parsed)==5 and row.get('prediction') is not None and parsed[slot]==row['prediction']
                        except (ValueError,TypeError,AttributeError,IndexError):pass
                        native_matches.append((line,audit,actual[start:start+len(target)],prediction_matches))
                if native_matches:
                    line,audit,exact,prediction_matches=next((item for item in reversed(native_matches) if item[3]),native_matches[-1])
                    methods[name][index]=dict(exact_example=exact,url=relative+f'#line-{line}',audit_path=source.relative_to(self.root).as_posix(),audit_sha256=source_hash,audit_line=line,
                        request_key=audit.get('request_body_sha256'),endpoint=audit.get('base_url'),matches_recorded_prediction=prediction_matches,matching_attempts=len(native_matches),
                        all_attempt_urls=[relative+f'#line-{item[0]}' for item in native_matches],provenance='Literal case substring verified in saved native five-example scorer request_body.messages; full batch and explanation retained.')
                    continue
                matches=by_text.get(escape_added_tokens(text),[])
                if not matches:continue
                predicted='[1]' if row.get('prediction') is True else '[0]' if row.get('prediction') is False else None
                compatible=[item for item in matches if str(item[1].get('response') or '').strip()==predicted and item[1].get('finish_reason')!='length']
                line,audit=(compatible or matches)[-1]
                actual=audit['messages'][-1]['content']
                methods[name][index]=dict(exact_example=actual[len(prefix):-len(SUFFIX)],url=relative+f'#line-{line}',
                    audit_path=source.relative_to(self.root).as_posix(),audit_sha256=source_hash,audit_line=line,
                    request_key=audit.get('request_key'),endpoint=audit.get('endpoint_url'),
                    matches_recorded_prediction=bool(compatible),matching_attempts=len(matches),
                    all_attempt_urls=[relative+f'#line-{n}' for n,_ in matches],
                    provenance='Literal excerpt of saved final-user API message. Matched using the explanation and case text (with recorded added-token display escaping); not generated for display.')
        return methods
    def write_page(self,relative,source,source_hash,records):
        cards=[]
        for line,row in records:
            messages=row['messages'];final=messages[-1]['content'];target=f'user-{line}'
            all_messages=''.join('<details><summary>'+str(i+1)+' · '+esc(message.get('role','message'))+'</summary><pre><span>'+literal(message.get('content',''))+'</span></pre></details>' for i,message in enumerate(messages))
            metadata={k:v for k,v in row.items() if k!='messages'}
            cards.append(f'<article id="line-{line}"><h2>Audit line {line}</h2><p>Request key: <code>{esc(row.get("request_key","not recorded"))}</code><br>Endpoint: {esc(row.get("endpoint_url","not recorded"))} · model: {esc(row.get("model","not recorded"))}<br>Recorded API prompt tokens: {esc(row.get("usage",{}).get("prompt_tokens","not recorded"))} · output budget: {esc(row.get("max_tokens","not recorded"))}</p><h3>Exact audited final user message</h3><button data-copy="{target}">Copy exact final user text</button><pre class="exact-user" id="{target}"><span>{literal(final)}</span></pre><details><summary>All exact API messages · system, few-shots and final user</summary>{all_messages}</details><details><summary>Recorded response / request metadata</summary><pre>{esc(json.dumps(metadata,ensure_ascii=False,indent=2))}</pre></details></article>')
        audit_relative=source.relative_to(self.root).as_posix()
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact training-scorer inputs</title><style>body{max-width:1150px;margin:auto;padding:24px;font:14px/1.6 system-ui;background:#f4f7f7;color:#223a43}a{color:#087f79}article{background:white;border:1px solid #d5e2e4;border-radius:9px;padding:22px;margin:24px 0;scroll-margin-top:15px}h1{font-size:25px}h2{font-size:19px}h3{font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;background:#f1f6f5;font-size:12px;border-radius:6px}details{padding:12px;border:1px solid #dce5e7;border-radius:6px;margin:10px 0}summary{cursor:pointer;font-weight:600}button{border:1px solid #c2d6d1;border-radius:5px;background:white;padding:7px;color:#087f79;cursor:pointer}p,code{overflow-wrap:anywhere}:target{outline:2px solid #80bcb0}</style><h1>Exact saved training-scorer input</h1><p>These strings come directly from the saved scorer audit. Nothing has been rephrased or re-highlighted. The final user message includes the explanation, complete example window and response instructions. System and few-shot messages are preserved separately. This is the audited API-message payload, not a newly reconstructed model chat-template string.</p><p>Detection consumes the whole displayed window; it does not receive a dashboard activation overlay. Fuzz markers such as &lt;&lt; and &gt;&gt; are retained literally when present in the request. Window construction is run-specific; inspect the evidence manifest. No activation-position rule is inferred from this display.</p>'''
        page+='<p><a href="../../../'+esc(audit_relative)+'">Original audit JSONL</a> · <a href="../../index.html">Trajectory explorer</a><br>Source SHA-256: <code>'+esc(source_hash)+'</code></p>'+''.join(cards)
        page+='''<script>document.addEventListener('click',async event=>{const button=event.target.closest('[data-copy]');if(!button)return;const text=document.getElementById(button.dataset.copy).textContent;try{if(navigator.clipboard&&isSecureContext)await navigator.clipboard.writeText(text);else{const box=document.createElement('textarea');box.value=text;document.body.append(box);box.select();const ok=document.execCommand('copy');box.remove();if(!ok)throw Error('Clipboard unavailable')}button.textContent='Copied exact text'}catch{button.textContent='Clipboard unavailable; select the full text below'}})</script></html>'''
        dest=self.output/relative;dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix('.html.tmp');temp.write_text(page);temp.replace(dest)
