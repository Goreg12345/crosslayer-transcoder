#!/usr/bin/env python3
"""CPU-only manual bridge for the actual 17 MOLT MCP tools.

All writes live in a separate manual workspace. The existing dashboard performs
subject-model inference; scoring uses only an explicitly configured New York
endpoint, with no local-model fallback.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import uuid
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.molt_agentic_full_mcp import FullTools, build_server
from tools.molt_agentic_training import TrainingCorpus
from tools.molt_scorer_runtime import load_scorer_runtime
from tools.molt_tool_payload import payload_preview


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def isolated_directory(directory,run_directory):
    source=Path(directory).resolve();manual=Path(run_directory).resolve()
    if manual==source or source in manual.parents or any(part.startswith('molt-agentic-') for part in manual.parts):
        raise ValueError('Manual workspace must be separate from dataset and agentic pilot outputs')
    return manual


class ManualCorpus(TrainingCorpus):
    def __init__(self,directory,run_directory):
        self.manual_directory=isolated_directory(directory,run_directory)
        selection=self.manual_directory/'manual-selection.json'
        write_json(selection,dict(selected=[],scope='manual exploration; feature range comes from dataset metadata'))
        super().__init__(directory,selection,output_directory=self.manual_directory)
        self.features=frozenset(range(json.loads((Path(directory)/'metadata.json').read_text())['n_features']))
        self._initial={feature:[] for feature in self.features}
        self._loaded_initial=set()
        self.provenance['manual_scope']='all features; initial windows filtered to manifest training conversations only'

    def _feature(self,feature):
        super()._feature(feature)
        with self._lock:
            if feature not in self._loaded_initial:
                path=self.evidence_directory/'evidence'/f'{feature}.json'
                if path.exists():
                    # No baseline explanation/result file is read. Some rare legacy
                    # explanation windows were reassigned from evaluation; exclude them.
                    for row in json.loads(path.read_text()).get('train',[]):
                        if row['conversation'] in self.train_conversations and row['conversation'] not in self.test_conversations:
                            self._check_window(row['conversation'],row['start'],row['stop'])
                            self._initial[feature].append(row)
                self._loaded_initial.add(feature)
        return feature

    def diagnostic_samples(self,feature):
        samples=super().diagnostic_samples(feature)
        if not samples['train']:
            samples['train']=deepcopy(samples['test'])
        return samples


class RemoteOnlyScorer:
    def __init__(self,corpus):self.corpus=corpus

    async def availability(self):
        runtime=load_scorer_runtime(self.corpus.output_directory)
        if runtime is None:
            return dict(available=False,reason='New York scorer is not configured. No local scoring fallback.')
        if runtime['base_url']!='http://127.0.0.1:18000/v1':
            return dict(available=False,reason='Only the explicit New York scorer is permitted. No local scoring fallback.')
        if self.corpus.scorer_runtime is None or self.corpus.scorer_runtime['runtime_sha256']!=runtime['runtime_sha256']:
            return dict(available=False,reason='Scorer configuration changed; restart the manual service before scoring. No local scoring fallback.')
        def probe():
            with urlopen(runtime['base_url']+'/models',timeout=3) as response:
                result=json.load(response)
            if runtime['model'] not in [row.get('id') for row in result.get('data',[])]:
                raise ValueError('Configured New York model is not available')
        try:await asyncio.to_thread(probe)
        except Exception as exc:
            return dict(available=False,endpoint=runtime['base_url'],reason=f'New York scorer unavailable: {exc}. No local scoring fallback.')
        return dict(available=True,endpoint=runtime['base_url'],model=runtime['model'])

    async def score_explanation(self,feature,explanation):
        state=await self.availability()
        if not state['available']:raise ValueError(state['reason'])
        if self.corpus.scorer_runtime is None:
            raise ValueError('Restart the manual bridge after configuring the New York scorer; local fallback is forbidden')
        return await self.corpus.score_explanation(feature,explanation)


class ManualTools(FullTools):
    """Reuse FullTools implementations while permitting features without corpus hits."""
    def __init__(self,directory,run_directory,inspection_url,corpus):
        self.directory=Path(directory);self.evidence_directory=self.directory/'autointerp-layer'
        meta=json.loads((self.directory/'metadata.json').read_text())
        self.features=corpus.features;self.layer=meta['layer'];self.model_name=meta['model']
        self.train_conversations=corpus.train_conversations;self.test_conversations=corpus.test_conversations
        self.run_directory=isolated_directory(directory,run_directory);self.state_directory=self.run_directory/'tool-state'
        self.corpus=corpus;self.scorer=RemoteOnlyScorer(corpus)
        self._state_lock=threading.RLock();self._lock=threading.Lock();self._engine=None;self._subject_client=None
        self.device='remote-existing-dashboard';self.inspection_url=inspection_url
        self._engine_factory=lambda directory,device:self.subject()

    def _feature(self,feature):
        if type(feature) is not int or feature not in self.features:raise ValueError('Feature is outside the layer feature range')
        return []

    def get_latent_info(self,feature):
        self._feature(feature)
        _,values,conversations,_=self.corpus._data(feature)
        return dict(feature=feature,layer=self.layer,model=self.model_name,
                    activation_kind='nonnegative scalar MOLT gate; not a fixed decoder vector',
                    evidence_scope='manifest training conversations only',active_training_tokens=len(values),
                    active_training_conversations=len(set(conversations.tolist())),
                    training_max_activation=float(max(values)) if len(values) else 0,
                    initial_training_windows=len(self.corpus.initial_examples(feature)['examples']),
                    note='No observed training activations' if not len(values) else 'Synthetic probes and steering use the existing subject server')


class Playground:
    def __init__(self,state,server,tokenizer_path=None):
        self.tokenizer=None
        if tokenizer_path is not None and Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            self.tokenizer=Tokenizer.from_file(str(tokenizer_path))
        self.state=state;self.server=server;self.history_directory=state.run_directory/'history'
        self.calls=asyncio.Semaphore(2)
        self._history_payload_cache={}

    async def inventory(self):
        tools=await self.server.list_tools()
        return dict(tools=[t.model_dump(mode='json',by_alias=True) for t in tools],
                    feature_count=len(self.state.features),layer=self.state.layer,
                    scope='All layer features. Corpus feedback is training-only; drafts and memory belong to this manual workspace.',
                    scoring=await self.state.scorer.availability())

    async def call(self,name,arguments):
        if not isinstance(arguments,dict):raise ValueError('arguments must be an object')
        self.state._feature(arguments.get('feature'))
        identifier=uuid.uuid4().hex
        started=time.monotonic()
        row=dict(id=identifier,feature=arguments['feature'],tool=name,arguments=arguments,
                 created_at=datetime.now(timezone.utc).isoformat(),status='running')
        path=self.history_directory/f'{identifier}.json';write_json(path,row)
        try:
            async with self.calls:
                result=await self.server.call_tool(name,arguments)
            row.update(status='error' if getattr(result,'is_error',False) else 'complete',
                       result=result.model_dump(mode='json',by_alias=True))
        except Exception as exc:
            cause=exc
            while cause.__cause__ is not None:cause=cause.__cause__
            row.update(status='error',error=f'{type(cause).__name__}: {cause}')
        row['finished_at']=datetime.now(timezone.utc).isoformat()
        row['elapsed_seconds']=round(time.monotonic()-started,3)
        row['result_payload_bytes']=len(json.dumps(row.get('result',row.get('error')),ensure_ascii=False,separators=(',',':')).encode())
        row['model_payload']=payload_preview(row.get('result'),self.tokenizer,error=row.get('error'))
        row['compact_result_tokens']=row['model_payload']['model_tokens']
        row['token_count_note']=row['model_payload']['tokenizer_note']
        write_json(path,row)
        return row

    def history(self,feature):
        self.state._feature(feature)
        rows=[]
        for path in self.history_directory.glob('*.json'):
            row=json.loads(path.read_text())
            if row['feature']==feature:
                rows.append((path,row))
        recent=sorted(rows,key=lambda item:item[1]['created_at'],reverse=True)[:30]
        for path,row in recent:
            # Enrich only visible calls, without rerunning tools or rewriting history.
            if 'model_payload' not in row and row.get('status')!='running':
                key=(str(path),path.stat().st_mtime_ns)
                if key not in self._history_payload_cache:
                    self._history_payload_cache[key]=payload_preview(row.get('result'),self.tokenizer,error=row.get('error'))
                    if len(self._history_payload_cache)>64:
                        del self._history_payload_cache[next(iter(self._history_payload_cache))]
                row['model_payload']=self._history_payload_cache[key]
        return [row for _,row in recent]


def create_app(playground):
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    async def tools(request):return JSONResponse(await playground.inventory())
    async def history(request):
        try:return JSONResponse(dict(history=playground.history(int(request.query_params.get('feature','0')))))
        except (ValueError,TypeError) as exc:return JSONResponse(dict(error=str(exc)),status_code=400)
    async def call(request):
        from urllib.parse import urlsplit
        origin=request.headers.get('origin')
        if origin and urlsplit(origin).netloc!=request.headers.get('host'):
            return JSONResponse(dict(error='Cross-origin requests are not accepted'),status_code=403)
        if request.headers.get('content-type','').split(';')[0]!='application/json':
            return JSONResponse(dict(error='Use application/json'),status_code=415)
        try:
            raw=await request.body()
            if len(raw)>131072:raise ValueError('Request too large')
            body=json.loads(raw)
            return JSONResponse(await playground.call(body['name'],body['arguments']))
        except (ValueError,TypeError,KeyError) as exc:return JSONResponse(dict(error=str(exc)),status_code=400)
    app=playground.server.streamable_http_app(host='127.0.0.1')
    app.routes.extend([Route('/tools',tools),Route('/history',history),Route('/call',call,methods=['POST'])])
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,default=Path('results/molt-manual-playground'))
    parser.add_argument('--inspection-url',default='http://127.0.0.1:8765')
    parser.add_argument('--scorer-runtime',type=Path,help='Explicit New York scorer runtime; no automatic pilot runtime or local fallback')
    parser.add_argument('--port',type=int,default=8006)
    parser.add_argument('--tokenizer',type=Path,default=Path('/home/georg/Code/vllm-utils/models/Qwen3.8-27B-NVFP4/tokenizer.json'))
    parser.add_argument('--compact-tools',action='store_true',help='Serve the readable compact-text tools (molt_compact_tools.py)')
    args=parser.parse_args()
    run=isolated_directory(args.directory,args.run_directory);run.mkdir(parents=True,exist_ok=True)
    if args.scorer_runtime:
        runtime=json.loads(args.scorer_runtime.read_text())
        # Existing New York SSH tunnel. Never silently use either local vLLM port.
        if runtime.get('base_url')!='http://127.0.0.1:18000/v1':
            raise ValueError('Manual scoring is restricted to the explicit New York tunnel on18000')
        write_json(run/'scorer-runtime.json',runtime)
    elif (run/'scorer-runtime.json').exists():
        runtime=load_scorer_runtime(run)
        if runtime['base_url']!='http://127.0.0.1:18000/v1':raise ValueError('Local scoring fallback is forbidden')
    corpus=ManualCorpus(args.directory,run)
    state=ManualTools(args.directory,run,args.inspection_url,corpus)
    if args.compact_tools:
        from tools.molt_compact_tools import build_server as build_compact_server
        server=build_compact_server(state)
    else:
        server=build_server(state)
    import uvicorn
    uvicorn.run(create_app(Playground(state,server,args.tokenizer)),host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
