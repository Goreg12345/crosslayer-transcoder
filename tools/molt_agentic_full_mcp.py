#!/usr/bin/env python3
"""Full agentic-Delphi tool surface adapted to blinded MOLT investigations.

Every tool requires an assigned feature. Memory and drafts are isolated by run
and feature. No baseline explanations or held-out scoring endpoints are offered.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from molt_agentic_mcp import (
    TrainingEvidence, RemoteInspectionEngine, _integer, _safe_strings,
    restore_probe_tokens, safe_result,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def bounded_text(value, name, maximum=20000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be nonempty and at most {maximum} characters')
    return value


class SubjectClient(RemoteInspectionEngine):
    def request(self, path, body=None, *, retry_busy=False):
        request = Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={'Content-Type':'application/json'} if body is not None else {})
        deadline = time.monotonic() + 180
        while True:
            try:
                with urlopen(request, timeout=180) as response:
                    return json.load(response)
            except HTTPError as exc:
                if not retry_busy or exc.code not in (409,423,429,503) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def inspect(self, text, prompt_format, feature):
        return self.request('/api/inspect', {'prompt':text,'format':prompt_format,'feature':feature}, retry_busy=True)

    def steer(self, feature, prompt, prompt_format, mode, strength, positions, max_new_tokens, seed):
        tokens = self.request('/api/tokenize', {'prompt':prompt,'format':prompt_format})
        length = len(tokens['ids'])
        if positions is None:
            positions = list(range(length))
        elif not isinstance(positions,list) or len(positions) != len(set(positions)):
            raise ValueError('positions must contain unique prompt token positions')
        for pos in positions:
            _integer(pos,'position',0,length-1)
        config = {'feature':feature,'prompt':prompt,'format':prompt_format,
                  'tokenization_id':tokens['tokenization_id'],'mode':mode,'strength_reference':'raw',
                  'assignments':[{'position':pos,'value':strength} for pos in positions],
                  'generated':{'start':0,'end':max_new_tokens-1,'value':strength},
                  'max_new_tokens':max_new_tokens,'temperature':0,'top_p':1,'seed':seed}
        submitted = self.request('/api/generate',config,retry_busy=True)
        job_id = submitted['id']
        deadline = time.monotonic()+300
        while time.monotonic()<deadline:
            job=self.request('/api/jobs/'+job_id)
            if job['state']=='done':
                result=job['result']
                # Never forward service metadata/stats outside this explicit result surface.
                return {'baseline':result['baseline'],'steered':result['steered'],
                        'first_prediction':result.get('first_prediction'),
                        'intervention':{'mode':mode,'strength':strength,'strength_reference':'raw',
                            'prompt_positions':positions,'generated_positions':[0,max_new_tokens-1],
                            'max_new_tokens':max_new_tokens,'seed':seed,'temperature':0},
                        'prompt_tokens':tokens['tokens']}
            if job['state'] in ('error','cancelled'):
                raise RuntimeError('Subject steering job '+job['state']+': '+job.get('error','cancelled'))
            time.sleep(.25)
        self.request('/api/jobs/'+job_id+'/cancel',{})
        raise TimeoutError('Subject steering exceeded 300 seconds and cancellation was requested')


class FullTools(TrainingEvidence):
    def __init__(self, directory, selection_path, run_directory, *, inspection_url,
                 corpus=None, scorer=None, subject_client=None, evidence_directory=None):
        super().__init__(directory,selection_path,evidence_directory=evidence_directory)
        self.run_directory=Path(run_directory)
        self.state_directory=self.run_directory/'tool-state'
        self.corpus=corpus
        self.scorer=scorer
        self._state_lock=threading.RLock()
        self._subject_client=subject_client
        self.inspection_url=inspection_url
        self._engine_factory=lambda directory,device:self.subject()

    def subject(self):
        if self._subject_client is None:
            self._subject_client=SubjectClient(self.directory,url=self.inspection_url)
        return self._subject_client

    def read_state(self,feature):
        self._feature(feature)
        path=self.state_directory/f'{feature}.json'
        return json.loads(path.read_text()) if path.exists() else {'feature':feature,'insights':[],'explanations':[]}

    def write_state(self,feature,data):
        self._feature(feature)
        self.state_directory.mkdir(parents=True,exist_ok=True)
        path=self.state_directory/f'{feature}.json'
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        tmp.replace(path)

    @safe_result
    def steer_generation(self,feature,prompt,mode='multiply',strength=2.0,prompt_format='raw',
                         positions=None,max_new_tokens=32,seed=42):
        self._feature(feature)
        bounded_text(prompt,'prompt')
        if mode not in ('multiply','set'):
            raise ValueError('mode must be multiply or set; set strength=0 ablates the gate')
        if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=10:
            raise ValueError('strength must be finite and between 0 and 10')
        if prompt_format not in ('raw','chat'):
            raise ValueError('prompt_format must be raw or chat')
        _integer(max_new_tokens,'max_new_tokens',1,64)
        _integer(seed,'seed',0,2**32-1)
        with self._lock:  # Shared with inherited verify_hypothesis.
            result=self.subject().steer(feature,restore_probe_tokens(prompt),prompt_format,mode,strength,positions,max_new_tokens,seed)
        return {'feature':feature,'semantics':'Error-preserving intervention on actual input-dependent MOLT transform contribution, not a fixed SAE decoder direction.',**result}

    @safe_result
    def remember_insight(self,feature,insight,category='general'):
        bounded_text(insight,'insight',10000);bounded_text(category,'category',100)
        with self._state_lock:
            state=self.read_state(feature)
            if len(state['insights'])>=200:raise ValueError('Feature memory limit reached')
            entry={'timestamp':now(),'category':category,'insight':insight}
            state['insights'].append(entry);self.write_state(feature,state)
        return {'feature':feature,'status':'saved','insight':entry}

    @safe_result
    def recall_insights(self,feature,category=None,limit=20):
        _integer(limit,'limit',1,100)
        with self._state_lock:
            rows=self.read_state(feature)['insights']
        rows=[r for r in rows if category is None or r['category']==category]
        return {'feature':feature,'insights':list(reversed(rows))[:limit],'scope':'this feature and this run only'}

    @safe_result
    def save_explanations(self,feature,explanations):
        if not isinstance(explanations,list) or not 1<=len(explanations)<=10:
            raise ValueError('explanations must contain 1–10 draft strings')
        for text in explanations:bounded_text(text,'explanation',4000)
        with self._state_lock:
            state=self.read_state(feature)
            if len(state['explanations'])+len(explanations)>100:raise ValueError('Feature draft limit reached')
            entries=[{'timestamp':now(),'explanation':text} for text in explanations]
            state['explanations'].extend(entries);self.write_state(feature,state)
        return {'feature':feature,'status':'saved','drafts':entries,'note':'Saving drafts does not finalize the runner explanation or score held-out data.'}

    @safe_result
    def corpus_call(self,name,feature,**kwargs):
        self._feature(feature)
        if self.corpus is None:raise RuntimeError('Training corpus is not configured')
        return getattr(self.corpus,name)(feature,**kwargs)


TOOL_NAMES=['ping','echo','secret_word','get_server_info','list_latents','latent_info',
    'get_top_activating_examples','get_quantile_examples','get_non_activating_examples',
    'get_random_activating_examples','steer_generation','verify_hypothesis','score_explanation',
    'save_explanation','save_explanations','remember_insight','recall_insights']


def build_server(state):
    from mcp.server.mcpserver import MCPServer
    server=MCPServer('molt-agentic-full-training')

    @server.tool()
    def ping(feature:int)->dict:
        """Health check for the assigned feature (upstream ping)."""
        state._feature(feature);return {'feature':feature,'message':'pong','timestamp':now()}

    @server.tool()
    def echo(feature:int,message:str)->dict:
        """Echo text for protocol debugging; control-token strings are escaped."""
        state._feature(feature);bounded_text(message,'message')
        return _safe_strings({'feature':feature,'echo':message})

    @server.tool()
    def secret_word(feature:int)->dict:
        """Upstream debug handshake; returns its public test word, not a credential."""
        state._feature(feature);return {'feature':feature,'secret_word':'pineapple'}

    @server.tool()
    def get_server_info(feature:int)->dict:
        """Describe full tool availability, training-only scoring and MOLT-specific adaptations."""
        state._feature(feature)
        return {'feature':feature,'tools':TOOL_NAMES,'model':state.model_name,'layer':state.layer,
                'scope':'Training conversations and self-authored probes only. Memories/drafts isolated per feature and run.',
                'training_corpus_configured':state.corpus is not None,'training_scorer_configured':state.scorer is not None,
                'disabled':['get_delphi_explanation: would reveal blinded baseline'],
                'score_explanation':'Training-only feedback, not held-out evaluation; see returned scorer protocol.',
                'steering':'Actual context-dependent MOLT transform intervention; multiply=1 is no-op, set=0 ablates.',
                'fixed_decoder_direction':'Not applicable: MOLT output direction depends on the current input.'}

    @server.tool()
    def list_latents(feature:int)->dict:
        """Return the assigned feature only; cross-feature investigation is intentionally isolated."""
        state._feature(feature);return {'feature':feature,'latents':[{'feature':feature,'layer':state.layer}]}

    @server.tool()
    def latent_info(feature:int)->dict:
        """Training-window gate information. MOLT has no fixed SAE decoder direction; use contextual steering logits."""
        return state.get_latent_info(feature)

    @server.tool()
    def get_top_activating_examples(feature:int,count:int=5,offset:int=0)->dict:
        """Highest activating corpus training windows, paginated by offset; never held-out examples."""
        return state.corpus_call('positive_examples',feature,count=count,offset=offset,selection='top')

    @server.tool()
    def get_quantile_examples(feature:int,quantiles:list[float]|None=None,examples_per_quantile:int=2)->dict:
        """Read activation quantiles from the training corpus only; inspect returned population semantics."""
        quantiles=[.25,.5,.75,.95] if quantiles is None else quantiles
        if not 1<=len(quantiles)<=5:raise ValueError('Use 1–5 quantiles')
        return {'feature':feature,'groups':[state.corpus_call('positive_examples',feature,count=examples_per_quantile,selection='quantile',quantile=q) for q in quantiles]}

    @server.tool()
    def get_non_activating_examples(feature:int,count:int=5,seed:int=42)->dict:
        """Sample zero-gate windows from training conversations only, for negative evidence."""
        return state.corpus_call('negative_examples',feature,count=count,seed=seed)

    @server.tool()
    def get_random_activating_examples(feature:int,count:int=5,seed:int=42)->dict:
        """Sample activating windows from training conversations only."""
        return state.corpus_call('positive_examples',feature,count=count,seed=seed,selection='random')

    @server.tool()
    def verify_hypothesis(feature:int,texts:list[str],prompt_format:str='raw')->dict:
        """Actual Qwen3/MOLT gates for 1–8 invented probes. Known ⟪think⟫-style displays restore subject control tokens. Returns escaped text and numeric IDs."""
        return state.verify_hypothesis(feature,texts,prompt_format)

    @server.tool()
    def steer_generation(feature:int,prompt:str,mode:str='multiply',strength:float=2.0,
                         prompt_format:str='raw',positions:list[int]|None=None,max_new_tokens:int=32,seed:int=42)->dict:
        """Compare deterministic baseline/steered generations and first-token logits. multiply=1 no-op; set=0 ablates. Intervene on selected prompt positions (None=all) and every generated position. Actual context-dependent MOLT contribution; max64 new tokens."""
        return state.steer_generation(feature,prompt,mode,strength,prompt_format,positions,max_new_tokens,seed)

    @server.tool()
    async def score_explanation(feature:int,explanation:str)->dict:
        """Score a hypothesis using training-only feedback. Never accesses held-out evaluation or baseline labels; inspect returned scoring protocol."""
        state._feature(feature);bounded_text(explanation,'explanation',4000)
        if state.scorer is None:raise RuntimeError('Training-only scorer is not configured')
        result=state.scorer.score_explanation(feature,explanation)
        if inspect.isawaitable(result):result=await result
        return _safe_strings(result)

    @server.tool()
    def save_explanation(feature:int,explanation:str)->dict:
        """Persist one draft explanation in this feature's run-local state; does not finalize the agent."""
        return state.save_explanations(feature,[explanation])

    @server.tool()
    def save_explanations(feature:int,explanations:list[str])->dict:
        """Persist 1–10 alternative draft explanations for this feature only."""
        return state.save_explanations(feature,explanations)

    @server.tool()
    def remember_insight(feature:int,insight:str,category:str='general')->dict:
        """Persist an observation/hypothesis for this feature and run only."""
        return state.remember_insight(feature,insight,category)

    @server.tool()
    def recall_insights(feature:int,category:str|None=None,limit:int=20)->dict:
        """Recall this feature's own saved insights; no other features or baseline notes are accessible."""
        return state.recall_insights(feature,category,limit)

    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--inspection-url',required=True)
    parser.add_argument('--scoring-url',default='http://127.0.0.1:8002/v1')
    parser.add_argument('--scoring-model',default='qwen3.8-27b-nvfp4')
    parser.add_argument('--transport',choices=['stdio','streamable-http'],default='streamable-http')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8004)
    args=parser.parse_args()
    from molt_agentic_training import TrainingCorpus
    from tools.molt_scorer_runtime import load_scorer_runtime
    runtime=load_scorer_runtime(args.run_directory)
    if runtime is None and (args.scoring_url.rstrip('/') != 'http://127.0.0.1:8002/v1' or args.scoring_model != 'qwen3.8-27b-nvfp4'):
        raise ValueError('Training scorer is pinned to Qwen3.8-27B on local endpoint8002')
    corpus=TrainingCorpus(args.directory,args.selection,output_directory=args.run_directory)
    scorer=corpus
    state=FullTools(args.directory,args.selection,args.run_directory,inspection_url=args.inspection_url,corpus=corpus,scorer=scorer)
    inventory={'tools':TOOL_NAMES,'upstream':'agentic-delphi origin/mcp_integration_JP',
               'disabled':{'get_delphi_explanation':'Blinded baseline explanations are forbidden'},
               'adaptations':{'latent_id':'feature:int; required on every tool','steering':'Input-dependent MOLT contribution, not a fixed decoder vector',
                'score_explanation':'Training-only feedback; held-out evaluation reserved for final scoring',
                'memory':'Isolated to run and assigned feature','echo':'True escaped echo instead of upstream fixed debug sentence'}}
    args.run_directory.mkdir(parents=True,exist_ok=True)
    (args.run_directory/'tool-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    server=build_server(state)
    if args.transport=='stdio':server.run(transport='stdio')
    else:server.run(transport='streamable-http',host=args.host,port=args.port)


if __name__=='__main__':main()
