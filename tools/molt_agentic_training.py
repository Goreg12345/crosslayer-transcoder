"""Training-only corpus exploration and actual Delphi candidate diagnostics.

The benchmark's held-out evidence is never scored here. Internal diagnostics are
adaptive training feedback, not estimates of generalization or final-test scores.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np

from crosslayer_transcoder.dashboard.explanation_samples import index_features
from crosslayer_transcoder.dashboard.token_display import escape_added_tokens

ENDPOINT = 'http://127.0.0.1:8002/v1'
MODEL = 'qwen3.8-27b-nvfp4'


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


class TrainingCorpus:
    def __init__(self, directory, selection_path, evidence_directory=None, *,
                 output_directory=None, diagnostic_size=16):
        self.directory = Path(directory)
        self.evidence_directory = Path(evidence_directory or self.directory/'autointerp-layer')
        self.output_directory = Path(output_directory or Path(selection_path).parent)
        from tools.molt_scorer_runtime import load_scorer_runtime
        self.scorer_runtime=load_scorer_runtime(self.output_directory)
        self.diagnostic_size = _integer(diagnostic_size,'diagnostic_size',1,32)
        selection = json.loads(Path(selection_path).read_text())
        self.features = frozenset(row['id'] for row in selection['selected'])
        manifest_path = self.evidence_directory/'manifest.json'
        settings = json.loads(manifest_path.read_text())['settings']
        metadata = json.loads((self.directory/'metadata.json').read_text())
        self.layer = metadata['layer']
        for key in ('layer','checkpoint','global_step'):
            if settings.get(key) != metadata.get(key):
                raise ValueError(f'Corpus and evidence {key} mismatch')
        train = set(settings['train_conversations'])
        heldout = set(settings['test_conversations'])
        if not train or not heldout or train & heldout:
            raise ValueError('Training and held-out conversation sets must be nonempty/disjoint')
        self._initial = {}
        evidence_hashes = {}
        for feature in self.features:
            path = self.evidence_directory/'evidence'/f'{feature}.json'
            evidence = json.loads(path.read_text())
            self._initial[feature] = evidence['train']
            heldout.update(r['conversation'] for group in ('test','negatives') for r in evidence[group])
            evidence_hashes[str(feature)] = _digest(path)
        if train & heldout:
            raise ValueError('Benchmark held-out evidence appears in training conversation set')
        self.train_conversations = frozenset(train)
        self.test_conversations = frozenset(heldout)
        cache_sha256 = _digest(self.directory/'activations.npz')
        if settings.get('cache_sha256') and settings['cache_sha256'] != cache_sha256:
            raise ValueError('Activation cache differs from frozen campaign manifest')
        self.provenance = dict(cache_sha256=cache_sha256,vocab_sha256=_digest(self.directory/'vocab.json'),
            manifest_sha256=_digest(manifest_path),
            selection_sha256=_digest(selection_path),evidence_sha256=evidence_hashes,
            scope='training-only; adaptive diagnostic; never final held-out evaluation',
            subject_checkpoint=metadata['checkpoint'],subject_global_step=metadata['global_step'],
            scorer_model=MODEL,scorer_endpoint=ENDPOINT,diagnostic_size=diagnostic_size,
            seed=42,display_policy='escape every Qwen added token in all scorer messages')
        if self.scorer_runtime is not None:
            self.provenance.update(scorer_model=self.scorer_runtime['model'],scorer_endpoint=self.scorer_runtime['base_url'],
                                   scorer_context_limit=self.scorer_runtime['context_limit'],scorer_runtime_sha256=self.scorer_runtime['runtime_sha256'])
        # The archive is read once per server. Feature indices avoid repeated full scans.
        with np.load(self.directory/'activations.npz') as archive:
            self.cache = {key: archive[key] for key in ('tokens','offsets','features','positions','values')}
        index_features(self.cache,metadata['n_features'])
        if any(c < 0 or c >= len(self.cache['offsets'])-1 for c in train | heldout):
            raise ValueError('Conversation index outside corpus cache')
        self.vocab = {str(k):escape_added_tokens(v) for k,v in json.loads((self.directory/'vocab.json').read_text()).items()}
        self._arrays, self._negative_rows, self._diagnostics = {}, {}, {}
        self._lock = threading.RLock()
        self._endpoint = None
        self._score_locks = {}
        for feature, rows in self._initial.items():
            for row in rows:
                self._check_window(row['conversation'],row['start'],row['stop'])
                if row['tokens'] != self.cache['tokens'][row['start']:row['stop']].astype(int).tolist():
                    raise ValueError('Initial explanation evidence token IDs disagree with corpus')

    def _feature(self, feature):
        if type(feature) is not int or feature not in self.features:
            raise ValueError('Feature is not in the permitted cohort')
        return feature

    def _check_window(self, conversation, start, stop):
        if conversation not in self.train_conversations or conversation in self.test_conversations:
            raise ValueError('Only training conversations may be exposed or scored')
        offsets = self.cache['offsets']
        if not int(offsets[conversation]) <= start < stop <= int(offsets[conversation+1]):
            raise ValueError('Window leaves its permitted training conversation')

    def _data(self, feature):
        self._feature(feature)
        with self._lock:
            if feature not in self._arrays:
                order,cuts = self.cache['feature_index']
                selected = order[cuts[feature]:cuts[feature+1]]
                positions = self.cache['positions'][selected].astype(np.int64)
                values = self.cache['values'][selected].astype(np.float32)
                conversations = np.searchsorted(self.cache['offsets'],positions,side='right')-1
                keep = np.isin(conversations,list(self.train_conversations)) & (values>0)
                positions,values,conversations = positions[keep],values[keep],conversations[keep]
                sort = np.argsort(positions,kind='stable')
                positions,values,conversations = positions[sort],values[sort],conversations[sort]
                if not np.all(np.isfinite(values)) or len(np.unique(positions))!=len(positions):
                    raise ValueError('Non-finite or duplicate feature activations in cache')
                ranked = np.argsort(-values,kind='stable')
                self._arrays[feature] = positions,values,conversations,ranked
            return self._arrays[feature]

    def _window(self, feature, conversation, start, stop):
        self._check_window(conversation,start,stop)
        positions,values,_,_ = self._data(feature)
        left,right = np.searchsorted(positions,[start,stop])
        ids = self.cache['tokens'][start:stop].astype(int).tolist()
        activations = np.zeros(len(ids),dtype=np.float32)
        activations[positions[left:right]-start] = values[left:right]
        return dict(conversation=int(conversation),start=int(start),stop=int(stop),
                    tokens=ids,str_tokens=[self.vocab[str(i)] for i in ids],activations=activations.tolist())

    def _positive(self, feature, index):
        positions,values,conversations,_ = self._data(feature)
        conversation = int(conversations[index])
        stop = int(positions[index])+1
        row = self._window(feature,conversation,max(int(self.cache['offsets'][conversation]),stop-64),stop)
        row['anchor_activation'] = float(values[index])
        return row

    @staticmethod
    def _public(row):
        tokens = [escape_added_tokens(t) for t in row['str_tokens']]
        return dict(conversation=row['conversation'],start=row['start'],stop=row['stop'],
                    coordinate_system='start/stop are global cache token offsets, not within-conversation positions',
                    sampling_note=('Positive windows end at an active token by construction; window endings are not causal evidence.' if any(row['activations']) else 'Zero-gate window from training conversation; window boundaries are sampling artifacts.'),
                    evidence_scope='training-only corpus window',tokens=tokens,str_tokens=tokens,
                    token_ids=list(row['tokens']),activations=list(row['activations']),
                    max_activation=max(row['activations']),anchor_activation=row.get('anchor_activation'),
                    marked_text=''.join(f'<<{t}>>' if a>0 else t for t,a in zip(tokens,row['activations'])))

    def initial_examples(self, feature):
        self._feature(feature)
        rows=[self._window(feature,r['conversation'],r['start'],r['stop']) for r in self._initial[feature]]
        return dict(feature=feature,scope='training',selection='original explanation-training windows',
                    total_available=len(rows),examples=[self._public(r) for r in rows])

    def positive_examples(self, feature, count=5, offset=0, selection='top', quantile=.5, seed=42):
        _integer(count,'count',1,64);_integer(offset,'offset',0,10_000_000);_integer(seed,'seed',0,2**32-1)
        positions,_,_,ranked = self._data(feature)
        if selection == 'random':
            order = np.random.default_rng(seed+feature).permutation(len(positions))
        elif selection == 'quantile':
            if isinstance(quantile,bool) or not isinstance(quantile,(int,float)) or not math.isfinite(quantile) or not 0<=quantile<=1:
                raise ValueError('quantile must be in [0,1]')
            ascending = ranked[::-1]
            center = round(quantile*max(0,len(ascending)-1))
            start = max(0,min(max(0,len(ascending)-count),center-count//2))
            order = ascending[start:]
        elif selection == 'top':
            order = ranked
        else:
            raise ValueError('selection must be top, random, or quantile')
        rows = [self._positive(feature,int(i)) for i in order[offset:offset+count]]
        return dict(feature=feature,scope='training',selection=selection,total_available=len(positions),
                    quantile_population='all positive token-gate values in permitted training conversations',
                    examples=[self._public(r) for r in rows])

    def _negatives(self, feature):
        self._feature(feature)
        with self._lock:
            if feature in self._negative_rows:
                return self._negative_rows[feature]
            positions,_,_,_ = self._data(feature)
            rng = np.random.default_rng(42+feature)
            rows=[]
            for conversation in rng.permutation(sorted(self.train_conversations)):
                conversation=int(conversation)
                lo,hi=map(int,self.cache['offsets'][conversation:conversation+2])
                if hi-lo<16:
                    continue
                first=min(lo+64,hi)
                stops=np.unique(np.concatenate(([hi],rng.integers(first,hi+1,size=64))))
                rng.shuffle(stops)
                for stop in stops:
                    stop=int(stop);start=max(lo,stop-64)
                    left,right=np.searchsorted(positions,[start,stop])
                    if left==right:
                        row=self._window(feature,conversation,start,stop)
                        assert not any(row['activations'])
                        rows.append(row)
                        break
                if len(rows)>=128:
                    break
            self._negative_rows[feature]=rows
            return rows

    def negative_examples(self, feature, count=5, offset=0, seed=42):
        _integer(count,'count',1,64);_integer(offset,'offset',0,10_000_000);_integer(seed,'seed',0,2**32-1)
        rows=self._negatives(feature)
        order=np.random.default_rng(seed+feature).permutation(len(rows))
        return dict(feature=feature,scope='training',selection='deterministic zero-gate windows',
                    total_available=len(rows),examples=[self._public(rows[int(i)]) for i in order[offset:offset+count]])

    def diagnostic_samples(self, feature):
        """Frozen training feedback split, never the benchmark's test/negative rows."""
        self._feature(feature)
        with self._lock:
            if feature not in self._diagnostics:
                positions,_,conversations,ranked=self._data(feature)
                unique=[];seen=set()
                for index in ranked:
                    conversation=int(conversations[index])
                    if conversation not in seen:
                        if int(positions[index])+1-int(self.cache['offsets'][conversation])>=16:
                            unique.append(int(index));seen.add(conversation)
                if len(unique)<self.diagnostic_size:
                    raise ValueError('Insufficient distinct training-positive conversations for diagnostic')
                chosen=np.linspace(0,len(unique)-1,self.diagnostic_size,dtype=int)
                positive=[self._positive(feature,unique[int(i)]) for i in chosen]
                used={r['conversation'] for r in positive}
                negative=[r for r in self._negatives(feature) if r['conversation'] not in used][:self.diagnostic_size]
                if len(negative)<self.diagnostic_size:
                    raise ValueError('Insufficient disjoint zero-gate training-negative windows for diagnostic')
                initial=[self._window(feature,r['conversation'],r['start'],r['stop']) for r in self._initial[feature]]
                samples=dict(train=initial,test=positive,negatives=negative)
                for rows in samples.values():
                    for row in rows:self._check_window(row['conversation'],row['start'],row['stop'])
                self._diagnostics[feature]=samples
            return deepcopy(self._diagnostics[feature])

    async def score_explanation(self, feature, explanation, scorers=('detection','fuzz')):
        """Run real pinned Delphi scorers on fixed training-only diagnostic cases."""
        self._feature(feature)
        if not isinstance(explanation,str) or not explanation.strip() or len(explanation)>4000:
            raise ValueError('explanation must be nonempty and at most 4000 characters')
        if isinstance(scorers,str):scorers=[scorers]
        if not scorers or len(set(scorers))!=len(scorers) or set(scorers)-{'detection','fuzz'}:
            raise ValueError('Supported diagnostics: detection and fuzz only. Embedding and loss_recovery are unavailable; they are not substituted.')
        from tools.explain_molt_dashboard import Endpoint, atomic_json
        from tools.run_molt_pilot import TaskClient, make_record
        from tools.run_molt_layer_explanations import score_method
        from crosslayer_transcoder.dashboard.pilot import load_response_cache
        from crosslayer_transcoder.dashboard.output_limits import RequestFailureBudget
        explanation=escape_added_tokens(explanation.strip())
        samples=self.diagnostic_samples(feature)
        identity=dict(feature=feature,explanation=explanation,scorers=sorted(scorers),samples=samples,provenance=self.provenance)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        root=self.output_directory/'training-scores'/str(feature)
        root.mkdir(parents=True,exist_ok=True)
        path=root/f'{key}.json';audit=root/f'{key}.jsonl'
        lock=self._score_locks.setdefault(key,asyncio.Lock())
        async with lock:
            if path.exists():
                result=json.loads(path.read_text())
                if result.get('status')=='complete':return result
            if self._endpoint is None:
                if self.scorer_runtime is None:
                    self._endpoint=Endpoint(MODEL,ENDPOINT,'local-not-required',16,context_limit=262144)
                else:
                    from tools.molt_scorer_runtime import training_endpoint
                    self._endpoint=training_endpoint(Endpoint,self.scorer_runtime)
            record=make_record(self.layer,feature,samples)
            record.explanation=explanation
            cache=load_response_cache(audit)
            budget=RequestFailureBudget()
            def client(task):
                return TaskClient(self._endpoint,task,audit,cache,output_limits=budget)
            methods={}
            for scorer in scorers:
                methods[scorer]=await score_method(scorer,record,client,feature)
            # Attach only training window identities to feedback; highlighted wrong
            # predictions already include the exact shown text from actual Delphi.
            lookup={tuple(r['str_tokens']):r for group in ('test','negatives') for r in samples[group]}
            for result in methods.values():
                for row in result.get('rows',[]):
                    source=lookup.get(tuple(row['str_tokens']))
                    if source:
                        row['training_window']={k:source[k] for k in ('conversation','start','stop')}
                result['wrong_examples']=[r for r in result.get('rows',[]) if r.get('prediction') is not None and not r.get('correct')]
            result=dict(feature=feature,status='complete' if all(m['status']=='complete' for m in methods.values()) else 'incomplete',
                        scope='training_diagnostic_not_final_test',explanation=explanation,
                        warning='Adaptive training feedback only. These scores are not final held-out benchmark metrics.',
                        cases_per_method=2*self.diagnostic_size,methods=methods,provenance=self.provenance,
                        diagnostic_sha256=key,audit=str(audit),
                        positive_windows=[{k:r[k] for k in ('conversation','start','stop')} for r in samples['test']],
                        negative_windows=[{k:r[k] for k in ('conversation','start','stop')} for r in samples['negatives']])
            atomic_json(path,result)
            return result

    async def aclose(self):
        if self._endpoint is not None:
            await self._endpoint.aclose()
