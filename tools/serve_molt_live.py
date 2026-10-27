"""Read-only corrected-run static server with checkpoint/stream SSE updates."""
import argparse
from datetime import datetime,timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import unquote,urlsplit


def now():return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    try:return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except (ValueError,AttributeError):return 0


def preview(value,limit=1800):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return text[:limit]+('… [full result in trajectory]' if len(text)>limit else '')


def project_agent(agent):
    messages=agent.get('messages',[]);events=agent.get('events',[])
    assistant=next((m for m in reversed(messages) if m.get('role')=='assistant'),{})
    model=next((e for e in reversed(events) if e.get('type')=='model_response'),{})
    tools=[e for e in events if e.get('type')=='tool_result'][-3:]
    usage=model.get('response',{}).get('usage')
    settings=model.get('request_settings',{})
    return dict(status=agent.get('status','pending'),updated_at=agent.get('updated_at') or agent.get('finished_at'),
        reasoning=assistant.get('reasoning_content') or assistant.get('reasoning') or '',text=assistant.get('content') or '',
        tool_calls=assistant.get('tool_calls') or [],usage=usage,prompt_tokens=usage.get('prompt_tokens') if usage else None,
        output_budget=settings.get('max_tokens'),context_limit=262144,turn=model.get('turn'),attempt=model.get('attempt'),
        last_tool_results=[dict(tool=e.get('tool'),at=e.get('at'),turn=e.get('turn'),tool_call_id=e.get('tool_call_id'),
            preview=preview(e.get('result')),arguments_preview=preview(e.get('arguments'),500)) for e in tools],
        model_responses=sum(e.get('type')=='model_response' for e in events),
        tool_result_count=sum(e.get('type')=='tool_result' for e in events),error=agent.get('error'))


class LiveStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.cache={};self.lock=threading.Lock();self.last_poll=0;self.current=None
        self.parse_counts={}

    def read(self,path,project=None):
        path=self.root/path
        try:stat=path.stat()
        except FileNotFoundError:return {}
        signature=(stat.st_mtime_ns,stat.st_size)
        previous=self.cache.get(path)
        if previous and previous[0]==signature:return previous[1]
        try:
            value=json.loads(path.read_text())
            if project:value=project(value)
        except (ValueError,OSError):return previous[1] if previous else {}
        self.cache[path]=(signature,value);self.parse_counts[path]=self.parse_counts.get(path,0)+1
        return value

    def snapshot(self,force=False):
        with self.lock:
            if not force and self.current is not None and time.monotonic()-self.last_poll<.5:return self.current
            selection=self.read(Path('selection.json'));run=self.read(Path('run.json'))
            epoch=timestamp(run.get('resumed_at') or run.get('started_at'));agents=[]
            for row in selection.get('selected',[]):
                feature=row['id'];checkpoint=self.read(Path(f'agents/{feature}.json'),project_agent)
                stream=self.read(Path(f'live-state/{feature}.json'))
                cp_time=timestamp(checkpoint.get('updated_at'));stream_time=timestamp(stream.get('updated_at'))
                stage=run.get('feature_stages',{}).get(str(feature))
                cp_status=checkpoint.get('status','pending')
                status=stage or cp_status
                if stage not in ('scoring','complete') and cp_time>=epoch:
                    status='explanation_complete' if cp_status=='complete' else cp_status
                state=dict(checkpoint,feature=feature,rank=row.get('rank'),cohort=row.get('cohort'),status=status,
                    source='checkpoint',checkpoint_updated_at=checkpoint.get('updated_at'),stage=stage,
                    sequence=None,stream_status=None,links=dict(trajectory=f'/features/{feature}.html',
                    raw_agent=f'/agents/{feature}.json',live_stream=f'/live-state/{feature}.json'))
                # Old streams never masquerade as current generation after restart.
                fresh=stream_time>=epoch and stream_time>cp_time
                if fresh and stream.get('feature')==feature:
                    for key in ('turn','attempt','request_id','reasoning','text','tool_calls','usage','prompt_tokens',
                                'output_budget','context_limit','updated_at','sequence','error'):
                        if key in stream:state[key]=stream[key]
                    state.update(source='stream',stream_status=stream.get('status'))
                    if stream.get('status')=='generating':state['status']='generating'
                    elif stream.get('status')=='stream_error':state['status']='stream_error'
                    elif stream.get('status')=='response_complete' and status not in ('scoring','complete','explanation_complete','error'):
                        state['status']='tool_use' if stream.get('tool_calls') else 'response_complete'
                    # Usage comes only from actual server stream stats; never estimate.
                    state['usage']=stream.get('usage')
                state.setdefault('reasoning','');state.setdefault('text','');state.setdefault('tool_calls',[])
                state.setdefault('last_tool_results',[]);agents.append(state)
            # Do not expose huge frozen source hashes/protocol bodies on every poll.
            run_view={k:v for k,v in run.items() if k in ('status','phase','started_at','resumed_at','updated_at','counts','errors','concurrency','scoring_concurrency','feature_stages')}
            self.current=dict(run=run_view,agents=agents,updated_at=now())
            self.last_poll=time.monotonic();return self.current


class Handler(SimpleHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def __init__(self,*args,store,**kwargs):
        self.store=store;super().__init__(*args,directory=str(store.root),**kwargs)

    def log_message(self,*args):pass

    def do_HEAD(self):
        path=unquote(urlsplit(self.path).path)
        if '..' in Path(path).parts or not (self.store.root/path.lstrip('/')).resolve().is_relative_to(self.store.root):
            self.send_error(403,'Path outside run directory');return
        return super().do_HEAD()

    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/live/snapshot':
            data=json.dumps(self.store.snapshot(),ensure_ascii=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        if path=='/api/live/events':return self.events()
        # Prevent both explicit traversal and symlink escape, unlike permissive static servers.
        if '..' in Path(path).parts or not (self.store.root/path.lstrip('/')).resolve().is_relative_to(self.store.root):
            self.send_error(403,'Path outside run directory');return
        return super().do_GET()

    def events(self):
        self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
        self.send_header('Cache-Control','no-cache, no-transform');self.send_header('Connection','keep-alive')
        self.send_header('X-Accel-Buffering','no');self.end_headers()
        previous={};previous_run=None;heartbeat=0
        def send(event,value):
            data=json.dumps(value,ensure_ascii=False,separators=(',',':'))
            self.wfile.write(f'event: {event}\ndata: {data}\n\n'.encode());self.wfile.flush()
        try:
            while True:
                snapshot=self.store.snapshot()
                run_key=json.dumps(snapshot['run'],sort_keys=True)
                if run_key!=previous_run:send('run',snapshot['run']);previous_run=run_key
                for state in snapshot['agents']:
                    key=json.dumps(state,sort_keys=True)
                    if previous.get(state['feature'])!=key:
                        send('update',state);previous[state['feature']]=key
                if time.monotonic()-heartbeat>=10:
                    self.wfile.write(b': heartbeat\n\n');self.wfile.flush();heartbeat=time.monotonic()
                time.sleep(1)
        except (BrokenPipeError,ConnectionResetError,TimeoutError):return


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bind',default='100.76.58.1');p.add_argument('--port',type=int,default=8771)
    args=p.parse_args();store=LiveStore(args.root)
    server=ThreadingHTTPServer((args.bind,args.port),partial(Handler,store=store));server.daemon_threads=True
    print(f'Live read-only dashboard: http://{args.bind}:{args.port}/live/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
