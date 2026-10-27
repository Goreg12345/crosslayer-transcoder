"""Refresh corrected-run dashboard views without issuing inference requests."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main(run):
    root=Path(__file__).resolve().parents[1];run=run.resolve()
    folder=run/'observability';folder.mkdir(exist_ok=True)
    path=folder/'publisher.json'
    state=json.loads(path.read_text()) if path.exists() else {'steps':{}}
    steps=[('molt_dashboard',30,[str(root/'local/autointerp-venv/bin/python'),'tools/publish_molt_corrected_dashboard.py','--run',str(run),'--dashboard',str(root/'results/molt-qwen-dashboard-campaign-layer22')]),
           ('trajectories',60,[str(root/'local/autointerp-venv/bin/python'),'tools/render_molt_trajectory_explorer.py','--root',str(run)]),
           ('inspect',120,[str(root/'local/inspect-view-venv/bin/python'),'tools/export_molt_corrected_inspect.py','--run',str(run),'--storage',str(run/'observability/inspect')])]
    with (folder/'publisher.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for name,interval,command in steps:
            if time.time()-state['steps'].get(name,{}).get('completed_unix',0)<interval:continue
            started=time.time()
            try:
                with (folder/f'{name}-publisher.log').open('a') as log:
                    outcome=subprocess.run(command,cwd=root,env=dict(os.environ,TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(root)),stdout=log,stderr=subprocess.STDOUT,timeout=600)
                result=dict(status='ok' if outcome.returncode==0 else 'error',returncode=outcome.returncode)
            except Exception as error:result=dict(status='error',error=f'{type(error).__name__}: {error}')
            result.update(completed_unix=time.time(),seconds=time.time()-started)
            state['steps'][name]=result
            state.update(updated_at=datetime.now(timezone.utc).isoformat(),status='ok' if all(v['status']=='ok' for v in state['steps'].values()) else 'error')
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
            print(json.dumps(dict(step=name,**result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/molt-agentic-delphi-v3-32-context128'))
    main(parser.parse_args().run)
