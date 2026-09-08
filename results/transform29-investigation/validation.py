import json
from pathlib import Path
import numpy as np,torch
from datasets import Dataset
from transformers import AutoTokenizer
import urllib.request,time
class ExistingEngine:
    def inspect(self,prompt,fmt,feature):
        body=json.dumps(dict(prompt=prompt,format=fmt,feature=feature)).encode()
        request=urllib.request.Request('http://100.76.58.1:8766/api/inspect',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)
O=Path('results/transform29-investigation');torch.set_num_threads(4);e=ExistingEngine();t=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
r={'probes':[],'fresh':[]};
def save():(O/'validation.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
texts={
'code_comment':'def fibonacci(n):\n    # The small brown dog walked through the quiet garden and settled beneath the old apple tree.\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n',
'code_string':'text = "The small brown dog walked through the quiet garden and settled beneath the old apple tree."\nprint(text)',
'code_prose':'The function checks if n is less than or equal to one and returns n. Otherwise it returns the sum of fibonacci of n minus one and fibonacci of n minus two.',
'json':'{"name": "Alice", "age": 25, "city": "London", "description": "The small brown dog walked through the quiet garden and settled beneath the old apple tree."}',
'number_words':'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty',
'number_digits':'1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20',
'copy_cycle':' dog cat bird horse fish mouse'*8,
'novel_list':' dog cat bird horse fish mouse lion tiger elephant rabbit fox wolf bear giraffe zebra monkey panda eagle duck chicken turkey penguin whale dolphin shark salmon trout turtle frog snake lizard crocodile spider ant bee butterfly',
'sentence_repeat':' The small brown dog walked through the quiet garden.'*6,
'python2':'numbers = [1, 2, 3, 4, 5]\nsquares = [x * x for x in numbers]\nprint(sum(squares))',
'python3':'class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return "woof"',
'sql':'SELECT name, age FROM users WHERE age > 18 ORDER BY name LIMIT 10;',
'javascript':'const add = (a, b) => a + b;\nconst result = add(2, 3);\nconsole.log(result);',
'poem':'The silver moon above the sea\nReflects the dreams inside of me\nAnd every wave upon the shore\nReminds me of the days before',
'gibberish':'flarn gribble zork plimble snorf wuggles blarg frindle sprocket quizzle drang smool fleeb glorp snazzle wibble',
}
for name,text in texts.items():
 data=e.inspect('<|im_start|>user\n'+text+'<|im_end|>\n','raw',29);z=np.array([x['gate'] for x in data['tokens'][3:-2]])
 r['probes'].append(dict(name=name,rate=float((z>0).mean()),n=len(z),tokens=data['tokens']));print(name,round((z>0).mean(),3),flush=True);save()
# Independent contiguous replication, excluding the 128 exploratory conversations.
ds=Dataset.from_file('/home/georg/.cache/huggingface/datasets/HuggingFaceH4___ultrachat_200k/default/0.0.0/8049631c405ae6576f93f445c6b8166f76f5505a/ultrachat_200k-test_sft.arrow')
allids=[];allg=[];offsets=[0]
for i in range(128,192):
 ids=t.apply_chat_template(ds[i]['messages'],tokenize=True,add_generation_prompt=False)[:1024]
 data=e.inspect(t.decode(ids),'raw',29);assert data['ids']==ids
 z=np.array([x['gate'] for x in data['tokens']]);allids.extend(ids);allg.extend(z);offsets.append(len(allids))
 r['fresh'].append(dict(conversation=i,n=len(ids),active=int((z>0).sum())))
 if i%8==7:print('FRESH',i,flush=True)
fresh=O/'fresh';fresh.mkdir(exist_ok=True);g=np.array(allg);pos=np.where(g>0)[0]
np.savez_compressed(fresh/'activations.npz',tokens=np.array(allids),offsets=np.array(offsets),positions=pos,features=np.full(len(pos),29),values=g[pos]);(fresh/'vocab.json').write_text(json.dumps({str(i):t.decode([i]) for i in set(allids)},ensure_ascii=False));save()
print('VALIDATION DONE',flush=True)
