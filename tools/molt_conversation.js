/* Explicit Qwen ChatML serialization. No history filtering or content trimming. */
(function(root){
  'use strict';
  const endings=new Set(['assistant_answer','assistant','assistant_think','continue_last','continue_thinking','closed']);
  const roles=new Set(['system','user','assistant','raw']);
  function validate(conversation){
    if(!conversation||!Array.isArray(conversation.turns)||!conversation.turns.length)throw Error('Add at least one turn.');
    if(conversation.turns.length>100)throw Error('Use at most 100 turns.');
    if(!endings.has(conversation.ending))throw Error('Unknown generation starting point.');
    conversation.turns.forEach((turn,i)=>{
      if(!turn||!roles.has(turn.role))throw Error(`Turn ${i+1}: unknown role.`);
      if(!Array.isArray(turn.segments)||!turn.segments.length)throw Error(`Turn ${i+1}: add a text or thinking segment.`);
      turn.segments.forEach(segment=>{
        if(!segment||!['text','think'].includes(segment.kind)||typeof segment.text!=='string')throw Error(`Turn ${i+1}: invalid segment.`);
      });
    });
    if(conversation.ending==='continue_thinking'){
      const last=conversation.turns.at(-1);
      if(last.role!=='assistant'||last.segments.at(-1).kind!=='think')throw Error('To continue thinking, end with an assistant turn whose final segment is thinking.');
    }
    return conversation;
  }
  function serialize(conversation){
    validate(conversation);
    const lastIndex=conversation.turns.length-1;
    let text='';
    conversation.turns.forEach((turn,i)=>{
      const raw=turn.role==='raw';
      if(!raw)text+=`<|im_start|>${turn.role}\n`;
      turn.segments.forEach((segment,j)=>{
        if(segment.kind==='think'){
          text+='<think>\n'+segment.text;
          const open=i===lastIndex&&j===turn.segments.length-1&&conversation.ending==='continue_thinking';
          if(!open)text+='\n</think>\n\n';
        }else text+=segment.text;
      });
      const continueLast=i===lastIndex&&['continue_last','continue_thinking'].includes(conversation.ending);
      if(!raw&&!continueLast)text+='<|im_end|>\n';
    });
    if(conversation.ending==='assistant_answer')text+='<|im_start|>assistant\n<think>\n\n</think>\n\n';
    if(conversation.ending==='assistant')text+='<|im_start|>assistant\n';
    if(conversation.ending==='assistant_think')text+='<|im_start|>assistant\n<think>\n';
    return text;
  }
  const api={validate,serialize};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.MoltConversation=api;
})(typeof globalThis!=='undefined'?globalThis:this);
