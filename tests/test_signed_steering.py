"""Signed interventions reverse contributions while legacy suppression still clamps."""
import pytest
import torch
from crosslayer_transcoder.dashboard.steering import SteeringEngine, target_gate
from tests.test_dashboard_steering import config, engine


def test_signed_targets_and_legacy_clamp():
    natural=torch.tensor([0.,2.])
    torch.testing.assert_close(target_gate(natural,-3,'add'),torch.tensor([0.,0.]))
    torch.testing.assert_close(target_gate(natural,-3,'add',True),torch.tensor([-3.,-1.]))
    torch.testing.assert_close(target_gate(natural,-2,'set',True),torch.tensor([-2.,-2.]))
    torch.testing.assert_close(target_gate(natural,-2,'multiply',True),torch.tensor([0.,-4.]))


def validating_engine():
    e=SteeringEngine.__new__(SteeringEngine)
    e.meta={'n_features':1};e.feature_stats=[{'mean_active':1.5}]
    e.tokenize=lambda prompt,fmt: {'ids':[1,2,3],'tokenization_id':'checked'}
    return e


@pytest.mark.parametrize('mode',['set','multiply'])
def test_negative_prompt_and_generated_amounts_require_opt_in(mode):
    e=validating_engine()
    body=dict(prompt='test',tokenization_id='checked',feature=0,mode=mode,
              strength_reference='raw',assignments=[{'position':2,'value':-2}],
              generated={'start':0,'end':1,'value':-3},max_new_tokens=4)
    with pytest.raises(ValueError,match='between 0 and 100'):e.validate(body)
    checked=e.validate(dict(**body,allow_negative=True))
    assert checked['allow_negative'] is True
    assert checked['assignments'][2]==-2 and checked['generated']['value']==-3


@pytest.mark.parametrize('invalid',[1,'true',None])
def test_signed_flag_requires_boolean(invalid):
    e=validating_engine()
    with pytest.raises(ValueError,match='must be a boolean'):
        e.validate(dict(prompt='test',tokenization_id='checked',feature=0,allow_negative=invalid))


def test_signed_add_reverses_edit_at_closed_gate_and_isolates_tokens(engine):
    with torch.no_grad():
        engine.gates.encoder.weight.zero_();engine.gates.encoder.bias.fill_(-1)
    layer=engine.model.model.layers[0]
    def edit(amount,signed):
        before,after,trace=[],[],[]
        first=layer.mlp.register_forward_hook(lambda m,a,o: before.append(o.detach().clone()))
        try:
            with torch.inference_mode(),engine.intervention(config(assignments={1:amount},allow_negative=signed),trace,[0]):
                last=layer.mlp.register_forward_hook(lambda m,a,o: after.append(o.detach().clone()))
                try:engine.model.model(input_ids=torch.tensor([[1,2,3]]))
                finally:last.remove()
        finally:first.remove()
        torch.testing.assert_close(before[0][:,[0,2]],after[0][:,[0,2]],atol=0,rtol=0)
        return after[0]-before[0],trace
    positive,_=edit(2,True);negative,trace=edit(-2,True);legacy,_=edit(-2,False)
    assert trace[0]['before']==0 and trace[0]['after']==-2 and trace[0]['delta_l2']>0
    torch.testing.assert_close(negative,-positive,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(legacy,torch.zeros_like(legacy),atol=0,rtol=0)
    assert not layer.mlp._forward_hooks and not layer.post_attention_layernorm._forward_pre_hooks
