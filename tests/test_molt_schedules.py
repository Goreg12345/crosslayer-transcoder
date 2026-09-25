from types import SimpleNamespace

import pytest
import torch

from crosslayer_transcoder.model.clt_lightning import MoltModule
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer, DimensionwiseOutputStandardizer


def build_molt():
    return Molt(d_acts=2, N=1, ranks=[1],
                nonlinearity=JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=1),
                input_standardizer=DimensionwiseInputStandardizer(n_layers=1, activation_dim=2),
                output_standardizer=DimensionwiseOutputStandardizer(n_layers=1, activation_dim=2))


def make_module(**kwargs):
    module = MoltModule(model=build_molt(), learning_rate=4e-5,
                        lambda_sparsity=1.5e-4, **kwargs)
    module._trainer = SimpleNamespace(max_steps=100, global_step=0)
    module.log = lambda *args, **kwargs: None
    return module


@pytest.mark.parametrize('step,expected', [(0, 0), (40, 7.5e-5), (80, 1.5e-4), (90, 1.5e-4), (100, 1.5e-4), (110, 1.5e-4)])
def test_sparsity_reaches_peak_at_eighty_percent(step, expected):
    module = make_module(sparsity_ramp_fraction=0.8)
    module._trainer.global_step = step
    assert module.current_sparsity_penalty() == pytest.approx(expected)


def test_legacy_schedule_defaults_unchanged():
    module = make_module()
    module._trainer.global_step = 80
    assert module.current_sparsity_penalty() == pytest.approx(1.2e-4)
    assert isinstance(module.configure_optimizers(), torch.optim.Adam)


def test_lr_optimizer_steps_and_checkpoint_resume():
    module = make_module(lr_decay_start_fraction=0.8)
    config = module.configure_optimizers()
    assert config['lr_scheduler']['interval'] == 'step'
    opt, sched = config['optimizer'], config['lr_scheduler']['scheduler']
    for step in range(101):
        expected = 4e-5 if step <= 80 else 4e-5 * (100-step)/20
        assert opt.param_groups[0]['lr'] == pytest.approx(expected)
        if step == 90:
            opt_state, sched_state = opt.state_dict(), sched.state_dict()
        opt.step()
        sched.step()
    resumed = make_module(lr_decay_start_fraction=0.8).configure_optimizers()
    resumed_opt, resumed_sched = resumed['optimizer'], resumed['lr_scheduler']['scheduler']
    resumed_opt.load_state_dict(opt_state)
    resumed_sched.load_state_dict(sched_state)
    assert resumed_opt.param_groups[0]['lr'] == pytest.approx(2e-5)
    resumed_opt.step()
    resumed_sched.step()
    assert resumed_opt.param_groups[0]['lr'] == pytest.approx(1.8e-5)


@pytest.mark.parametrize('kwargs', [dict(sparsity_ramp_fraction=0), dict(sparsity_ramp_fraction=1.1), dict(lr_decay_start_fraction=1), dict(lr_decay_start_fraction=-0.1), dict(lr_decay_start_fraction=0.8, warmup_steps=10), dict(lr_decay_start_fraction=0.8, lr_decay_step=90)])
def test_invalid_schedules_rejected(kwargs):
    with pytest.raises(ValueError):
        make_module(**kwargs)
