import json
from argparse import Namespace

import numpy as np

from crosslayer_transcoder.dashboard.collection import GateCollection
from tools.collect_molt_layers import resume_batch_start
from tools.run_molt_interpretation_campaign import ORDER, Inventory
from tools.run_molt_layer_explanations import pipeline_settings


def test_campaign_covers_every_layer_once_and_puts_current_layer_first():
    assert len(ORDER) == len(set(ORDER)) == 36
    assert set(ORDER) == set(range(36))
    assert ORDER[:5] == [22, 0, 1, 10, 30]


def test_collection_resume_replays_only_missing_conversations(tmp_path):
    def record(i):
        return dict(
            tokens=np.array([i]),
            positions=np.array([0]),
            features=np.array([0]),
            values=np.array([1.0]),
        )

    a = GateCollection(tmp_path / "a", {"layer": 0})
    b = GateCollection(tmp_path / "b", {"layer": 1})
    a.append([record(i) for i in range(32)])
    b.append([record(i) for i in range(17)])
    assert resume_batch_start({0: a, 1: b}, 16) == 16
    # The shared model replays the aligned batch, but each store skips committed rows.
    for store in (a, b):
        store.append([record(i) for i in range(16, 48) if i >= store.conversations])
        store.finish()
        with np.load(store.root / "activations.npz") as archive:
            np.testing.assert_array_equal(archive["tokens"], np.arange(48))
            np.testing.assert_array_equal(archive["offsets"], np.arange(49))


def test_other_layers_get_separate_provenance_and_no_pilot_import(tmp_path):
    reference = tmp_path / "reference.json"
    methods = {name: name for name in ("detection", "fuzz", "active_fuzz", "intruder")}
    reference.write_text(
        json.dumps(
            dict(
                settings=dict(
                    methods=methods, layer=22, checkpoint="old", n_features=2480
                )
            )
        )
    )
    args = Namespace(
        model="endpoint-model", base_url="endpoint", method_reference=reference
    )
    meta = dict(
        checkpoint="new",
        global_step=100000,
        tokens=10000000,
        layer=3,
        n_features=10,
        conversations=20,
    )
    settings, pilot = pipeline_settings(tmp_path / "layer3", meta, args, "new-cache")
    assert pilot is None
    assert settings["layer"] == 3 and settings["checkpoint"] == "new"
    assert settings["cache_sha256"] == "new-cache"
    assert settings["n_features"] == 10
    assert set(settings["methods"]) == set(methods)
    assert not set(settings["train_conversations"]) & set(
        settings["test_conversations"]
    )
    assert settings["scope"] == "all transforms in layer 3"


def test_inventory_requires_matching_complete_cache(tmp_path, monkeypatch):
    import tools.run_molt_interpretation_campaign as module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ORDER", [0])
    inventory = Inventory()
    monkeypatch.setattr(inventory, "step", lambda layer: 100000)
    root = module.sample_path(0)
    root.mkdir(parents=True)
    (root / "activations.npz").touch()
    (root / "vocab.json").write_text("{}")
    (root / "collection-progress.json").write_text(
        json.dumps(dict(complete=True, tokens=10000000))
    )
    meta = dict(
        tokens=10000000,
        global_step=100000,
        layer=22,
        checkpoint=str(module.checkpoint_path(0).resolve()),
    )
    (root / "metadata.json").write_text(json.dumps(meta))
    assert not inventory.rows()[0]["cached"]
    meta["layer"] = 0
    (root / "metadata.json").write_text(json.dumps(meta))
    assert inventory.rows()[0]["cached"]


def test_failed_layer_does_not_block_others_and_cooldown_survives_restart(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import tools.run_molt_interpretation_campaign as module

    monkeypatch.chdir(tmp_path)
    module.PORTAL.mkdir(parents=True)
    states = {0: "incomplete", 1: "pending"}
    fingerprints = {0: "first-cache", 1: "second-cache"}
    summary_path = module.sample_path(0) / "autointerp-layer/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"fingerprint": fingerprints[0]}))
    inventory = SimpleNamespace(
        read=lambda p: json.loads(p.read_text()) if p.exists() else {}
    )
    monkeypatch.setattr(module, "unit_active", lambda unit: False)

    def publish(*args, **kwargs):
        return [
            dict(
                layer=i,
                cached=True,
                api_state=states[i],
                interpretation_fingerprint=fingerprints[i],
            )
            for i in states
        ]

    calls = []

    def explain(layer, inventory):
        calls.append(layer)
        if layer == 0:
            raise module.LayerIncompleteError("One malformed explanation")
        states[layer] = "complete"

    monkeypatch.setattr(module, "publish", publish)
    monkeypatch.setattr(module, "explain_layer", explain)
    module.api_queue(inventory, once=True)
    assert calls == [0, 1]
    assert states == {0: "incomplete", 1: "complete"}
    retries = inventory.read(module.PORTAL / ".api-retries.json")
    assert retries["0"]["retry_after"] > module.time.time()
    module.api_queue(inventory, once=True)
    assert calls == [0, 1]  # Restart does not burn another three resumptions.

    # A changed sample must not inherit the previous sample's cooldown.
    fingerprints[0] = "new-cache"
    summary_path.write_text(json.dumps({"fingerprint": fingerprints[0]}))
    module.api_queue(inventory, once=True)
    assert calls == [0, 1, 0]

    # After the scheduled retry succeeds, remove its stale failure record.
    monkeypatch.setattr(module.time, "time", lambda: retries["0"]["retry_after"] + 100)
    monkeypatch.setattr(
        module,
        "explain_layer",
        lambda layer, inventory: states.update({layer: "complete"}),
    )
    module.api_queue(inventory, once=True)
    assert states[0] == "complete"
    assert inventory.read(module.PORTAL / ".api-retries.json") == {}
