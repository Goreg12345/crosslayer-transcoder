"""Run in local/autointerp-venv; all endpoint requests are mocked."""

import asyncio

import httpx
import pytest

pytest.importorskip("delphi", reason="Requires the isolated autointerp environment")
from tools.explain_molt_dashboard import ContextLimitError, Endpoint


def test_global_concurrency_limit_and_usage():
    async def run():
        active = peak = 0

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "[1]"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 3},
                },
            )

        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 3)
        await endpoint.client.aclose()
        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await asyncio.gather(
                *(
                    endpoint.generate(
                        [
                            {"role": "system", "content": "Classify"},
                            {"role": "user", "content": "Example 0: test"},
                        ],
                        max_tokens=32,
                    )
                    for _ in range(12)
                )
            )
            assert peak == endpoint.usage["peak_concurrency"] == 3
            assert endpoint.in_flight == 0
            assert endpoint.usage["requests"] == 12
            assert endpoint.usage["prompt_tokens"] == 1200
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("tokens,allowed", [(9000, True), (9900, False)])
def test_borderline_byte_bound_uses_exact_count_and_reserves_output(tokens, allowed):
    async def run():
        requests = []

        async def handler(request):
            import json

            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "label"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": tokens, "completion_tokens": 1},
                },
            )

        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 2, 10000)
        await endpoint.client.aclose()
        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        prompt = [
            {"role": "system", "content": "Explain"},
            {"role": "user", "content": "long text " * 1100},
        ]
        try:
            if allowed:
                await endpoint.generate(prompt)
                assert [x["max_tokens"] for x in requests] == [1, 384]
            else:
                with pytest.raises(ContextLimitError):
                    await endpoint.generate(prompt)
                assert [x["max_tokens"] for x in requests] == [1]
            assert endpoint.usage["context_probes"] == 1
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())


def test_permanent_http_failure_is_not_retried():
    async def run():
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(400, json={"error": "invalid request"})

        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 2)
        await endpoint.client.aclose()
        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await endpoint.generate([{"role": "system", "content": "Explain"}])
            assert len(requests) == 1
            assert endpoint.in_flight == 0
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())


def test_truncated_explanation_changes_format_preserves_evidence_and_cache(
    tmp_path, monkeypatch
):
    import json
    from crosslayer_transcoder.dashboard.pilot import load_response_cache, response_key
    import tools.explain_molt_dashboard as module

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    async def run():
        requests = []
        audit = tmp_path / "audit.jsonl"

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            retry = "at most three distinct examples" in body["messages"][0]["content"]
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "[EXPLANATION]: A word prefix."
                                if retry
                                else "word " * 100
                            },
                            "finish_reason": "stop" if retry else "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 2)
        await endpoint.client.aclose()
        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        prompt = [
            {"role": "system", "content": "Explain"},
            {"role": "user", "content": "All original evidence"},
        ]
        try:
            result = await endpoint.generate(
                prompt, task="explanation", audit_path=audit
            )
            assert len(requests) == 2
            assert requests[0]["messages"] != requests[1]["messages"]
            assert all(r["messages"][-1] == prompt[-1] for r in requests)
            assert prompt[0]["content"] == "Explain"
            assert result.cache_key == response_key(
                "explanation", requests[1]["messages"], 384
            )
            cache = load_response_cache(audit)
            assert cache == {result.cache_key: result.text}
            assert endpoint.usage["explanation_reformats"] == 1
            # A restart can replay the audited fallback without accepting truncation.
            again = await endpoint.generate(
                prompt, task="explanation", response_cache=cache
            )
            assert again.text == result.text
            assert len(requests) == 3
            assert endpoint.usage["cached_requests"] == 1
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())


def test_explanation_format_retries_are_bounded(monkeypatch):
    import json
    import tools.explain_molt_dashboard as module

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)

    async def run():
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "unfinished"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 2)
        await endpoint.client.aclose()
        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(module.IncompleteCompletionError):
                await endpoint.generate(
                    [{"role": "system", "content": "Explain"}], task="explanation"
                )
            assert len(requests) == 4
            assert len({r["messages"][0]["content"] for r in requests}) == 4
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())


def test_occupancy_snapshot_includes_active_requests_without_double_counting(
    monkeypatch,
):
    from types import SimpleNamespace
    import tools.explain_molt_dashboard as module

    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 3)
    try:
        clock[0] = 10
        endpoint._change_in_flight(1)
        clock[0] = 15
        endpoint._change_in_flight(1)
        clock[0] = 20
        snapshot = endpoint.usage_snapshot()
        assert snapshot["in_flight"] == 2
        assert snapshot["concurrency_limit"] == 3
        assert snapshot["in_flight_seconds_total"] == 15
        assert snapshot["average_in_flight"] == 0.75
        assert endpoint.usage_snapshot() == snapshot
        clock[0] = 25
        endpoint._change_in_flight(-1)
        clock[0] = 30
        endpoint._change_in_flight(-1)
        clock[0] = 40
        snapshot = endpoint.usage_snapshot()
        assert snapshot["in_flight"] == 0
        assert snapshot["in_flight_seconds_total"] == 30
        assert snapshot["measurement_seconds"] == 40
    finally:
        asyncio.run(endpoint.client.aclose())


def test_waiting_request_cancellation_preserves_slot_and_metrics():
    async def run():
        endpoint = Endpoint("test", "https://example.invalid/v1", "unused", 1)
        entered = asyncio.Event()
        release = asyncio.Event()
        await endpoint.client.aclose()

        async def handler(request):
            entered.set()
            await release.wait()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "[1]"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 4},
                },
            )

        endpoint.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        prompt = [
            {"role": "system", "content": "Classify"},
            {"role": "user", "content": "Example 0: test"},
        ]
        try:
            first = asyncio.create_task(
                endpoint.generate(prompt, task="binary", max_tokens=32)
            )
            await entered.wait()
            waiting = asyncio.create_task(
                endpoint.generate(prompt, task="binary", max_tokens=32)
            )
            await asyncio.sleep(0)
            snapshot = endpoint.usage_snapshot()
            assert snapshot["in_flight"] == snapshot["waiting_for_slot"] == 1
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            assert endpoint.usage_snapshot()["waiting_for_slot"] == 0
            release.set()
            await first
            await endpoint.generate(prompt, task="binary", max_tokens=32)
            snapshot = endpoint.usage_snapshot()
            assert snapshot["in_flight"] == snapshot["waiting_for_slot"] == 0
            assert snapshot["request_latency_observations"] == 2
            assert snapshot["request_latency_seconds_total"] > 0
            assert snapshot["in_flight_seconds_total"] > 0
            assert snapshot["request_failures"] == 0
        finally:
            await endpoint.client.aclose()

    asyncio.run(run())
