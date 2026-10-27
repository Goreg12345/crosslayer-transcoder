"""Dispatch independent Delphi requests across replicas of the same model."""

import asyncio
import time

from delphi.clients.client import Client

from tools.explain_molt_dashboard import Endpoint


class EndpointPool(Client):
    def __init__(self, model, backends, key, context_limit=10000):
        super().__init__(model)
        if not isinstance(backends, list) or not backends:
            raise ValueError("Endpoint pool requires a nonempty list of backends")
        urls = set()
        for backend in backends:
            if not isinstance(backend, dict):
                raise TypeError("Each backend must be an object")
            url = backend.get("base_url")
            if not isinstance(url, str):
                raise TypeError("Backend base_url must be a string")
            url = url.rstrip("/")
            concurrency = backend.get("concurrency")
            if not url.startswith(("http://", "https://")) or url in urls:
                raise ValueError("Each backend must have a distinct HTTP base_url")
            if type(concurrency) is not int or concurrency < 1:
                raise ValueError("Backend concurrency must be a positive integer")
            urls.add(url)
        self.backends = [
            Endpoint(model, row["base_url"], key, row["concurrency"], context_limit)
            for row in backends
        ]
        self.names = [row.get("name", row["base_url"]) for row in backends]
        self.urls = [endpoint.url for endpoint in self.backends]
        self.url = self.urls[0] if len(self.urls) == 1 else None
        self.concurrency_limit = sum(e.concurrency_limit for e in self.backends)
        self.started_at = time.monotonic()
        self.waiting = 0
        self.peak_concurrency = 0
        self.available = asyncio.Queue()
        # Interleave initial slots, then return each slot when its request finishes.
        # Faster replicas naturally take more work without statically splitting layers.
        for slot in range(max(e.concurrency_limit for e in self.backends)):
            for endpoint in self.backends:
                if slot < endpoint.concurrency_limit:
                    self.available.put_nowait(endpoint)
        for endpoint in self.backends:
            endpoint.on_in_flight_change = self._request_changed

    def _request_changed(self):
        self.peak_concurrency = max(
            self.peak_concurrency, sum(e.in_flight for e in self.backends)
        )

    @property
    def usage(self):
        keys = set().union(*(e.usage for e in self.backends))
        result = {key: sum(e.usage.get(key, 0) for e in self.backends) for key in keys}
        result["max_prompt_tokens"] = max(
            e.usage["max_prompt_tokens"] for e in self.backends
        )
        result["peak_concurrency"] = self.peak_concurrency
        return result

    def usage_snapshot(self):
        snapshots = [endpoint.usage_snapshot() for endpoint in self.backends]
        elapsed = time.monotonic() - self.started_at
        occupied = sum(row["in_flight_seconds_total"] for row in snapshots)
        return dict(
            **self.usage,
            concurrency_limit=self.concurrency_limit,
            in_flight=sum(row["in_flight"] for row in snapshots),
            waiting_for_slot=self.waiting
            + sum(row["waiting_for_slot"] for row in snapshots),
            in_flight_seconds_total=occupied,
            measurement_seconds=elapsed,
            average_in_flight=occupied / elapsed if elapsed else 0,
            backends=[
                dict(name=name, endpoint_url=endpoint.url, **snapshot)
                for name, endpoint, snapshot in zip(
                    self.names, self.backends, snapshots
                )
            ],
        )

    async def generate(self, prompt, **kwargs):
        self.waiting += 1
        try:
            endpoint = await self.available.get()
        finally:
            self.waiting -= 1
        try:
            return await endpoint.generate(prompt, **kwargs)
        finally:
            self.available.put_nowait(endpoint)

    async def aclose(self):
        await asyncio.gather(*(endpoint.aclose() for endpoint in self.backends))
