"""Bound repeated model/answer failures using the existing response audit."""

import asyncio

from .pilot import response_key, validate_response


REQUEST_FAILURE_ATTEMPTS = 10


class RequestDiscarded(RuntimeError):
    def __init__(self, request_key, attempts, reasons):
        self.request_key = request_key
        self.attempts = attempts
        self.reasons = dict(reasons)
        super().__init__(
            f"Discarded after {attempts} consecutive model/answer failures "
            f"({', '.join(self.reasons)})"
        )

    def record(self):
        record = dict(request_key=self.request_key, failed_attempts=self.attempts,
                      failure_reasons=self.reasons)
        if "output_limit" in self.reasons:
            record["output_limit_attempts"] = self.reasons["output_limit"]
        return record


def failure_reason(row):
    if row.get("failure_reason") in ("context_length", "invalid_response", "output_limit"):
        return row["failure_reason"]
    if row.get("error") == "ContextLimitError":
        return "context_length"
    if row.get("finish_reason") == "length":
        return "output_limit"
    if row.get("error"):
        # Transport/authentication/server failures are not evidence that this
        # prompt cannot produce a usable answer.
        return None
    try:
        validate_response(row["task"], row.get("response"), row["messages"])
    except (ValueError, KeyError, TypeError):
        return "invalid_response"
    return None


class RequestFailureBudget:
    def __init__(self):
        self.counts = {}
        self.reasons = {}
        self.locks = {}

    def observe(self, row):
        # Older audits have no original-request key. Their exact prompt still
        # identifies repeated scoring failures, including existing stuck cases.
        key = row.get("request_key") or response_key(
            row["task"], row["messages"], row["max_tokens"]
        )
        reason = failure_reason(row)
        if reason:
            self.counts[key] = self.counts.get(key, 0) + 1
            reasons = self.reasons.setdefault(key, {})
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            self.counts.pop(key, None)
            self.reasons.pop(key, None)

    def check(self, key):
        attempts = self.counts.get(key, 0)
        if attempts >= REQUEST_FAILURE_ATTEMPTS:
            raise RequestDiscarded(key, attempts, self.reasons[key])

    def lock(self, key):
        # A feature shares this budget across all replica clients and methods.
        # Identical concurrent requests must not exceed the limit together.
        return self.locks.setdefault(key, asyncio.Lock())


# Preserve imports used by the original output-limit integration.
OUTPUT_LIMIT_ATTEMPTS = REQUEST_FAILURE_ATTEMPTS
OutputLimitDiscarded = RequestDiscarded
OutputLimitBudget = RequestFailureBudget
