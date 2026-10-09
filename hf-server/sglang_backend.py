"""Remote SGLang token-scoring backend for Simple-JEV."""

import math
import time

import httpx


class SGLangBackend:
    """Score compiled prompt branches through SGLang's native /v1/score API."""

    def __init__(
        self,
        endpoint,
        model,
        *,
        api_key=None,
        timeout=60.0,
        client=None,
    ):
        endpoint = endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("SGLang endpoint must not be empty")
        if not model:
            raise ValueError("SGLang model must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SGLang timeout must be finite and positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=endpoint,
            headers=headers,
            timeout=timeout,
        )

    async def score(self, compiled):
        """Return one ordered label-logprob row per compiled branch."""
        from hf_server import BackendResult

        if compiled.plan._request.options.raw_logits:
            raise ValueError(
                "SGLang /v1/score returns token logprobs, not raw logits; "
                "raw_logits diagnostics are unsupported"
            )
        branches = compiled.branches
        if not branches or any(
            not branch.token_ids or not branch.output_ids for branch in branches
        ):
            raise ValueError("Expected nonempty SGLang scoring prompts and labels")
        if any(branch.model_inputs is not None for branch in branches):
            raise ValueError("SGLang backend currently supports text input only")

        payload = {
            "model": self.model,
            "query": [],
            "items": [branch.token_ids for branch in branches],
            "label_token_ids": [branch.output_ids for branch in branches],
            "apply_softmax": False,
            "return_token_logprobs": True,
        }
        start = time.perf_counter()
        try:
            response = await self._client.post("/v1/score", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ValueError(
                f"SGLang scoring timed out after {self.timeout:g} seconds"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ValueError(
                f"SGLang scoring failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"SGLang scoring request failed: {exc}") from exc

        try:
            body = response.json()
            rows = body["token_logprobs"]
            if not isinstance(rows, list) or len(rows) != len(branches):
                raise ValueError("unexpected token_logprobs row count")
            questions = {
                question.branch_id: question for question in compiled.plan.questions
            }
            results = {}
            for branch, row in zip(branches, rows):
                if (
                    not isinstance(row, list)
                    or len(row) != len(branch.output_ids)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in row
                    )
                ):
                    raise ValueError(
                        f"invalid token_logprobs for branch {branch.branch_id!r}"
                    )
                results[branch.branch_id] = {
                    label: float(value)
                    for label, value in zip(
                        questions[branch.branch_id].output_labels,
                        row,
                    )
                }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid SGLang scoring response: {exc}") from exc

        usage = body.get("usage") or {}
        return BackendResult(
            results,
            {
                "backend": "sglang",
                "prefill_strategy": "sglang_radix_cache",
                "branch_prompt_tokens": sum(len(b.token_ids) for b in branches),
                "remote_prompt_tokens": usage.get("prompt_tokens"),
                "branch_output_tokens": 0,
                "scored_positions": len(branches),
                "backend_seconds": time.perf_counter() - start,
            },
        )

    async def aclose(self):
        if self._owns_client:
            await self._client.aclose()
