# Spec — iris_inference_service

Concrete contracts for the vLLM-only Iris inference service. This spec describes
the surface reviewers are approving; it is not an implementation plan.

## Files

| File | Status | Purpose |
|---|---|---|
| `lib/marin/src/marin/inference/iris_vllm.py` | new | Iris vLLM launcher, proxy, broker, worker actors |
| `tests/evals/test_iris_vllm_inference.py` | new | CI-safe broker/proxy/worker tests with deterministic OpenAI stub |
| `scripts/iris/run_vllm_eval_pressure_test.py` | new, manual-only | Runner for the canonical Iris pressure test |

No proto changes. No persistent schema. No new engine abstraction layer.

Existing contracts reused unchanged:

- `OpenAIEndpoint`, `RunningModel`, `ModelDeployment`, `ModelLauncher` from
  `marin.inference.types`.
- `LmEvalRun`, `LmEvalAdapter`, `run_lm_eval` from `marin.evaluation.lm_eval`.
- `VllmEnvironment` from `marin.inference.vllm_server`.

## Endpoint constants

```python
MARIN_REQUEST_ID_HEADER = "X-Marin-Inference-Request-Id"
JSON_CONTENT_TYPE = "application/json"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MAX_REQUEST_BODY_BYTES: int = 8 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT: float = 3600.0
DEFAULT_LEASE_TIMEOUT: float = DEFAULT_REQUEST_TIMEOUT + 60.0
DEFAULT_WORKER_LEASE_WAIT_TIMEOUT: float = 1.0
DEFAULT_WORKER_READY_TIMEOUT: float = 900.0
DEFAULT_CLEANUP_TIMEOUT: float = 10.0
```

`REQUEST_ID_PATTERN` is enforced before a request id is reflected in an HTTP
response header. `lease_timeout` must be greater than `request_timeout`; a
normal in-flight vLLM request should not be re-leased while the worker is still
waiting for its HTTP response.

## Endpoint enum

```python
class OpenAIEndpointKind(StrEnum):
    """OpenAI-compatible endpoints routed by the Iris proxy."""

    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"

    @property
    def http_path(self) -> str: ...

    @property
    def api_path(self) -> str: ...

    @staticmethod
    def from_http_path(path: str) -> "OpenAIEndpointKind | None": ...
```

`http_path` returns `/v1/completions` or `/v1/chat/completions`.
`api_path` returns `completions` or `chat/completions`, appended under a vLLM
API root ending in `/v1`.

## Broker wire shapes

```python
@dataclass(frozen=True)
class OpenAIRequestEnvelope:
    """Opaque OpenAI-compatible request submitted by the proxy."""

    request_id: str
    endpoint: OpenAIEndpointKind
    payload_json: str


@dataclass(frozen=True)
class OpenAIResponseEnvelope:
    """Opaque OpenAI-compatible response returned by a worker."""

    status_code: int
    payload_json: str
    content_type: str = JSON_CONTENT_TYPE


@dataclass(frozen=True)
class InferenceLease:
    """A broker lease for one request."""

    lease_id: str
    worker_id: str
    request: OpenAIRequestEnvelope
    expires_at: float


@dataclass(frozen=True)
class WorkerReadyState:
    """Readiness report from one worker replica."""

    worker_id: str
    model_id: str


class BrokerWaitOutcome(StrEnum):
    """Terminal states for waiting on one broker request."""

    READY = "ready"
    TIMEOUT = "timeout"
    UNKNOWN_REQUEST = "unknown_request"
    BROKER_STOPPED = "broker_stopped"


@dataclass(frozen=True)
class BrokerWaitResult:
    """Result of waiting for a terminal broker response."""

    outcome: BrokerWaitOutcome
    response: OpenAIResponseEnvelope | None = None


@dataclass(frozen=True)
class LeaseResult:
    """Result of a lease call.

    `lease=None` means no work was available before timeout. `stopped=True`
    means the broker is shutting down and workers should exit.
    """

    lease: InferenceLease | None
    stopped: bool = False
```

`BrokerWaitResult.response` is non-`None` only when `outcome` is
`BrokerWaitOutcome.READY`.

## Broker API

```python
class BrokerRequestStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool: ...


class IrisInferenceBroker:
    """In-memory broker scoped to one eval run."""

    def __init__(self, lease_timeout: float = DEFAULT_LEASE_TIMEOUT, now: Callable[[], float] = time.monotonic) -> None:
        """Create an empty broker.

        Raises:
            ValueError: if `lease_timeout <= 0`.
        """

    def submit(self, request: OpenAIRequestEnvelope) -> bool:
        """Submit one logical request.

        Returns `True` for a new request and `False` for an identical duplicate.
        Raises `ValueError` if the same request id was already submitted with
        different endpoint or payload data. Raises `RuntimeError` if the broker
        is stopped.
        """

    def lease(self, worker_id: str, wait_timeout: float | None = None) -> LeaseResult:
        """Lease pending work after first expiring overdue leases.

        Returns no work on timeout. A leased request becomes pending again when
        `expires_at <= now()`. Raises `ValueError` if `wait_timeout < 0`.
        """

    def complete(self, request_id: str, lease_id: str, response: OpenAIResponseEnvelope) -> bool:
        """Store a successful terminal result.

        Returns whether this call stored the terminal result. Returns `False`
        if the request is already terminal or if `lease_id` is stale. Raises
        `KeyError` for an unknown request id.
        """

    def fail(self, request_id: str, lease_id: str, response: OpenAIResponseEnvelope) -> bool:
        """Store a failed terminal result with the same semantics as complete."""

    def poll(self, request_id: str) -> OpenAIResponseEnvelope | None:
        """Return the terminal response if one exists."""

    def wait(self, request_id: str, timeout: float | None = None) -> BrokerWaitResult:
        """Wait for a terminal response.

        Returns a typed outcome so the proxy can distinguish unknown requests,
        timeouts, broker shutdown, and successful terminal responses. Raises
        `ValueError` if `timeout < 0`.
        """

    def status(self, request_id: str) -> BrokerRequestStatus | None:
        """Return the current lifecycle state for a request, if known."""

    def record_worker_ready(self, state: WorkerReadyState) -> None:
        """Record that one worker has started vLLM and read `/v1/models`.

        Re-recording the same `worker_id` with the same `model_id` is
        idempotent. Re-recording it with a different `model_id` raises
        `ValueError`.
        """

    def wait_for_workers_ready(self, worker_count: int, timeout: float) -> tuple[WorkerReadyState, ...]:
        """Wait until at least `worker_count` distinct workers are ready."""

    def stop(self) -> None:
        """Stop accepting submissions and wake blocked workers/proxies."""
```

The broker has no persistence contract. Restarting it starts with an empty
request table. Broker methods are linearizable: all request-table and
worker-readiness mutations happen under one lock, blocked `lease`, `wait`, and
`wait_for_workers_ready` calls are woken on submit, terminal result, readiness,
lease expiry, and stop, and stale lease terminal reports never overwrite the
current lease or a terminal result.

## Worker API

```python
@dataclass(frozen=True)
class IrisVllmWorkerConfig:
    """Configuration for one worker replica."""

    model: ModelDeployment
    host: str = "127.0.0.1"
    port: int | None = None
    vllm_timeout: int = 3600
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT
    extra_vllm_args: tuple[str, ...] = ()


class IrisVllmWorker:
    """Actor that starts vLLM and forwards broker leases to the local server."""

    def __init__(self, broker: ActorHandle, config: IrisVllmWorkerConfig) -> None: ...

    def run(self, max_requests: int | None = None) -> int:
        """Start vLLM, process leases until stopped, then tear vLLM down.

        Returns the number of forwarded requests. Raises if vLLM fails to start
        or if `max_requests < 0`.
        """
```

`IrisVllmWorker` uses `VllmEnvironment` with native mode. It must not expose a
Levanter/JAX mode. The worker converts `ModelDeployment` to `ModelConfig` as:

- `ModelConfig.name = deployment.model_name`
- `ModelConfig.path = deployment.model_path`
- `ModelConfig.engine_kwargs = dict(deployment.engine_kwargs)`

The launcher must add `--served-model-name <deployment.model_name>` to vLLM
startup args, and `extra_vllm_args` must not contain another
`--served-model-name`. The launcher returns `OpenAIEndpoint.model` as
`deployment.model_name`; workers still record the `/v1/models` id for readiness
and diagnostics.

The worker forwards request bodies without parsing them:

- `OpenAIEndpointKind.COMPLETIONS` -> `{env.server_url}/completions`
- `OpenAIEndpointKind.CHAT_COMPLETIONS` -> `{env.server_url}/chat/completions`

Non-2xx HTTP responses from vLLM are completed and forwarded as-is. On
`requests.RequestException`, the worker reports a `502`
`OpenAIResponseEnvelope` via `broker.fail` with the current lease id.
Non-network exceptions from vLLM startup or worker code propagate so Iris/Fray
can surface a failed worker.

## Proxy API

```python
@dataclass(frozen=True)
class RunningIrisInferenceProxy:
    """Handle for a local eval-side proxy."""

    base_url: str
    request_id_header: str = MARIN_REQUEST_ID_HEADER


def serve_iris_inference_proxy(
    broker: ActorHandle,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> AbstractContextManager[RunningIrisInferenceProxy]:
    """Run a local OpenAI-compatible proxy backed by a broker actor.

    The yielded `base_url` is an OpenAI API root ending in `/v1`.
    """
```

Supported HTTP methods and paths:

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/v1/completions` | Submit completions envelope and return broker response |
| `POST` | `/v1/chat/completions` | Submit chat envelope and return broker response |

Error contract:

| Status | Condition |
|---|---|
| `400` | Unsafe or duplicate `X-Marin-Inference-Request-Id`; invalid JSON; non-object JSON; `stream: true` |
| `404` | Unsupported path |
| `409` | Same request id reused with different envelope |
| `413` | Request body exceeds `MAX_REQUEST_BODY_BYTES` |
| `415` | Explicit non-JSON content type |
| `503` | Broker stopped, or broker reports the request as unknown after submit |
| `504` | Broker wait timed out |

If no request id header is present, the proxy generates a `uuid.uuid4().hex`
id. It echoes `MARIN_REQUEST_ID_HEADER` only after validation/generation. The
broker compares duplicate submissions by exact endpoint and exact UTF-8 decoded
payload string; a retry that changes JSON whitespace with the same request id
is a conflict. The proxy should call `broker.wait.submit(...)`, not a short
direct actor call, for request waits. Its serving implementation must either be
single-threaded, copy Iris/Fray context into handler threads, or capture a
broker handle that is safe to use without thread-local context lookups.

## Launcher API

```python
@dataclass(frozen=True)
class IrisVllmLauncherConfig:
    """Iris resources and timeouts for one served eval run."""

    worker_count: int
    worker_resources: ResourceConfig
    broker_resources: ResourceConfig
    lease_timeout: float = DEFAULT_LEASE_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    worker_lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT
    worker_ready_timeout: float = DEFAULT_WORKER_READY_TIMEOUT
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 0
    service_name: str | None = None
    extra_vllm_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class IrisVllmLauncher(ModelLauncher):
    """ModelLauncher that serves one model through vLLM workers on Iris."""

    client: Client
    config: IrisVllmLauncherConfig

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        """Launch broker, proxy, and vLLM worker replicas.

        Yields `RunningModel` whose endpoint base URL is the proxy `/v1` root.
        The endpoint model id is the served model id accepted by vLLM.
        """
```

Validation:

- `worker_count > 0`
- all timeout values are positive
- `lease_timeout > request_timeout`
- `extra_vllm_args` must not include `--served-model-name`
- `worker_resources` must request the accelerator topology for the vLLM worker
  job, e.g. `ResourceConfig.with_tpu("v5p-8")`

`worker_count` is replica count. `worker_count=2` with
`ResourceConfig.with_tpu("v5p-8")` means two worker replicas, each requesting a
full `v5p-8` resource shape.

Startup contract:

- create the broker;
- start worker replicas;
- wait for `worker_count` readiness records within `worker_ready_timeout`;
- require every readiness model id to equal `deployment.model_name`;
- start the local proxy;
- yield `RunningModel(OpenAIEndpoint(base_url=proxy.base_url, model=deployment.model_name), tokenizer=deployment.tokenizer)`.

Context exit contract:

- stop the local proxy;
- call `broker.stop()` to wake workers and waiters;
- wait up to `DEFAULT_CLEANUP_TIMEOUT` for worker operations to end;
- surface partial-start and cleanup failures with broker state, worker ids, and
  vLLM diagnostics when available.

## Manual pressure-test contract

`scripts/iris/run_vllm_eval_pressure_test.py` should accept:

```text
--model-name <served name>
--model-path <gs://... or hf id>
--tokenizer <hf id or local path>
--output-path <local or gs:// path>
--tpu-type v5p-8
--worker-count <int>
--task mmlu_sl_verb_5shot|humaneval_5shot  # repeatable
--limit <int>                              # optional debug cap
--dry-run
```

Default task set:

- `mmlu_sl_verb_5shot` with `LmEvalAdapter.LOCAL_COMPLETIONS`
- `humaneval_5shot` with `LmEvalAdapter.LOCAL_CHAT_COMPLETIONS` and
  `apply_chat_template=True`

The pressure-test output must record:

- Iris job ids for eval, broker, and worker group
- vLLM model id returned by `/v1/models`
- output directory containing lm-eval results and samples
- request counts by endpoint
- throughput summary: wall time, requests/sec, generated tokens/sec when
  available
- failure summary with vLLM diagnostics if either task fails

This script is manual-only and should not run in normal CI.
If `--output-path` is a `gs://` path, the runner owns uploading the lm-eval
output directory after each task; otherwise it writes locally.

## Tests

CI tests should cover:

- broker submit/lease/complete/wait lifecycle;
- expired lease requeue;
- stale lease terminal result ignored;
- duplicate valid terminal result keeps the first response;
- conflicting duplicate submit returns `409` through the proxy;
- unsafe request-id header returns `400` and is not echoed;
- invalid JSON, non-object JSON, `stream: true`, non-JSON content type, and
  oversized request bodies map to the documented proxy errors;
- proxy/worker route completions and chat-completions to a deterministic
  OpenAI-compatible stub;
- real lm-eval tiny scoring path gated by optional deps.

Manual tests should run the pressure-test task pair on Iris before the first
implementation PR is treated as production-shaped.

## Out of scope

- Levanter/JAX behind OpenAI HTTP.
- Engine polymorphism (`EngineKind.LEVANTER`, `EngineKind.JAX`, etc.).
- Persistent broker queue or broker restart recovery.
- OpenAI server-sent streaming.
- Cancellation of in-flight vLLM requests.
- Multi-tenant always-on inference service.
- Harbor/Evalchemy migration in the first implementation PR.
- Parameter syncing for RL or in-training-loop evals.
