# Iris vLLM Inference Service

Run Marin evals against vLLM served on Iris without making each evaluator know
how to start a model. The eval side should keep talking to `RunningModel` and
OpenAI-compatible HTTP; Iris launchers and workers should own vLLM startup,
readiness, worker-preemption tolerance, diagnostics, and cleanup. The first
milestone is concrete: run `mmlu_sl_verb_5shot` and `humaneval_5shot` from
lm-eval against the 1e22 MoE served by vLLM on Iris `v5p-8`. That milestone is
a stepping stone toward moving eval workloads off Ray and toward a common
served-model path for lm-eval first, with Harbor and Evalchemy later.

## Background

Marin already has the core eval/serving boundary: `ModelLauncher` returns a
`RunningModel`, and `run_lm_eval` consumes that object without knowing how the
model was launched. Current vLLM eval code still mixes evaluator logic with
engine startup, while earlier drafts tried to support both vLLM and Levanter as
OpenAI HTTP engines. The updated team direction is simpler: this Iris inference
service supports vLLM only. See [research.md](./research.md) for the code refs
and prototype findings that shaped the proposal.

## Challenges

The hard part is not the HTTP shape; vLLM already exposes the OpenAI-compatible
API. The hard part is where the serving lifecycle lives once the model is
elastic and preemptible. Workers need to start vLLM, wait for readiness, pull
work from a broker, forward requests locally, and report terminal results in a
way the proxy can correlate with client requests and current leases. The
eval-side proxy also has to use Iris/Fray actor calls correctly: long requests
should use durable actor operations, and any threaded HTTP serving path must not
lose Iris context.

The other real risk is logprob coverage. Generation through vLLM is the paved
path; lm-eval scoring tasks such as MMLU exercise prompt logprobs through the
completions API. We should not add Levanter/JAX back as a second served engine
to mask that risk. Instead, `mmlu_sl_verb_5shot` becomes the pressure test that
tells us whether vLLM currently satisfies the scoring contract or whether we
need a vLLM fix / workaround.

## Costs / Risks

- Dropping Levanter/JAX from the served HTTP design removes a fallback for
  scoring evals. If vLLM cannot satisfy prompt-logprob tasks, those tasks stay
  on existing non-served paths until vLLM is fixed.
- A single broker is a coordinator bottleneck. This is acceptable for the first
  batch-eval service, but throughput has to be measured before calling it
  production-ready.
- The proxy/broker path adds another failure surface between lm-eval and vLLM.
  It buys worker-preemption tolerance, but not full recovery: broker, proxy, or
  eval-job restart loses in-flight requests because the queue is in memory.
  Errors need crisp request ids and diagnostics.
- The first implementation will not cover streaming, cancellation, multi-tenant
  serving, persistent queues, or in-training-loop dynamic weights.

## Design

Keep three roles, each with one owner:

- **Eval job:** runs the evaluator and a local OpenAI-compatible proxy. It sees
  only `RunningModel(endpoint=OpenAIEndpoint(base_url=..., model=...))`.
- **Broker actor:** stores an in-memory request table for one eval run. It
  accepts opaque OpenAI request envelopes, leases pending work to workers,
  expires leases after a timeout longer than the worker request timeout, and
  accepts terminal results only from the current lease.
- **vLLM worker group:** one Iris job with `N` replicas. Each replica starts
  native vLLM with `--served-model-name` set to the deployment name, waits for
  `/v1/models`, records readiness with the broker, leases work from the broker,
  forwards to its local `http://127.0.0.1:<port>/v1`, and reports the response
  body/status back to the broker.

The key constraint is that **vLLM is the only served engine**. There is no
`EngineKind.LEVANTER`, no JAX OpenAI server, and no polymorphic engine adapter
in this design. Existing Levanter eval paths can continue elsewhere while this
system proves the vLLM route.

The proxy exposes only:

- `/v1/completions`
- `/v1/chat/completions`

It generates or accepts a safe request id, submits an
`OpenAIRequestEnvelope(request_id, endpoint, payload_json)` to the broker, then
waits for the broker result with a long-running Fray actor operation. It may
retry broker submission with the same request id if the actor call fails. The
broker treats identical duplicate submits as idempotent and rejects conflicting
reuses of the same request id. The proxy validates only the minimum HTTP shape:
JSON object bodies, safe request ids, supported paths, bounded body size, and
no streaming responses.

The worker forwards request bodies opaquely. It does not parse lm-eval, Harbor,
Evalchemy, or OpenAI semantics beyond choosing the endpoint path. This keeps
the broker/proxy contract reusable for other evaluators that can call an
OpenAI-compatible server.

The main tradeoff is a pull broker rather than direct proxy load-balancing. A
pull broker lets replacement workers pick up expired work without the proxy
knowing which workers are alive. The cost is one coordinator and an in-memory
queue. That is the right MVP shape for batch evals because it gives useful
preemption behavior without turning this into a persistent online serving
gateway.

The first implementation should reuse `VllmEnvironment` for local vLLM
startup/readiness and diagnostics, then wrap it in an Iris/Fray
`ModelLauncher`. That launcher returns `RunningModel` with the proxy's `/v1`
API root and the deployment model name that vLLM was told to serve. Engine
kwargs remain vLLM kwargs; they are not generic service fields.

## Testing

Normal CI should cover the contract without requiring real vLLM or Iris:

- broker lifecycle: submit, lease, lease expiry, complete/fail, duplicate
  submit, stale lease result ignored, and first valid terminal result kept;
- proxy safety: request-id validation before echoing headers;
- proxy -> broker -> worker -> deterministic OpenAI-compatible stub for both
  completions and chat completions;
- real lm-eval tiny scoring path when optional `lm_eval`, `torch`, and
  tokenizer deps are installed.

The design pressure test is manual and realistic:

1. Run `mmlu_sl_verb_5shot` from lm-eval against the 1e22 MoE served by vLLM on
   Iris `v5p-8`.
2. Run `humaneval_5shot` from lm-eval against the same model/service shape.

This pair is intentional. MMLU SL Verb proves the completions/logprob path;
HumanEval proves generative chat/template behavior through the chat-completions
adapter. The pressure test should record throughput, failure modes, vLLM
diagnostics, output layout, and whether lease expiry or worker replacement was
exercised.

## Open Questions

- Does vLLM already satisfy the prompt-logprob behavior needed by
  `mmlu_sl_verb_5shot` on the target 1e22 MoE, or is an upstream vLLM fix /
  Marin workaround required before scoring evals move to this service?
- What minimum throughput makes the single-broker design acceptable for near
  term eval runs?
- Should pressure-test results be uploaded by the manual runner itself, or is
  local output plus Iris logs enough for the first implementation PR?
