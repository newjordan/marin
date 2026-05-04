# Research — iris_inference_service

This design replaces an earlier multi-engine framing with a vLLM-only Iris
inference service. The earlier direction assumed both vLLM and Levanter could
sit behind the same OpenAI-compatible HTTP path. Team guidance has since
changed: Marin should use vLLM as the only served inference engine for this
system and stop trying to make Levanter/JAX satisfy the OpenAI HTTP serving
contract.

## In-repo findings

- Served evals already have the launcher boundary this design should reuse:
  [`OpenAIEndpoint`, `RunningModel`, `ModelDeployment`, and `ModelLauncher`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/types.py#L10-L45).
  The key decision is that eval runners consume `RunningModel`; launcher code
  owns serving lifecycle and cleanup.
- The current lm-eval served runner converts `RunningModel` into
  `local-completions` or `local-chat-completions` args and writes results with
  `EvaluationTracker`: [`run_lm_eval`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/lm_eval.py#L42-L70)
  and [`build_lm_eval_model_args`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/lm_eval.py#L73-L92).
- The existing lm-eval harness evaluator already starts vLLM locally, gets a
  model id, and points lm-eval at `/v1/completions` or
  `/v1/chat/completions`: [`LMEvaluationHarnessEvaluator.evaluate`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py#L86-L190).
  This is the closest current implementation to preserve, but it couples eval
  code to engine startup.
- `VllmEnvironment` already owns vLLM startup/readiness/diagnostics and
  exposes the `/v1` API root once ready: [`VllmEnvironment`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L360-L430).
  It also makes native vLLM the only Iris-supported mode and rejects Docker
  sidecar mode: [`resolve_vllm_mode`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L241-L252).
- vLLM readiness is already defined as `GET {server_url}/models == 200`, where
  `server_url` is the `/v1` API root: [`_poll_until_ready`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L306-L344)
  and [`_get_first_model_id`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/inference/vllm_server.py#L347-L357).
- Fray actor calls have two paths: short `.remote()` calls and long-running
  `.submit()` operations that poll durably. The proxy-to-broker wait path
  should use the long-running path for requests that may take minutes:
  [`ActorMethod.submit`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/actor.py#L97-L112).
- Fray actor context is stored in `ContextVar`s and child threads do not inherit
  it automatically: [`current_actor`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/actor.py#L62-L76).
  A prototype confirmed this matters for an eval-side HTTP proxy: threaded
  handlers can lose the Iris context needed to resolve actor handles.
- `FrayIrisClient` already maps Fray jobs/actors to Iris jobs and replicas:
  [`FrayIrisClient.submit`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/fray/src/fray/iris_backend.py#L550-L576).
  The design should reuse this machinery instead of inventing a new scheduler.
- Harbor and Evalchemy already use vLLM, but with evaluator-specific process
  logic. Harbor starts a local vLLM server and points the agent at
  `api_base`: [`HarborEvaluator.evaluate`](https://github.com/marin-community/marin/blob/ce77573c01a590dcb7a39abaf18aa541903c9036/lib/marin/src/marin/evaluation/evaluators/harbor_evaluator.py#L197-L276).
  Evalchemy uses its own in-process vLLM path and patches around vLLM-TPU
  package quirks. This design should not try to normalize those frameworks in
  the first PR, but it should keep the service boundary generic enough that
  they can eventually consume `RunningModel`.

## Prototype findings that shaped the design

A closed draft prototype,
[`#5351`](https://github.com/marin-community/marin/pull/5351), exercised the
shape below. It is evidence for the design, not the implementation PR we expect
to merge.

- The broker/proxy/worker split works locally with `fray.LocalClient`: eval
  client -> OpenAI proxy -> broker actor -> worker actor -> deterministic
  OpenAI-compatible server -> client response.
- The proxy should expose `/v1/completions` and `/v1/chat/completions`, while
  worker-side engines should receive `engine_base_url` as the OpenAI API root
  ending in `/v1`.
- Broker submit must be idempotent for identical `(request_id, envelope)` and
  reject the same `request_id` with different data.
- If multiple workers report a terminal result for the same request, the broker
  should keep the first valid current-lease result and ignore stale leases.
- The first local proxy was single-threaded to preserve Iris context. That is
  acceptable for an MVP but not proof of production throughput.
- CodeQL flagged echoing an arbitrary request-id HTTP header into response
  headers. The spec should validate proxy request ids before reflecting them.

## Prior art / current direction

The main prior art here is not a general serving gateway; it is Marin's own
`VllmEnvironment` plus the new served-model eval boundary. vLLM already gives
the OpenAI-compatible surface and batching engine; Iris should provide the
elastic job topology, worker-preemption handling, worker lifecycle, and
diagnostics. The design should stay close to that split and avoid building a
new multi-tenant online inference service.

## Key changes from the earlier direction

- **Drop Levanter/JAX OpenAI serving from the design.** Levanter may still be
  used by other eval paths, but this Iris served inference service supports
  vLLM only.
- **Make vLLM prompt-logprob behavior a product risk, not a reason to add a
  second engine.** `mmlu_sl_verb_5shot` remains the right pressure test because
  it exercises the scoring contract we need.
- **Define engine lifecycle ownership.** Worker code starts vLLM, waits for
  `/v1/models`, records the served model id, forwards work, and tears vLLM down.
- **Treat the canonical Iris eval as acceptance evidence.** The design is not
  done until a real Iris run attempts `mmlu_sl_verb_5shot` and
  `humaneval_5shot` against the 1e22 MoE on `v5p-8`.
