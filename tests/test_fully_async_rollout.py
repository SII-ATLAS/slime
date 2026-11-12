"""CPU unit tests for the fully-async rollout worker's queue contract.

The module docstring of ``slime.rollout.fully_async_rollout`` promises that the
worker's output queue "stays warm" across ``generate_rollout`` calls: each call
takes ``rollout_batch_size`` completed groups and leaves the rest queued.

Three behaviours are pinned here:

  1. ``_generate_rollout_async`` consumes exactly ``rollout_batch_size`` groups
     and leaves the surplus in the queue. (It used to drain the whole queue and
     slice — throwing away fully generated, reward-scored groups whose prompts
     had already been consumed from the data buffer.)
  2. The task done-callback never blocks. It runs on the event-loop thread, so
     a bounded queue that filled up would freeze every in-flight generation.
  3. Backpressure exists anyway: ``_loop`` stops pulling new prompts while a
     full pool of completed groups is already waiting to be consumed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace

# ``fully_async_rollout`` imports ``sglang_rollout``, which needs sglang_router
# and (transitively) transformers — both deliberately absent from the CPU CI
# env. The tests below never dial a server or touch a tokenizer, so stub the
# imports, same as tests/test_agent/test_agent_rollout_cpu.py.
try:
    import sglang_router  # noqa: F401
except ImportError:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
try:
    import transformers  # noqa: F401
except ImportError:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.fully_async_rollout as fa
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.staleness import compute_staleness_metrics, sample_staleness
from slime.utils.types import Sample

NUM_GPUS = 0


def test_custom_data_source_keeps_original_entrypoint(monkeypatch):
    args = SimpleNamespace()
    # A custom data source may have its own unrelated scheduler attribute.
    source = SimpleNamespace(scheduler=object())
    expected = object()

    async def generate(received_args, rollout_id, received_source):
        assert received_args is args and received_source is source
        assert rollout_id == 3
        return expected

    monkeypatch.setattr(fa, "_get_worker", lambda args, source: _make_worker(monkeypatch))
    monkeypatch.setattr(fa, "_generate_rollout_async", generate)
    monkeypatch.setattr(fa, "run", asyncio.run)
    assert fa.generate_rollout_fully_async(args, 3, source) is expected


class _FakeGenerateState:
    def __init__(self, args):
        self.sampling_params = {}


class _FakeDataBuffer:
    """Finite fuel: one group per ``get_samples`` call until exhausted."""

    def __init__(self, groups):
        self._groups = deque(groups)
        self.requeued = []

    def get_samples(self, n):
        assert n == 1
        if not self._groups:
            return []
        return [self._groups.popleft()]

    def add_samples(self, groups):
        self.requeued.extend(groups)


def _make_group(index: int) -> list[Sample]:
    sample = Sample(index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    return [sample]


def _make_worker(monkeypatch, data_buffer=None, concurrency=4) -> fa.AsyncRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1)
    return fa.AsyncRolloutWorker(args, data_buffer or _FakeDataBuffer([]), concurrency=concurrency)


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_batch_size=4, rollout_data_transport="object-store", rollout_sample_filter_path=None
    )
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert len(out) == 4
    # FIFO: the oldest four groups ship first.
    assert [group[0].index for group in out] == [0, 1, 2, 3]
    # The other six are still queued for the next rollout, not thrown away.
    assert worker.queue_size() == 6
    assert [gid for gid, _ in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.unit
def test_get_completed_groups_limit(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(5):
        worker.output_queue.put((gid, _make_group(gid)))

    assert [gid for gid, _ in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3, 4]
    assert worker.get_completed_groups(limit=3) == []


@pytest.mark.unit
def test_concurrency_is_scaled_by_samples_per_prompt(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(n_samples_per_prompt=4)
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=2).concurrency == 1
    args.fully_async_group_prefetch_factor = 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 4


@pytest.mark.unit
def test_dynamic_filter_drops_groups_and_refills(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(8):
        group = _make_group(gid)
        group[0].reward = float(gid % 2)
        worker.output_queue.put((gid, group))
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)

    def keep_odd(args, group):
        return DynamicFilterOutput(keep=bool(group[0].reward), reason="even")

    monkeypatch.setattr(fa, "load_function", lambda path: keep_odd)
    args = SimpleNamespace(
        rollout_batch_size=3,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="test.keep_odd",
        reward_key=None,
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in result.samples] == [1, 3, 5]
    assert "_dropped_samples" not in result.metrics
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 3
    assert result.metrics["rollout/dynamic_filter/dropped_ratio"] == 0.5
    assert result.metrics["rollout/dynamic_filter/drop_even"] == 3
    assert result.metrics["rollout/raw_reward_before_filters"] == 0.5
    assert [gid for gid, _ in worker.get_completed_groups()] == [6, 7]


@pytest.mark.unit
@pytest.mark.parametrize("segmented", [False, True])
def test_uniform_reward_filter_replenishes_eight_sample_groups(monkeypatch, segmented):
    worker = _make_worker(monkeypatch)
    rewards = [[0.0] * 8, [1.0] * 8, [0.0, 1.0] * 4, [1.0, 0.0] * 4, [0.0] * 8]
    groups = []
    for gid, values in enumerate(rewards):
        group = []
        for index, reward in enumerate(values):
            sample = Sample(index=gid * 8 + index, reward=reward, status=Sample.Status.COMPLETED)
            # Multiple context windows still represent one trajectory reward.
            group.append([sample, sample] if segmented and index % 2 else [sample] if segmented else sample)
        groups.append(group)
        worker.output_queue.put((gid, group))
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=8,
        reward_key=None,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))
    assert result.samples == groups[2:4]
    assert all(len(group) == 8 for group in result.samples)
    assert result.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert result.metrics["rollout/dynamic_filter/drop_zero_std_1.0"] == 1
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 2
    assert result.metrics["rollout/raw_reward_before_filters"] == 0.5
    assert [gid for gid, _ in worker.get_completed_groups()] == [4]


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions,expected",
    [(["10"], 0), (["8"], 2), (["3", "10"], 7), ([], None), (["bad"], None), (["11"], None), (["²"], None)],
)
def test_sample_staleness_uses_oldest_valid_version(versions, expected):
    assert sample_staleness(SimpleNamespace(weight_versions=versions), 10) == expected


@pytest.mark.unit
def test_staleness_metrics_use_serving_snapshot():
    samples = [
        SimpleNamespace(weight_versions=["2"]),
        SimpleNamespace(weight_versions=["3"]),
        SimpleNamespace(weight_versions=[]),
    ]
    assert compute_staleness_metrics(samples, 10) == {
        "staleness/unknown_count": 1,
        "staleness/mean": 7.5,
        "staleness/max": 8,
    }
    assert compute_staleness_metrics(samples, None) == {}


@pytest.mark.unit
def test_done_callback_never_blocks_event_loop_thread(monkeypatch):
    """The callback runs on the loop thread; blocking there freezes every
    in-flight generation. Push more results than the old bounded-queue cap
    (1000) through it and require completion."""
    worker = _make_worker(monkeypatch)

    class _DoneTask:
        def __init__(self, gid):
            self._result = _make_group(gid)

        def result(self):
            return self._result

    def _push_all():
        for gid in range(1001):
            worker._make_done_cb(gid, _make_group(gid))(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)

    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_interrupted_segmented_group_retains_successful_siblings(monkeypatch):
    worker = _make_worker(monkeypatch)
    original = [Sample(index=i, prompt=f"original-{i}") for i in range(8)]
    callback = worker._make_done_cb(0, original)
    original[0].remove_sample = True
    original[0].prompt = "mutated by aborted generator"
    result = [[Sample(index=i, status=Sample.Status.COMPLETED)] for i in range(8)]
    result[3].append(Sample(index=3, status=Sample.Status.ABORTED))
    callback(SimpleNamespace(result=lambda: result))
    assert worker.queue_size() == 0
    [requeued] = worker.data_buffer.requeued
    assert len(requeued) == 8
    for index, sample in enumerate(requeued):
        if index == 3:
            assert isinstance(sample, Sample)
            assert sample.prompt == "original-3"
            assert sample.status == Sample.Status.PENDING and not sample.remove_sample
        else:
            assert sample is result[index]


def test_oldest_buffer_order_uses_retained_segment_versions():
    from slime.data.data_source import pop_oldest

    older = [[Sample(weight_versions=["2", "7"])], Sample()]
    newer = [[Sample(weight_versions=["4"])], Sample()]
    buffer = [newer, older]
    assert pop_oldest(None, None, buffer, 1) == [older]
    assert buffer == [newer]


@pytest.mark.parametrize("status,reward", [(Sample.Status.ABORTED, 0.0), (Sample.Status.COMPLETED, None)])
def test_retry_rejects_unfinished_segments_before_starting_generation(monkeypatch, status, reward):
    from slime.rollout import sglang_rollout

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: types.SimpleNamespace(aborted=False))
    group = [Sample(), [Sample(status=status, reward=reward)]]
    with pytest.raises(ValueError, match="Only completed segmented"):
        asyncio.run(sglang_rollout.generate_and_rm_group(types.SimpleNamespace(group_rm=False), group, {}))


@pytest.mark.unit
@pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED])
def test_unfinished_flat_group_is_requeued_without_discarding_partial_response(monkeypatch, status):
    worker = _make_worker(monkeypatch)
    group = [Sample(index=0, status=status, response="partial", response_length=7)]
    worker._make_done_cb(0, group)(SimpleNamespace(result=lambda: group))
    assert worker.queue_size() == 0
    assert worker.data_buffer.requeued == [group]
    assert worker.data_buffer.requeued[0][0] is group[0]


@pytest.mark.unit
def test_loop_backpressure_stops_topping_up_when_queue_is_full(monkeypatch):
    """With instantly-completing generations and plenty of fuel, the queue must
    plateau around ``concurrency`` instead of absorbing the whole dataset."""
    concurrency = 3
    fuel = 60
    data_buffer = _FakeDataBuffer([_make_group(i) for i in range(fuel)])

    async def _instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _instant_generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=concurrency)
    worker.poll_interval = 0.01

    worker.start()
    try:
        # Give the loop ample iterations to overshoot if it is going to.
        deadline = time.time() + 3.0
        max_seen = 0
        while time.time() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            if max_seen > 2 * concurrency:
                break
            time.sleep(0.02)
    finally:
        worker.close()

    # In-flight tasks may still land after the gate check, so allow one pool
    # beyond the gate — but nothing near the unthrottled fuel size.
    assert 0 < max_seen <= 2 * concurrency, f"queue grew to {max_seen} with concurrency={concurrency}"


def test_admission_pause_keeps_unfinished_agent_group_alive(monkeypatch):
    started, finish = threading.Event(), threading.Event()
    group = _make_group(7)
    invocations = []

    async def generate(args, received, sampling_params, evaluation):
        invocations.append(received)
        started.set()
        while not finish.is_set():
            await asyncio.sleep(0.01)
        return [[received[0]]]

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    worker = _make_worker(monkeypatch, data_buffer=_FakeDataBuffer([group]), concurrency=1)
    worker.start()
    try:
        assert started.wait(5)
        assert worker.pause(drain=False) is False
        assert worker.active == 1 and worker.queue_size() == 0
        # Weight publication pauses new admission; the existing sandbox task is
        # still the same coroutine when the next training round resumes it.
        worker.resume()
        finish.set()
        deadline = time.monotonic() + 5
        while worker.queue_size() == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert invocations == [group]
        assert worker.get_completed_groups()[0][1] == [[group[0]]]
        assert not worker.data_buffer.requeued
    finally:
        finish.set()
        worker.close()


@pytest.mark.parametrize("transport", ["object-store", "straw"])
def test_local_worker_lifecycle_is_scoped_to_source(monkeypatch, transport):
    from slime.data.data_source import DataSource

    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(fa, "get_rollout_num_engines", lambda args: 1)

    async def generate(args, group, sampling_params, evaluation):
        await asyncio.sleep(0.01)
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    args = SimpleNamespace(n_samples_per_prompt=1, sglang_server_concurrency=2, rollout_data_transport=transport)
    sources = [_FakeDataBuffer([_make_group(i) for i in range(20)]) for _ in range(2)]
    workers = [fa._get_worker(args, source) for source in sources]
    try:
        assert workers[0] is not workers[1]
        deadline = time.monotonic() + 5
        while not all(worker.queue_size() for worker in workers):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for worker in workers:
            assert worker.pause() is False
            saved = worker.state_dict()
            queued = worker.queue_size()
            time.sleep(0.03)
            assert worker.queue_size() == queued and not worker.active
            replacement = fa.AsyncRolloutWorker(args, worker.data_buffer, concurrency=2)
            replacement.load_state_dict(saved)
            assert replacement.get_completed_groups() == worker.get_completed_groups()
    finally:
        for source in sources:
            DataSource.close(source)
    assert all(not worker.worker_thread.is_alive() for worker in workers)


def test_recoverable_group_failure_is_replaced_without_fake_reward(monkeypatch):
    from slime.rollout.base_types import RecoverableRolloutError

    worker = _make_worker(monkeypatch)
    failed = [Sample(index=i, prompt="failed") for i in range(8)]

    def result():
        raise RecoverableRolloutError("grader_invalid", "no verdict", source="ava_math")

    worker._make_done_cb(0, failed)(SimpleNamespace(result=result))
    kept = [
        [Sample(index=10 + i * 8 + j, status=Sample.Status.COMPLETED, reward=1.0) for j in range(8)] for i in range(2)
    ]
    for i, group in enumerate(kept, 1):
        worker.output_queue.put((i, group))
    monkeypatch.setattr(fa, "_get_worker", lambda *args: worker)
    discarded = []
    monkeypatch.setattr(fa, "discard_rollout_group", lambda group, *args, **kwargs: discarded.append(group))
    args = SimpleNamespace(
        rollout_batch_size=2, rollout_data_transport="object-store", rollout_sample_filter_path=None
    )
    output = asyncio.run(fa._generate_rollout_async(args, 0, None))
    assert output.samples == kept and all(len(group) == 8 for group in output.samples)
    assert discarded == [failed] and worker.data_buffer.requeued == []
    assert all(sample.reward is None for sample in failed)
    assert worker.error is None
    assert output.metrics["rollout/failures/dropped_groups"] == 1
    assert output.metrics["rollout/failures/ava_math/grader_invalid"] == 1


def test_recoverable_group_failure_budget_stops_persistent_failure(monkeypatch):
    from slime.rollout.base_types import RolloutGroupFailure

    worker = _make_worker(monkeypatch)
    for i in range(3):
        worker.output_queue.put((i, RolloutGroupFailure(_make_group(i), "sandbox", "terminal")))
    monkeypatch.setattr(fa, "_get_worker", lambda *args: worker)
    monkeypatch.setattr(fa, "discard_rollout_group", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        rollout_batch_size=2,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        fully_async_max_consecutive_failed_groups=3,
    )
    with pytest.raises(RuntimeError, match="Consecutive failed"):
        asyncio.run(fa._generate_rollout_async(args, 0, None))


def test_group_failure_settles_siblings_and_preserves_unknown_errors(monkeypatch):
    from slime.rollout import sglang_rollout as sg
    from slime.rollout.base_types import RecoverableRolloutError

    monkeypatch.setattr(sg, "GenerateState", lambda args: SimpleNamespace(aborted=False))

    async def scenario(error):
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def generate(args, sample, params, **kwargs):
            if sample.index == 0:
                await started.wait()
                raise error
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(sg, "generate_and_rm", generate)
        with pytest.raises(type(error), match=str(error)):
            await sg.generate_and_rm_group(SimpleNamespace(group_rm=False), [Sample(index=0), Sample(index=1)], {})
        assert cleaned.is_set()

    asyncio.run(scenario(RecoverableRolloutError("sandbox", "known failure")))
    asyncio.run(scenario(ValueError("unknown bug")))


def test_ava_failure_policy_preserves_valid_zero_and_detects_collection_error(monkeypatch):
    from examples.ava_rl.failures import fail_attempt, validate_verifier
    from slime.rollout.base_types import RecoverableRolloutError

    monkeypatch.delenv("AVA_TOLERATE_ROLLOUT_FAILURES", raising=False)
    with pytest.raises(RuntimeError) as caught:
        fail_attempt("test", "failed")
    assert not isinstance(caught.value, RecoverableRolloutError)
    monkeypatch.setenv("AVA_TOLERATE_ROLLOUT_FAILURES", "1")
    for reason in ["1 failed in 0.2s", "1 passed in 0.2s", {"pytest_log": "1 failed"}]:
        validate_verifier({"reward": {"score": 0.0, "reason": reason}})
    for output in ["ERROR collecting test_final_state.py", "_____ ERROR collecting /tests/test.py _____\nImportError"]:
        with pytest.raises(RecoverableRolloutError) as caught:
            validate_verifier({"reward": {"score": 0.0, "reason": {"pytest_log": output}}})
        assert caught.value.reason == "verifier_collection_error"
    with pytest.raises(RecoverableRolloutError, match="infrastructure"):
        validate_verifier({"verifier_error": {"type": "RuntimeExecutionError"}})


@pytest.mark.parametrize(
    "status,recoverable,error_type,expected_calls",
    [
        (500, True, "SandboxReadyTimeoutException", 3),
        (500, True, "VerificationError", 1),
        (500, False, "ValueError", 1),
        (500, False, "TaskEnvError", 1),
        (403, True, "HttpError", 1),
    ],
)
def test_ava_rpc_respects_native_classification_over_real_http(
    monkeypatch, status, recoverable, error_type, expected_calls
):
    from aiohttp import web
    from examples.ava_rl.generate import Service
    from slime.rollout.base_types import RecoverableRolloutError

    monkeypatch.setenv("AVA_TOLERATE_ROLLOUT_FAILURES", "1")

    async def scenario():
        calls = []

        async def handle(request):
            calls.append(await request.json())
            return web.json_response(
                {
                    "error_type": error_type,
                    "message": "original native failure",
                    "recoverable": recoverable,
                    "retryable": True,
                },
                status=status,
            )

        app = web.Application()
        app.router.add_post("/v1/generate", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            state = SimpleNamespace(url=f"http://127.0.0.1:{port}", token="local-test")
            with pytest.raises(RuntimeError) as caught:
                await Service.call(state, "generate", {"key": "test"})
            assert isinstance(caught.value, RecoverableRolloutError) == (
                status == 500 and (recoverable or error_type == "TaskEnvError")
            )
            if isinstance(caught.value, RecoverableRolloutError):
                assert str(caught.value) == "Native Ava generate: original native failure"
        finally:
            await runner.cleanup()
        assert len(calls) == expected_calls

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
