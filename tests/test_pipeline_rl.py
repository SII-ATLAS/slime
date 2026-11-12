"""CPU checks for the SGLang control protocol used by PipelineRL."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


@pytest.fixture
def engine_module(monkeypatch):
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    utils = types.ModuleType("sglang.srt.utils")
    utils.kill_process_tree = lambda *_: None
    monkeypatch.setitem(sys.modules, server_args.__name__, server_args)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = Path(__file__).resolve().parents[1] / "slime/backends/sglang_utils/sglang_engine.py"
    spec = importlib.util.spec_from_file_location("pipeline_rl_engine_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flush_cache", [False, True])
def test_pause_mode_and_disk_reload_preserve_cache_only_when_enabled(engine_module, monkeypatch, flush_cache):
    engine = engine_module.SGLangEngine(types.SimpleNamespace(), rank=0)
    engine.node_rank = 0
    engine.server_host = "engine"
    engine.server_port = 30000
    requests = []

    def post(url, *, json, timeout=None):
        assert timeout is None  # Weight reloads keep their normal unbounded wait.
        requests.append((url, json))
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"success": True})

    monkeypatch.setattr(engine_module.requests, "post", post)
    engine.pause_generation(mode="abort" if flush_cache else "in_place")
    engine.update_weights_from_disk("/weights", weight_version="2", flush_cache=flush_cache)
    engine.continue_generation()
    assert requests == [
        ("http://engine:30000/pause_generation", {"mode": "abort" if flush_cache else "in_place"}),
        (
            "http://engine:30000/update_weights_from_disk",
            {"model_path": "/weights", "flush_cache": flush_cache, "weight_version": "2"},
        ),
        ("http://engine:30000/continue_generation", {}),
    ]


def test_nonzero_engine_rank_does_not_send_control_requests(engine_module, monkeypatch):
    engine = engine_module.SGLangEngine(types.SimpleNamespace(), rank=1)
    engine.node_rank = 1
    monkeypatch.setattr(
        engine_module.requests, "post", lambda *a, **kw: pytest.fail("Only node rank zero controls serving")
    )
    engine.pause_generation()
    engine.update_weights_from_disk("/weights", weight_version="2")
    engine.continue_generation()


@pytest.mark.parametrize(
    "interval,expected",
    [(-1, [1]), (-100, [1]), (0, [1]), (1, [1, 2, 3, 4, 5, 6, 7]), (2, [1, 3, 5, 7]), (3, [1, 4, 7])],
)
def test_flush_schedule_counts_training_updates_and_handles_restored_versions(interval, expected):
    from slime.utils.weight_sync import should_flush_cache

    assert [version for version in range(1, 8) if should_flush_cache(interval, version)] == expected
    # Restoring a nonzero serving version keeps the same phase.
    assert [version for version in range(5, 8) if should_flush_cache(interval, version)] == [
        v for v in expected if v >= 5
    ]


@pytest.mark.parametrize("interval", [-1, 0, 1, 2, 3])
def test_first_publication_after_resume_flushes_health_check_cache(interval):
    from slime.utils.weight_sync import should_flush_cache

    assert should_flush_cache(interval, weight_version=6, initial_weight_version=5)
    assert should_flush_cache(interval, weight_version=7, initial_weight_version=5) == should_flush_cache(interval, 7)


def test_engine_reset_uses_consistent_collective_order_across_workers(engine_module):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace

    class WorkerGroups(set):
        def __init__(self, order):
            super().__init__(order)
            self.order = order

        def __iter__(self):
            return iter(name for name in self.order if name in self)

    names = ["pipeline-0", "pipeline-1", "pipeline-2"]
    collectives = {name: Barrier(2) for name in names}
    engines = []
    calls = [[], []]
    for index, order in enumerate([names, names[::-1]]):
        engine = engine_module.SGLangEngine.__new__(engine_module.SGLangEngine)
        engine.args = SimpleNamespace(rollout_health_check_timeout=2)
        engine._weight_update_groups = WorkerGroups(order)

        def request(route, payload, timeout, index=index):
            assert route == "destroy_weights_update_group"
            group = payload["group_name"]
            # Different per-worker set orders would leave peers waiting in
            # different shutdown collectives, as in real NCCL teardown.
            collectives[group].wait(timeout=timeout)
            calls[index].append(group)

        engine._make_request = request
        engines.append(engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda engine: engine.reset_weights_update_groups(), engines))
    assert calls == [names, names]
    assert all(not engine._weight_update_groups for engine in engines)


@pytest.mark.parametrize("start,interval,end", [(0, 1, 5), (0, 64, 130), (20, 64, 130)])
def test_training_publication_interval_preserves_updates_and_saves(start, interval, end):
    """Execute the real driver loop with synchronous Ray/model test doubles."""
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "train.py"
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "train")
    events, publications, saves, consumed_versions = [], [], [], []
    version = start

    def publish(**kwargs):
        nonlocal version
        version = kwargs.get("weight_version", version + 1)
        publications.append(version)

    def remote(function):
        return SimpleNamespace(remote=function)

    actor = SimpleNamespace(
        update_weights=publish,
        async_train=lambda step, data: events.append(step),
        clear_memory=lambda: None,
    )
    manager = SimpleNamespace(
        training_ready=remote(lambda: None),
        generate=remote(lambda step: consumed_versions.append((step, version))),
        training_completed=remote(lambda step: None),
        pause_rollout_admission=remote(lambda: False),
        resume_rollout_admission=remote(lambda previous: None),
    )
    args = SimpleNamespace(
        release_train=False, update_weights_interval=interval,
        offload_rollout=False, offload_train=False, check_weight_update_equal=False,
        num_rollout=end, start_rollout_id=start, eval_interval=None, use_critic=False, save_interval=20,
    )
    namespace = {
        "ray": SimpleNamespace(get=lambda value: value),
        "create_training_models": lambda *a: (actor, None),
        "should_run_periodic_action": lambda step, every, *a: every is not None and (step + 1) % every == 0,
        "save_checkpoint": lambda args, step, *a, **kw: saves.append((step, list(events))),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace["train"](args, None, manager, None, None)
    assert events == list(range(start, end))
    assert publications == [start + 1] + [step + 2 for step in events if (step + 1) % interval == 0]
    assert [step for step, _ in saves] == [step for step in events if (step + 1) % 20 == 0]
    assert all(history == list(range(start, step + 1)) for step, history in saves)
    if interval == 64:
        assert dict(consumed_versions)[63] == start + 1
        assert dict(consumed_versions)[64] == 65
        assert dict(consumed_versions)[127] == 65
        assert dict(consumed_versions)[128] == 129
        assert publications[-1] == 129  # No hidden final publication at step130.

def test_sparse_publications_keep_f8_cache_refresh_frequency():
    from slime.utils.weight_sync import should_flush_cache

    versions = [1 + 64 * i for i in range(18)]
    assert [v for v in versions if should_flush_cache(8, v, update_weights_interval=64)] == [1, 513, 1025]
    assert should_flush_cache(8, 21, initial_weight_version=20, update_weights_interval=64)
    assert not should_flush_cache(8, 65, initial_weight_version=20, update_weights_interval=64)

def test_sparse_publication_installs_training_version_on_native_actor():
    from contextlib import nullcontext
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/actor.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_weights")
    function.decorator_list = []
    sent = []
    updater = SimpleNamespace(weight_version=1)

    def update():
        updater.weight_version += 1
        sent.append(updater.weight_version)

    updater.update_weights = update
    actor = SimpleNamespace(
        args=SimpleNamespace(debug_train_only=False, debug_rollout_only=False, rollout_external=True,
                             use_fault_tolerance=False, offload_train=False, use_critic=False),
        weight_updater=updater,
        rollout_manager=SimpleNamespace(get_updatable_engines_and_lock=SimpleNamespace(
            remote=lambda: ([object()], None, 0, [], [], []))),
    )
    namespace = {"ray": SimpleNamespace(get=lambda value: value), "nullcontext": nullcontext,
                 "print_memory": lambda *a: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace["update_weights"](actor, weight_version=65)
    assert sent == [65]
    with pytest.raises(ValueError, match="must increase"):
        namespace["update_weights"](actor, weight_version=65)
    assert sent == [65]

def test_sparse_publication_staleness_uses_training_clock():
    from types import SimpleNamespace
    from slime.utils.staleness import compute_staleness_metrics
    from slime.utils.types import Sample

    path = Path(__file__).resolve().parents[1] / "slime/observability/rollout_metrics.py"
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "log_rollout_data")
    logged = []
    namespace = {
        "compute_staleness_metrics": compute_staleness_metrics,
        "dict_add_prefix": lambda values, prefix: {prefix + k: v for k, v in values.items()},
        "compute_metrics_from_samples": lambda *a: {}, "compute_perf_metrics_from_samples": lambda *a: {},
        "logger": SimpleNamespace(info=lambda *a: None), "compute_rollout_step": lambda args, step: step,
        "logging_utils": SimpleNamespace(log=lambda args, values, **kw: logged.append(values)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(custom_rollout_log_function_path=None, load_debug_rollout_data=None, update_weights_interval=64)
    sample = Sample(prompt="question", response_length=2, weight_versions=["1"])
    namespace["log_rollout_data"](63, args, [sample], {}, 1, weight_version=1)
    assert logged[-1]["rollout/staleness/mean"] == 63
    assert logged[-1]["rollout/staleness/publication_lag"] == 63
    namespace["log_rollout_data"](64, args, [sample], {}, 1, weight_version=65)
    assert logged[-1]["rollout/staleness/mean"] == 64  # An old queued sample remains old after publishing.
    assert logged[-1]["rollout/staleness/publication_lag"] == 0

@pytest.mark.parametrize("interval", [0, -1, 1, 64])
@pytest.mark.parametrize("unsupported", [False, True])
def test_publication_interval_native_validation(interval, unsupported):
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "slime/utils/arguments.py"
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "slime_validate_args")
    # Isolate the native interval validation from CUDA/Megatron parser dependencies.
    checks = [n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("args.update_weights_interval")]
    assert len(checks) == 2
    args = SimpleNamespace(
        update_weights_interval=interval, colocate=unsupported, offload_rollout=False,
        release_train=False, use_critic=False, update_weight_mode="full", update_weight_transport="nccl",
        rollout_function_path="slime.rollout.fully_async_rollout.generate_rollout_fully_async",
        debug_train_only=False, debug_rollout_only=False, load_debug_rollout_data=None, train_backend="megatron", eval_interval=None,
    )
    code = compile(ast.Module(body=checks, type_ignores=[]), str(path), "exec")
    if interval < 1 or (interval > 1 and unsupported):
        with pytest.raises(ValueError):
            exec(code, {"args": args})
    else:
        exec(code, {"args": args})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
