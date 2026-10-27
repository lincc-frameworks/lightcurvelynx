"""Tests of worker preloading on every documented parallel backend.

Each backend uses at most two worker processes, and everything a test starts (pools,
clusters and ray) is shut down when the test ends, even if it fails.
"""

import contextlib
import multiprocessing
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import pytest

from lightcurvelynx.simulate import WorkerPreloadWarning, make_worker_initializer

BACKENDS = [
    "process_fork",
    "process_spawn",
    "mp_pool",
    "loky",
    "dask_processes",
    "ray_pool",
]


def _pool_factory(backend):
    """Return ``make_pool(initializer)``, which starts a two-worker pool of a pool-based backend.

    ``initializer=None`` gives a pool whose workers are not preloaded.
    """
    if backend in ("process_fork", "process_spawn"):
        context = multiprocessing.get_context(backend.removeprefix("process_"))
        return lambda init: ProcessPoolExecutor(2, mp_context=context, initializer=init)
    if backend == "mp_pool":
        return lambda init: multiprocessing.Pool(2, init)
    if backend == "loky":
        loky = pytest.importorskip("loky")
        return lambda init: loky.get_reusable_executor(max_workers=2, initializer=init)
    if backend == "ray_pool":
        ray_multiprocessing = pytest.importorskip("ray.util.multiprocessing")
        return lambda init: ray_multiprocessing.Pool(2, initializer=init)
    raise ValueError(f"Unknown backend {backend}")


@contextlib.contextmanager
def backend_context(backend):
    """Start a backend, yielding ``(baseline_executor, preloaded_executor_factory)``.

    The factory takes an initializer and returns a context manager for an executor whose
    workers are preloaded with it. For pool-based backends it first shuts down the baseline
    pool, so no more than two workers run at once. Everything started here is shut down on
    exit.

    Parameters
    ----------
    backend : str
        One of ``BACKENDS``.

    Yields
    ------
    baseline : executor
        An executor whose workers are not preloaded.
    make_preloaded : callable
        ``make_preloaded(initializer)`` returns a context manager for a preloaded executor.
    """
    with contextlib.ExitStack() as stack:
        if backend == "dask_processes":
            distributed = pytest.importorskip("distributed")
            cluster = stack.enter_context(
                distributed.LocalCluster(
                    n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
                )
            )
            client = stack.enter_context(distributed.Client(cluster))

            @contextlib.contextmanager
            def make_preloaded(init):
                client.register_worker_callbacks(init)
                yield client

            yield client, make_preloaded
            return

        if backend == "ray_pool":
            ray = pytest.importorskip("ray")
            ray.init(num_cpus=2, include_dashboard=False)
            stack.callback(ray.shutdown)
        make_pool = _pool_factory(backend)
        baseline_stack = stack.enter_context(contextlib.ExitStack())
        baseline = baseline_stack.enter_context(make_pool(None))

        def make_preloaded(init):
            baseline_stack.close()
            return make_pool(init)

        yield baseline, make_preloaded


@pytest.mark.parametrize("backend", BACKENDS)
def test_preloaded_backend_matches(backend, preload_inputs, assert_identical_results, run_preload_sim):
    """A preloaded run on each backend matches the non-preloaded run, without falling back."""
    model, si = preload_inputs
    with backend_context(backend) as (baseline, make_preloaded):
        with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
            expected = run_preload_sim(model, si, baseline)
        init = make_worker_initializer(model, si)
        with make_preloaded(init) as ex, warnings.catch_warnings():
            warnings.simplefilter("error", WorkerPreloadWarning)
            result = run_preload_sim(model, si, ex)
    assert_identical_results(expected, result)


def test_fork_pool_refuses_unpicklable(preload_inputs, run_preload_sim, make_stateful_model):
    """Data that refuses pickling raises before a fork pool exists, and on a non-preloaded one."""
    model, si = preload_inputs
    stateful_model = make_stateful_model(si)
    fork = multiprocessing.get_context("fork")
    with pytest.raises(RuntimeError, match="cannot be pickled"):
        with ProcessPoolExecutor(2, mp_context=fork, initializer=make_worker_initializer(stateful_model, si)):
            pass

    # A fork pool that was not preloaded still pickles every batch, so the run raises the same error.
    make_worker_initializer(model, si)  # Make the registry non-empty.
    with ProcessPoolExecutor(2, mp_context=fork) as executor:
        with pytest.warns(WorkerPreloadWarning, match="could not be pickled"):
            with pytest.raises(RuntimeError, match="cannot be pickled"):
                run_preload_sim(stateful_model, si, executor)


# A script that defines its model class in __main__, as a notebook does. Everything sits
# under the main guard, so worker processes that re-import the script (dask's spawned workers
# do) never define NotebookModel: as in a notebook, the class exists only in the parent and
# must reach the workers by value.
_NOTEBOOK_SCRIPT = """
import sys
import warnings
from pathlib import Path

import numpy as np

from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import WorkerPreloadWarning, make_worker_initializer, simulate_lightcurves
from lightcurvelynx.survey_info import SurveyInfo

if __name__ == "__main__":

    class NotebookModel(SEDModel):
        def compute_sed(self, times, wavelengths, graph_state, **kwargs):
            return np.full((len(times), len(wavelengths)), 100.0)

    backend, data_dir = sys.argv[1], Path(sys.argv[2])
    opsim = OpSim.from_db(data_dir / "opsim_small.db")
    passbands = PassbandGroup.from_preset(
        preset="LSST", table_dir=data_dir / "passbands", filters=["g", "r", "i", "z"]
    )
    si = SurveyInfo(obstable=opsim, passbands=passbands)
    model = NotebookModel(t0=0.0, ra=opsim["ra"][0], dec=opsim["dec"][0], redshift=0.0)
    warnings.simplefilter("error", WorkerPreloadWarning)
    init = make_worker_initializer(model, si)

    def run(executor):
        return simulate_lightcurves(
            model, 20, si, executor=executor, batch_size=5, rng=np.random.default_rng(100), progress_bar=False
        )

    if backend == "loky":
        import loky

        with loky.get_reusable_executor(max_workers=2, initializer=init) as executor:
            result = run(executor)
    elif backend == "dask":
        from distributed import Client, LocalCluster

        cluster_args = {"n_workers": 2, "threads_per_worker": 1, "processes": True, "dashboard_address": None}
        with LocalCluster(**cluster_args) as cluster, Client(cluster) as client:
            client.register_worker_callbacks(init)
            result = run(client)
    elif backend == "ray":
        import ray
        from ray.util.multiprocessing import Pool

        ray.init(num_cpus=2, include_dashboard=False)
        try:
            with Pool(2, initializer=init) as pool:
                result = run(pool)
        finally:
            ray.shutdown()
    print(f"ROWS {len(result)}")
"""


@pytest.mark.parametrize("backend", ["loky", "dask", "ray"])
def test_notebook_defined_class(backend, tmp_path, test_data_dir):
    """A model class defined in __main__ runs on preloaded loky, dask and ray workers."""
    pytest.importorskip({"loky": "loky", "dask": "distributed", "ray": "ray"}[backend])
    script = tmp_path / "notebook_model.py"
    script.write_text(_NOTEBOOK_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script), backend, str(test_data_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ROWS 20" in proc.stdout
