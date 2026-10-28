import concurrent.futures
import multiprocessing

import pytest

from lightcurvelynx.utils.parallelization_utils import (
    assert_safe_executor,
    dask_max_threads_per_process,
)


def test_assert_safe_executor_threadpool():
    """Test using a ThreadPoolExecutor raises a ValueError."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError):
            assert_safe_executor(executor)


def test_assert_safe_executor_processpool():
    """Test using a ProcessPoolExecutor (with and without "fork")."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        assert_safe_executor(executor)

    # We should fail if we set the context to "fork".
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        with pytest.raises(ValueError):
            assert_safe_executor(executor)


def test_assert_safe_executor_dask():
    """Test using a Dask client with various configurations."""
    distributed = pytest.importorskip("dask.distributed")

    # Fail if processes=False (i.e., using threads instead of separate processes)
    with distributed.Client(
        dashboard_address=None,
        n_workers=2,
        threads_per_worker=1,
        processes=False,
    ) as client:
        with pytest.raises(ValueError):
            assert_safe_executor(client)

    # Fail if threads_per_worker > 1
    with distributed.Client(
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=2,
        processes=True,
    ) as client:
        with pytest.raises(ValueError):
            assert_safe_executor(client)

    # Success case: one worker with one thread in its own process
    with distributed.Client(
        dashboard_address=None,
        n_workers=1,
        threads_per_worker=1,
        processes=True,
    ) as client:
        assert_safe_executor(client)


@pytest.mark.parametrize(
    "cluster_args, expected",
    [
        # One worker with one thread, in this process.
        ({"n_workers": 1, "threads_per_worker": 1, "processes": False}, 1),
        # Two workers with one thread each, each in its own process.
        ({"n_workers": 2, "threads_per_worker": 1, "processes": True}, 1),
        # One worker with two threads.
        ({"n_workers": 1, "threads_per_worker": 2, "processes": False}, 2),
        # Two workers with one thread each, both in this process. Their addresses contain this
        # process's id, so they would be easy to tell apart by address alone...
        ({"n_workers": 2, "threads_per_worker": 1, "processes": False}, 2),
        # ...but with tcp addresses (host and port) they are not.
        ({"n_workers": 2, "threads_per_worker": 1, "processes": False, "protocol": "tcp"}, 2),
    ],
)
def test_max_threads_per_process(cluster_args, expected):
    """Count the worker threads in the busiest process of a real dask client."""
    distributed = pytest.importorskip("dask.distributed")
    with distributed.Client(dashboard_address=None, **cluster_args) as client:
        assert dask_max_threads_per_process(client) == expected


class _FakeClient:
    """Stands in for a dask client, with the two methods ``max_threads_per_process`` calls.

    Parameters
    ----------
    processes : dict
        What ``run()`` returns: each worker's address mapped to its (host name, process id).
    nthreads : dict
        What ``nthreads()`` returns: each worker's address mapped to its number of threads.
    """

    def __init__(self, processes, nthreads):
        self._processes = processes
        self._nthreads = nthreads

    def run(self, function):
        """Return the given (host name, process id) of every worker."""
        return dict(self._processes)

    def nthreads(self):
        """Return the given number of threads of every worker."""
        return dict(self._nthreads)


def test_max_threads_per_process_same_pid_on_different_hosts():
    """Workers on different hosts are in different processes, even with the same process id."""
    client = _FakeClient(
        processes={"tcp://10.0.0.1:8786": ("node-a", 7), "tcp://10.0.0.2:8786": ("node-b", 7)},
        nthreads={"tcp://10.0.0.1:8786": 1, "tcp://10.0.0.2:8786": 1},
    )
    assert dask_max_threads_per_process(client) == 1


def test_max_threads_per_process_late_workers():
    """Workers that joined after run() returned each count as their own process."""
    client = _FakeClient(
        processes={"tcp://10.0.0.1:8786": ("node-a", 7)},
        nthreads={"tcp://10.0.0.1:8786": 1, "tcp://10.0.0.2:8786": 1, "tcp://10.0.0.3:8786": 1},
    )
    assert dask_max_threads_per_process(client) == 1
