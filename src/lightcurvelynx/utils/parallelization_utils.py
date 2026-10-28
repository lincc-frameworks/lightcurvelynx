"""Helper functions for running simulations in parallel, including framework specific
utilities (e.g. Dask, concurrent.futures, joblib).
"""

import concurrent.futures
import os
import socket


class JoblibExecutor:
    """Minimal executor-like wrapper for simulate_lightcurves, which only calls .map().

    Attribute
    ----------
    parallel : joblib.Parallel
        The joblib.Parallel instance used for executing tasks.
    """

    def __init__(self, parallel):
        self.parallel = parallel

    def map(self, fn, iterable):
        """The basic mapping function for LightCurveLynx simulations."""
        try:
            from joblib import delayed
        except ImportError as err:
            raise ImportError("Joblib is required for JoblibExecutor.") from err

        return self.parallel(delayed(fn)(item) for item in iterable)


def assert_safe_executor(executor):
    """Assert that the given executor is safe to use with the models.
    This is an evolving list of incompatibilities to check for in different
    executors.

    Parameters
    ----------
    executor : concurrent.futures.Executor
        The executor to check.

    Raises
    ------
    AssertionError
        If the executor is not single-threaded.
    """
    if isinstance(executor, concurrent.futures.ThreadPoolExecutor):
        raise ValueError(
            "Some of the models have been found to be incompatible with ThreadPoolExecutor. "
            "We are working on a patch. Please use a different executor in the meantime."
        )

    # Check for a dask client that runs more than one thread in a process: either a worker
    # with several threads or several workers in one process.
    if hasattr(executor, "nthreads") and dask_max_threads_per_process(executor) > 1:
        raise ValueError(
            "Some of the models have been found to be incompatible with multi-threaded executors. "
            "Please create your Dask client with threads_per_worker=1 and without processes=False, "
            "so that each worker process runs a single thread."
        )

    # Fail for a ProcessPoolExecutor using fork.
    if (
        hasattr(executor, "_mp_context")
        and hasattr(executor._mp_context, "get_start_method")
        and executor._mp_context.get_start_method() == "fork"
    ):
        raise ValueError(
            "Some of the models are incompatible with ProcessPoolExecutor and 'fork'. "
            "Please use different settings."
        )


# -----------------------------------------------
# Dask-specific utilities
# -----------------------------------------------


def _worker_process():
    """Return the host name and process id of the process this function runs in.

    Run on every worker of a dask client, it tells which workers share a process.

    Returns
    -------
    tuple of (str, int)
        The host name and the process id.
    """
    return socket.gethostname(), os.getpid()


def dask_max_threads_per_process(client):
    """Return the largest number of worker threads that run in any one process of a dask client.

    A worker with several threads (``threads_per_worker > 1``) runs them all in its process, and a
    local cluster started with ``processes=False`` runs all of its workers in this process.

    Parameters
    ----------
    client : dask.distributed.Client
        The client whose workers to check.

    Returns
    -------
    int
        The largest total number of worker threads in one process (0 if there are no workers).
    """
    # client.run() gives each worker's (host name, process id), keyed by the worker's address.
    # Addresses identify workers, not processes: two workers in one process have different
    # addresses. A worker that joined after run() returned has no entry and counts as its own
    # process.
    processes = client.run(_worker_process)
    threads_per_process = {}
    for address, num_threads in client.nthreads().items():
        process = processes.get(address, address)
        threads_per_process[process] = threads_per_process.get(process, 0) + num_threads
    return max(threads_per_process.values(), default=0)
