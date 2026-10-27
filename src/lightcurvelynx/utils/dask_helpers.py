"""Helper functions for running simulations on dask clients."""

import os
import socket


def _worker_process():
    """Return the host name and process id of the process this function runs in.

    Run on every worker of a dask client, it tells which workers share a process.

    Returns
    -------
    tuple of (str, int)
        The host name and the process id.
    """
    return socket.gethostname(), os.getpid()


def max_threads_per_process(client):
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
