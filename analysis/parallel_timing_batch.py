"""A basic script to run the parallel timing tests as a batch.

Example:
python parallel_timing_batch.py \
    --opsim_file='../data/opsim/baseline_v5.1.1_10yrs.db' \
    --passband_table_dir='../data/passbands/LSST' \
    --threads='1,2,4,8' \
    --samples='10000,20000,30000,40000,50000' \
    --num_simulations=10 \
    --output='parallel_timing_results.npz'
"""

import argparse
import functools
import timeit

import numpy as np
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ApproximateMOCSampler
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.obstable.opsim import OpSim, OpSimPoissonFluxNoiseModel
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.utils.extrapolate import LinearDecay


def run_timing_tests(args):
    """Run the timing tests based on the input arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The input arguments.
    """
    # Load the OpSim data and passband group.
    print(f"Loading OpSim data from {args.opsim_file}...")
    ops_data = OpSim.from_db(args.opsim_file)
    noise_model = OpSimPoissonFluxNoiseModel()
    print(f"Loaded OpSim data with {len(ops_data)} entries.")

    t_min, t_max = ops_data.time_bounds()
    print(f"Time bounds of OpSim data: {t_min} to {t_max}")

    print("Loading passband group 'LSST'...")
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=args.passband_table_dir,
    )
    print(f"Loaded passband group with {len(passband_group)} passbands.")

    # Construct the model.
    print("Constructing the SN model...")
    ra_dec_sampler = ApproximateMOCSampler.from_obstable(ops_data, depth=6)
    source = SncosmoWrapperModel(
        "salt2-h17",
        t0=NumpyRandomFunc("uniform", low=t_min, high=t_max),
        x0=1.0e-6,
        x1=0.5,
        c=0.2,
        ra=ra_dec_sampler.ra,
        dec=ra_dec_sampler.dec,
        redshift=0.01,
        node_label="source",
        time_extrapolation=LinearDecay(decay_width=50.0),
    )

    # Set up the lists of trials to run.
    num_samples_list = [int(x) for x in args.samples.split(",")]
    num_threads_list = [int(x) for x in args.threads.split(",")]
    num_simulations = args.num_simulations
    all_times = np.zeros((len(num_samples_list), len(num_threads_list), num_simulations))
    print("Simulation:")
    print(f"  Samples={num_samples_list}")
    print(f"  Threads={num_threads_list}")
    print(f"  Num simulations={num_simulations}")

    # Loop through each trial and time the simulations for each combination of num_samples and num_threads.
    for sample_idx, num_samples in enumerate(num_samples_list):
        for thread_idx, num_threads in enumerate(num_threads_list):
            # Create a partial function for these parameters.
            simulate_func = functools.partial(
                simulate_lightcurves,
                model=source,
                num_samples=num_samples,
                obstable=ops_data,
                passbands=passband_group,
                noise_model=noise_model,
                num_jobs=num_threads,
                obs_time_window_offset=(-100, 400),
                progress_bar=False,  # Disable progress bar
            )

            # Time each of the simulations and store the results.
            for trial in range(num_simulations):
                t = timeit.timeit(simulate_func, number=1)
                all_times[sample_idx, thread_idx, trial] = t

                # Display each timing result so users can follow along.
                print(f"Time for num_samples={num_samples}, num_threads={num_threads}: {t:.2f} seconds")

            # Output summary timing results per combination of num_samples and num_threads.
            total_time = np.sum(all_times[sample_idx, thread_idx])
            average_time = np.mean(all_times[sample_idx, thread_idx])
            print(
                f"Time (sec) for num_samples={num_samples}, num_threads={num_threads}: "
                f"Total={total_time:.2f}. Average={average_time:.2f}"
            )

    # Save the timing results to a .npz file for later analysis.
    np.savez(
        args.output, all_times=all_times, num_samples_list=num_samples_list, num_threads_list=num_threads_list
    )
    print(f"Saved timing results to {args.output}")


def main():
    """The main function to run the timing tests."""
    parser = argparse.ArgumentParser(
        prog="lightcurvelynx_parallel_timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--opsim_file",
        help="Path to the ObsTable file containing the OpSim data.",
        default="../data/opsim/baseline_v5.1.1_10yrs.db",
    )
    parser.add_argument(
        "--passband_table_dir",
        help="Directory containing the passband tables.",
        default="../data/passbands/LSST",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4",
        help="A comma separated list of the number of threads to test.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="10000,20000,30000,40000,50000",
        help="A comma separated list of the number of samples to test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parallel_timing_results.npz",
        help="Path to save the timing results as a .npz file.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10,
        help="Number of simulations to run for each combination of threads and samples.",
    )

    # Run the actual program.
    args = parser.parse_args()
    run_timing_tests(args)


if __name__ == "__main__":
    main()
