import argparse

import numpy as np

from lightcurvelynx.obstable.opsim import OpSim

_all_columns = [
    "observationId",
    "fieldRA",
    "fieldDec",
    "observationStartMJD",
    "visitExposureTime",
    "filter",
    "rotSkyPos",
    "numExposures",
    "airmass",
    "seeingFwhmEff",
    "seeingFwhmGeom",
    "skyBrightness",
]


def make_sampled_opsim(ra_vals, dec_vals, time_step, num_visits):
    """Create a fake OpSim data that scans a grid of (ra, dec) multiple times.

    Note
    ----
    This is a toy OpSim with lots of simplifying assumptions. It should not be
    used for scientific validation.

    Parameters
    ----------
    ra_vals : `numpy.ndarray`
       The values of RA to sample (in degrees).
    dec_vals : `numpy.ndarray`
       The values of dec to sample (in degrees).
    time_step : `float`
        The time between observations (in days).
    num_visits : `int`
        The number of times the full grid is sampled.
    """
    opsim_data = {
        "observationStartMJD": [],
        "fieldRA": [],
        "fieldDec": [],
    }

    for day in range(num_visits):
        t = 60676.0 + 5.0 * day  # Start date = Jan 1, 2025
        for ra in ra_vals:
            for dec in dec_vals:
                opsim_data["observationStartMJD"].append(t)
                opsim_data["fieldRA"].append(ra)
                opsim_data["fieldDec"].append(dec)
                t += time_step

    # Add other fields
    num_obs = len(opsim_data["observationStartMJD"])
    opsim_data["observationId"] = np.arange(num_obs)
    opsim_data["filter"] = np.repeat(["g", "r"], num_obs // 2)
    opsim_data["visitExposureTime"] = np.full((num_obs), 29.2)  # Most common value
    opsim_data["numExposures"] = np.full((num_obs), 2)  # Most common value
    opsim_data["airmass"] = np.full((num_obs), 1.3)  # Mean value
    opsim_data["seeingFwhmEff"] = np.full((num_obs), 1.12)  # Mean value
    opsim_data["seeingFwhmGeom"] = np.full((num_obs), 0.97)  # Mean value
    opsim_data["skyBrightness"] = np.full((num_obs), 20.0)  # Mean value

    for col in _all_columns:
        if col not in opsim_data:
            opsim_data[col] = np.zeros(num_obs)

    return OpSim(opsim_data)


def main():
    """Generate the fake OpSim data and save it to a file.

    To generate an updated small_db file use:
        python tests/lightcurvelynx/utils/make_fake_opsim.py tests/lightcurvelynx/data/opsim_small.db
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output filename")
    parser.add_argument(
        "--ra_start", type=float, default=0.0, help="The starting RA of the grid (in degrees)"
    )
    parser.add_argument("--ra_end", type=float, default=30.0, help="The end RA of the grid (in degrees)")
    parser.add_argument("--ra_step", type=float, default=3.0, help="The grid step size along RA (in degrees)")
    parser.add_argument(
        "--dec_start", type=float, default=-30.0, help="The starting dec of the grid (in degrees)"
    )
    parser.add_argument("--dec_end", type=float, default=0.0, help="The end dec of the grid (in degrees)")
    parser.add_argument(
        "--dec_step", type=float, default=3.0, help="The grid step size along dec (in degrees)"
    )
    parser.add_argument(
        "--time_step", type=float, default=0.003, help="The time between observations (in days)"
    )
    parser.add_argument(
        "--num_visits", type=int, default=3, help="The number of times the full grid is sampled"
    )
    args = parser.parse_args()

    # Create the sample points for RA and dec.
    ra_vals = np.arange(args.ra_start, args.ra_end, args.ra_step)
    dec_vals = np.arange(args.dec_start, args.dec_end, args.dec_step)
    print(f"Sampling along\n  ra={ra_vals}\n  dec={dec_vals}")

    # Create the OpSim and save it to a file.
    ops_data = make_sampled_opsim(ra_vals, dec_vals, args.time_step, args.num_visits)
    ops_data.write_opsim_table(args.output)


if __name__ == "__main__":
    main()
