#!/usr/bin/env python

import argparse
import sqlite3
from collections.abc import Collection
from pathlib import Path
from shutil import copy2

from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR
from lightcurvelynx.utils.data_download import download_data_file_if_needed


def make_opsim_shorten(
    *, opsim_url: str, n_rows: int, bands: Collection[str] = tuple("ugrizy"), output_path: str
) -> None:
    """Create a shortened version of an OpSim file.

    Parameters
    ----------
    opsim_url : `str`
        The URL to the OpSim SQLite file.
    n_rows : `int`
        The number of rows to keep in the shortened version.
    bands : str or list of str
        The bands to keep in the shortened version. Default is all LSST bands.
    output_path : `str`
        The local path to save the shortened version.
    """
    data_file_name = opsim_url.split("/")[-1]
    cached_path = _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR / "opsim" / data_file_name
    if not download_data_file_if_needed(cached_path, opsim_url):
        raise RuntimeError(f"Failed to download opsim data from {opsim_url}.")
    copy2(cached_path, output_path)

    table_to_halt = "observations"
    temp_table = f"temp_{table_to_halt}"
    with sqlite3.connect(output_path) as conn:
        cursor = conn.cursor()

        # Select and delete all tables but "observations"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (table,) in cursor.fetchall():
            if table != table_to_halt:
                cursor.execute(f"DROP TABLE {table};")

        # Create a temporary table with ~equal rows per band
        n_bands = len(bands)
        n_rows_per_band = n_rows // n_bands
        remainder = n_rows % n_bands
        cursor.execute(f"CREATE TABLE {temp_table} AS SELECT * FROM {table_to_halt} WHERE 0;")
        for i, band in enumerate(bands):
            limit = n_rows_per_band + (1 if i < remainder else 0)
            cursor.execute(
                f"""
                    INSERT INTO {temp_table}
                    SELECT * FROM {table_to_halt}
                    WHERE band = ?
                    ORDER BY observationStartMJD
                    LIMIT ?;
                """,
                (band, limit),
            )
        # Drop the original table
        cursor.execute(f"DROP TABLE {table_to_halt};")
        # Rename the temporary table to the original table
        cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_to_halt};")

        conn.commit()
        conn.execute("VACUUM;")


def parse_args(args):
    """Parse the command line arguments.

    Parameters
    ----------
    args : `list` of `str`
        The command line arguments.

    Returns
    -------
    `argparse.Namespace`
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Create a shortened version of an OpSim file.")
    parser.add_argument(
        "-i",
        "--input",
        default="https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs5.3/baseline/baseline_v5.3.0_10yrs.db",
        help="The fsspec-recognizable path to the OpSim SQLite file.",
    )
    parser.add_argument(
        "-n",
        "--n_rows",
        type=int,
        default=100,
        help="The number of rows to keep in the shortened version.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="gr",
        help="The bands to keep in the shortened version.",
    )
    default_output_path = Path(__file__).parent.parent / "data" / "opsim_shorten.db"
    parser.add_argument(
        "-o",
        "--output",
        default=str(default_output_path),
        help="The path to save the shortened version.",
    )
    return parser.parse_args(args)


def main(args=None):
    """Subsample an OpSim file and save it to a new file.

    Parameters
    ----------
    args : `list` of `str`, optional
        The command line arguments.
    """
    args = parse_args(args)
    make_opsim_shorten(opsim_url=args.input, n_rows=args.n_rows, bands=args.bands, output_path=args.output)


if __name__ == "__main__":
    main()
