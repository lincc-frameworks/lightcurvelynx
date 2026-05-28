#!/usr/bin/env python

import argparse
import sqlite3
from pathlib import Path
from shutil import copyfileobj

import fsspec


def make_opsim_shorten(opsim_path: str, n_rows: int, output_path: str):
    """Create a shortened version of an OpSim file.

    Parameters
    ----------
    opsim_path : `str`
        The fsspec-recognizable path to the OpSim SQLite file.
    n_rows : `int`
        The number of rows to keep in the shortened version.
    output_path : `str`
        The local path to save the shortened version.
    """
    # Save the whole opsim database to a new file
    with fsspec.open(opsim_path, mode="rb") as input_file:
        with open(output_path, "wb") as output_file:
            copyfileobj(input_file, output_file)

    table_to_halt = "observations"
    temp_table = f"temp_{table_to_halt}"
    with sqlite3.connect(output_path) as conn:
        cursor = conn.cursor()

        # Select and delete all tables but "observations"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (table,) in cursor.fetchall():
            if table != table_to_halt:
                cursor.execute(f"DROP TABLE {table};")

        # Create a temporary table to store the shortened version of "observations"
        cursor.execute(
            f"""
                   CREATE TABLE {temp_table} AS
                   SELECT * FROM {table_to_halt} LIMIT ?;
               """,
            (n_rows,),
        )
        # Drop the original table
        cursor.execute(f"DROP TABLE {table_to_halt};")
        # Rename the temporary table to the original table
        cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_to_halt};")

        # Clean up
        cursor.execute("VACUUM;")

        conn.commit()


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
    make_opsim_shorten(opsim_path=args.input, n_rows=args.n_rows, output_path=args.output)


if __name__ == "__main__":
    main()
