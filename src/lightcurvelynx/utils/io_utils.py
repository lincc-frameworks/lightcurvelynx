import gzip
import logging
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table


class SquashOutput:
    """Context manager to temporarily squash all output to stdout and stderr.

    Optionally, stdout can be redirected to a logger instead of being suppressed.

    Parameters
    ----------
    stdout_to_log : bool, optional
        If True, stdout and stderr will be redirected to the logger instead of
        being suppressed.
        Default is False.
    logger : logging.Logger, optional
        The logger to which to redirect the output if stdout_to_log is True.
        If None, the root logger will be used. Default is None.
    log_level : int, optional
        The logging level to use when redirecting stdout and stderr to the logger.
        Default is logging.DEBUG.

    Examples
    --------
    >>> with SquashOutput():
    >>>     # Code that produces unwanted output
    >>>     ...

    >>> with SquashOutput(stdout_to_log=True):
    >>>     # Printed output is sent to logger.debug
    >>>     ...
    """

    class _LoggerWriter:
        """File-like wrapper that forwards writes to a logger."""

        def __init__(self, logger, level=logging.DEBUG):
            self._logger = logger
            self._level = level

        def write(self, message):
            for line in message.rstrip().splitlines():
                if line:
                    self._logger.log(self._level, line)
            return len(message)

        def flush(self):
            return None

    def __init__(self, stdout_to_log=False, logger=None, log_level=logging.DEBUG):
        self._stdout_to_log = stdout_to_log
        self._log_level = log_level
        self._logger = logger if logger is not None else logging.getLogger()
        self._original_stdout = None
        self._original_stderr = None
        self._null_file = None
        self._stdout_logger = None

    def __enter__(self):
        # Save the original stdout and stderr streams.
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Redirect output streams.
        if self._stdout_to_log:
            # Create a log writter if needed, and redirect stdout and stderr to it.
            if self._stdout_logger is None:
                self._stdout_logger = SquashOutput._LoggerWriter(
                    self._logger,
                    level=self._log_level,
                )
            sys.stdout = self._stdout_logger
            sys.stderr = self._stdout_logger
        else:
            # Open the null file if we haven't already, and redirect stdout and stderr to it.
            if self._null_file is None:
                self._null_file = open(os.devnull, "w")  # noqa: SIM115
            sys.stdout = self._null_file
            sys.stderr = self._null_file

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original stdout and stderr streams.
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        # Close the null file (if we opened one).
        if self._null_file is not None:
            self._null_file.close()  # noqa: SIM115
            self._null_file = None


class SquashLogging:
    """A context manager to temporarily squash all logging (below a certain level)
    to a logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which to redirect the logging output.
    level : int, optional
        The logging level below which to suppress logging output. Default: logging.ERROR

    Examples
    --------
    >>> with SquashLogging(logger=logging.getLogger(), level=logging.ERROR):
    >>>     # Code that produces unwanted logging at any level below logging.ERROR
    >>>     ...
    """

    def __init__(self, logger, level=logging.ERROR):
        if logger is None:
            raise ValueError("Logger must be provided to SquashLogging.")
        self._logger = logger
        self._old_level = None
        self._new_level = level

    def __enter__(self):
        # Save the original logging level and set the new level.
        self._old_level = self._logger.level
        self._logger.setLevel(self._new_level)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original logging level.
        self._logger.setLevel(self._old_level)


def write_results_as_hats(base_catalog_path, results, *, catalog_name=None, overwrite=False):
    """Write results to a HATS catalog.

    Parameters
    ----------
    base_catalog_path : str or Path
        The base path to the output hats directory.
    results : nested_pandas.NestedFrame
        The results to write, as a NestedFrame where each row is a sample.
    catalog_name : str, optional
        The name of the catalog to write. If None, the name will be derived from the
        base_catalog_path. Default: None
    overwrite : bool
        Whether to overwrite the output directory if it already exists.
        Default: False
    """
    base_catalog_path = Path(base_catalog_path)
    logging.debug(f"Writing results as HATS Catalog to {base_catalog_path}")

    # See if the (optional) LSDB package is installed.
    try:
        from lsdb import from_dataframe
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "The lsdb package is required to write results as HATS files. "
            "Please install it via 'pip install lsdb'."
        ) from err

    # Convert the results into an LSDB Catalog and output that. We just generate and output
    # the basic catalog (no margins or other extras).
    catalog = from_dataframe(results, ra_column="ra", dec_column="dec")
    catalog.write_catalog(
        base_catalog_path,
        catalog_name=catalog_name,
        as_collection=False,
        overwrite=overwrite,
    )


def read_numpy_data(file_path):
    """Read in a numpy array from different formats depending on the file extension.
    Automatically detects and handles files in .npy, .npz, .csv, .ecsv, and .txt
    formats.

    Parameters
    ----------
    file_path : str
        The path to the file to read.

    Returns
    -------
    data : numpy.ndarray
        The data read from the file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found.")

    # Load the data according to the format.
    if file_path.suffix == ".npy":
        data = np.load(file_path)
    elif file_path.suffix == ".npz":
        # For npz files, extract the first array
        data = np.load(file_path)["arr_0"]
    elif file_path.suffix in [".csv", ".ecsv"]:
        data = np.loadtxt(file_path, delimiter=",", comments="#")
    elif file_path.suffix in [".txt"]:
        data = np.loadtxt(file_path, comments="#")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}.")

    return data


def write_numpy_data(file_path, data):
    """Write a numpy array to a file in a format determined by the file extension.
    Automatically detects and handles files in .npy, .npz, .csv, .ecsv, and .txt
    formats.

    Parameters
    ----------
    file_path : str
        The path to the file to write.
    data : numpy.ndarray
        The data to write to the file.
    """
    file_path = Path(file_path)
    if file_path.suffix == ".npy":
        np.save(file_path, data)
    elif file_path.suffix == ".npz":
        np.savez_compressed(file_path, arr_0=data)
    elif file_path.suffix in [".csv", ".ecsv"]:
        np.savetxt(file_path, data, delimiter=",")
    elif file_path.suffix in [".txt", ".dat"]:
        np.savetxt(file_path, data)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}.")


def read_grid_data(input_file, format="ascii", validate=False):
    """Read 2-d grid data from a text, csv, ecsv, or fits file.

    Each line is of the form 'x0 x1 value' where x0 and x1 are the grid
    coordinates and value is the grid value. The rows should be sorted by
    increasing x0 and, within an x0 value, increasing x1.

    Parameters
    ----------
    input_file : str or file-like object
        The input data file.
    format : str
        The file format. Should be one of the formats supported by
        astropy Tables such as 'ascii', 'ascii.ecsv', or 'fits'.
        Default: 'ascii'
    validate : bool
        Perform additional validation on the input data.
        Default: False

    Returns
    -------
    x0 : numpy.ndarray
        A 1-d array with the values along the x-axis of the grid.
    x1 : numpy.ndarray
        A 1-d array with the values along the y-axis of the grid.
    values : numpy.ndarray
        A 2-d array with the values at each point in the grid with
        shape (len(x0), len(x1)).

    Raises
    ------
    ValueError 
        if any data validation fails.
    """
    logging.debug(f"Loading file {input_file} (format={format})")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    data = Table.read(input_file, format=format, comment=r"\s*#")
    if len(data.colnames) != 3:
        raise ValueError(
            f"Incorrect format for grid data in {input_file} with format {format}. "
            f"Expected 3 columns but found {len(data.colnames)}."
        )
    x0_col = data.colnames[0]
    x1_col = data.colnames[1]
    v_col = data.colnames[2]

    # Get the values along the x0 and x1 dimensions.
    x0 = np.sort(np.unique(data[x0_col].data))
    x1 = np.sort(np.unique(data[x1_col].data))

    # Get the array of values.
    if len(data) != len(x0) * len(x1):
        raise ValueError(
            f"Incomplete data for {input_file} with format {format}. Expected "
            f"{len(x0) * len(x1)} entries but found {len(data)}."
        )

    # If we are validating, loop through the entire table and check that
    # the x0 and x1 values are in the expected order.
    if validate:
        counter = 0
        for i in range(len(x0)):
            for j in range(len(x1)):
                if data[x0_col][counter] != x0[i]:
                    raise ValueError(
                        f"Incorrect x0 ordering in {input_file} at line={counter}."
                        f"Expected {x0[i]} but found {data[x0_col][counter]}."
                    )
                if data[x1_col][counter] != x1[j]:
                    raise ValueError(
                        f"Incorrect x1 ordering in {input_file} at line={counter}. "
                        f"Expected {x1[j]} but found {data[x1_col][counter]}."
                    )
                counter += 1

    # Build the values matrix.
    values = data[v_col].data.reshape((len(x0), len(x1)))

    return x0, x1, values


def _read_lclib_data_from_open_file(input_file):
    """Read SNANA's lclib data from a text file.

    Parameters
    ----------
    input_file : file
        The input data file containing SNANA's lclib data.

    Returns
    -------
    curves : list of astropy.table.Table
        A list of Astropy Tables, each representing a light curve.
    """
    colnames = []
    curves = []
    meta = {}
    current_model = {}
    parnames = []
    in_doc_block = False

    for l_num, line in enumerate(input_file):
        # Strip out the trailing comment. Then skip lines that are either
        # empty or do not contain a key-value pair.
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue

        # Split the line into key and value.
        key, value = line.split(":", 1)
        value = value.strip()

        # Handle the keys corresponding to a documentation block.
        if key == "DOCUMENTATION":
            in_doc_block = True
        elif key == "DOCUMENTATION_END":
            in_doc_block = False
        if in_doc_block:
            # If we are in a documentation block, just continue to the next line.
            continue

        if key == "COMMENT":
            continue  # Skip comments.
        elif key == "FILTERS":
            # Create a list of data columns with time and each filter.
            colnames = ["time"]
            for c in value:
                colnames.append(c)
        elif key == "END_EVENT":
            curr_id = meta.get("id", "")
            if curr_id != value:
                raise ValueError(f"Event ID mismatch (line {l_num}): found {value}, expected {curr_id}.")

            # Save the table we have so far.
            curves.append(Table(current_model, meta=meta))
        elif key == "START_EVENT":
            if len(colnames) == 0:
                raise ValueError(f"Error on line= {l_num}: No filters defined.")

            # Start a new light curve, but resetting the lists of data from the columns.
            current_model["type"] = []  # Initialize the type list.
            for col in colnames:
                current_model[col] = []
            meta["id"] = value
        elif key == "S" or key == "T":
            # Save an observation or template to the current light curve.
            current_model["type"].append(key)  # Get the type from the key.

            # Get the time and magnitudes from the columns.
            col_vals = value.split()
            if len(col_vals) != len(colnames):
                raise ValueError(f"Expected {len(colnames)} values on line={l_num}: {col_vals}")
            for col_idx, col in enumerate(colnames):
                current_model[col].append(float(col_vals[col_idx]))
        elif key == "MODEL_PARNAMES" or key == "MODEL_PARAMETERS":
            parnames = value.split(",")
        elif key == "PARVAL":
            if "," in value:
                all_vals = value.split(",")
            else:
                all_vals = value.split()

            if len(all_vals) != len(parnames):
                raise ValueError(f"Expected {len(parnames)} parameter values on line={l_num}: {all_vals}")
            meta["PARVAL"] = {key: value for key, value in zip(parnames, all_vals, strict=False)}
        else:
            # Save everything else to the meta dictionary.
            meta[key] = value

    return curves


def read_lclib_data(input_file):
    """Read SNANA's LCLIB data from a text file.

    Parameters
    ----------
    input_file : str or Path
        The path to the SNANA LCLIB data file.

    Returns
    -------
    curves : list of astropy.table.Table
        A list of Astropy Tables, each representing a light curve.
    """
    input_file = Path(input_file)
    logging.debug(f"Loading SNANA LCLIB data from {input_file}")
    if not input_file.is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    # Use the file suffix to determine how to read the file.
    suffix = input_file.suffix.lower()
    if suffix in [".gz", ".gzip"]:
        # Open as a gzipped text file.
        with gzip.open(input_file, "rt") as file_ptr:
            curves = _read_lclib_data_from_open_file(file_ptr)
    elif suffix in [".dat", ".txt", ".text"]:
        # Try to open the file as a regular text file.
        with open(input_file, "r") as file_ptr:
            curves = _read_lclib_data_from_open_file(file_ptr)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported file format: {suffix}.")

    return curves


def read_sqlite_table(db_path, table_name=None, sql_query=None):
    """Read a table from a SQLite database into a pandas DataFrame.

    Parameters
    ----------
    db_path : str or Path
        The path to the SQLite database file.
    table_name : str, optional
        The name of the table to read. If not provided, sql_query must be provided.
    sql_query : str, optional
        A custom SQL query to execute. If provided, this query will be used instead of reading a table.

    Returns
    -------
    df : pandas.DataFrame
        The table data as a pandas DataFrame.
    """
    if table_name is None and sql_query is None:  # pragma: no cover
        raise ValueError("Either table_name or sql_query must be provided.")
    if sql_query is None:
        sql_query = f"SELECT * FROM {table_name}"

    logger = logging.getLogger(__name__)
    logger.debug(f"Reading table {table_name} from SQLite database {db_path}")

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file {db_path} not found.")

    # Connect to the SQLite database.
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Read the specified table into a DataFrame.
    try:
        df = pd.read_sql_query(sql_query, con)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Error executing sql query '{sql_query}' from database {db_path}: {e}") from e

    # Close the database connection.
    con.close()

    return df
