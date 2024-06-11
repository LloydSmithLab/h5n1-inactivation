#!/usr/bin/env python3

"""
Clean data and prepare it for inference
"""

import argparse
from typing import Iterable

import pandas as pd
import polars as pl


def parse_plate(
    excel_file: str,
    plate_header_row: int,
    plate_metadata_row: int = None,
    plate_n_rows: int = 8,
    **kwargs
) -> (pd.DataFrame, pd.DataFrame):
    """
    Parse a 96 well plate and any metadata
    about it to a tuple of Pandas dataframes.

    Parameters
    ----------
    excel_file : str
        Path to an excel file to parse
    plate_header_row : int
        Row of the excel file containing the plate header
    plate_metadata_row : int
        Row of metadata about the plate
    plate_n_rows : int
        Number of non-header rows per plate. Default
        8, for a 8x12 96-well plate.
    **kwargs
        Other keyword arguments passed to pandas.read_excel

    Returns
    -------
    Tuple of Pandas DataFrames, the first representing the parsed plate,
    the other holding parsed metadata, if any.
    """
    plate = pd.read_excel(
        excel_file,
        skiprows=0,
        header=plate_header_row,
        nrows=plate_n_rows,
        **kwargs
    )

    if plate_metadata_row is not None:
        metadata = pd.read_excel(
            excel_file,
            skiprows=plate_metadata_row,
            nrows=1,
            header=None,
            **kwargs
        ).dropna(axis="columns")

    return (plate, metadata)


def validate_plate_shape(
    parsed_plate: pl.DataFrame, expected_shape=(8, 13)
) -> None:
    """
    Validate that a parsed plate has the expected
    shape.

    Parameters
    ----------
    parsed_plate : pl.DataFrame
        Parsed plate to validate, as a polars DataFrame.
    expected_shape : expected shape for the parsed plate.
        Default (8, 14): 8 row x 12 column plate, with
        auxiliary columns for the row letter label
        and the total number of positive wells.

    Returns
    -------
    None

    Raises
    ------
    ValueError if given a plate of unexpected shape.
    """
    observed_shape = parsed_plate.shape
    if not observed_shape == expected_shape:
        raise ValueError(
            "Unexpected parsed plate shape {}; "
            "expected {}."
            "".format(observed_shape, expected_shape)
        )


def validate_longform_data(
    data_long: pl.DataFrame,
) -> None:
    """
    Validate that data pivoted to long
    has the expected format and entries
    corresponding to a parsed 96-well plate.

    Parameter
    ---------
    data_long : pl.DataFrame
        Polars DataFrame to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError if validation conditions are not met,
    with an explanation of the unmet condition.
    """
    if not all(
        data_long["well_column"].is_in(range(1, 13))
    ):
        raise ValueError(
            "Unexpected well column indices; "
            "expected only integers in the range "
            "1 through 12"
        )
    if not all(
        data_long["log10_dilution"].is_in(range(-8, 1))
    ):
        raise ValueError(
            "Unexpected log10 dilution factor; "
            "expected only integers in the range "
            "0 through -8"
        )
    if not all(data_long["replicate"].is_in(range(1, 4))):
        raise ValueError(
            "Unexpected replicate number; "
            "expected only integers in the range "
            "1 through 3"
        )


def clean_single_plate(
    parsed_plate: pl.DataFrame | pd.DataFrame,
) -> pl.DataFrame:
    """
    Clean a single 96-well plate, and pivot
    it to longform tidy data, with validation.

    Parameters
    ----------
    parsed_plate : pandas or polars DataFrame or other object coerceible to a
                   polars DataFrame
        Data to clean representing a single 96-well plate, as the output of
        parse_plate()[0].

    Returns
    -------
    A long-form tidy polars DataFrame representing the plate.

    Raises
    ------
    ValueError if validation of the cleaned longform data fails.
    """
    plate = pl.DataFrame(parsed_plate)
    validate_plate_shape(plate, (8, 14))
    well_columns = ["{}".format(x) for x in range(1, 13)]
    plate.columns = (
        ["well_row"] + well_columns + ["n_positive_wells"]
    )
    plate = plate.drop(
        ["n_positive_wells"]
    ).with_row_index("minus_log10_dilution")
    plate_long = (
        plate.melt(
            id_vars=["minus_log10_dilution", "well_row"],
            value_name="well_status",
            variable_name="well_column",
        )
        .with_columns(
            well_status=pl.when(
                pl.col("well_status") == "+"
            )
            .then(pl.lit(True))
            .when(pl.col("well_status") == "-")
            .then(pl.lit(False))
            .otherwise(None),
            well_column=pl.col("well_column").cast(
                pl.Int32
            ),
        )
        .with_columns(
            replicate=((pl.col("well_column") - 1) / 4)
            .floor()
            .cast(pl.Int32)
            + 1,
            log10_dilution=-(
                pl.col("minus_log10_dilution").cast(
                    pl.Int32
                )
            ),
        )
    )

    validate_longform_data(plate_long)

    return plate_long


def parse_duration_to_minutes(duration_raw: str) -> float:
    """
    Parse a duration string in the form
    '<number><3 letter unit>', stripping away
    any whitespace, return the duration value
    as a float in units of minutes.
    As this function is specific to cleaning a
    particular dataset, 'min'` and `'sec' are the
    only parseable input units.

    Parameters
    ----------
    duration_raw : str
        Duration string in the form
        '<number><3 letter unit>', potentially with leading
        and/or trailing whitespace, and potentially with
        whitespace between the duration and the unit. Valid examples
        include: '30 sec', '1sec' '52   sec ', ' 5 min ', and ' 22min'.

    Returns
    -------
    The parsed duration as a float in units of minutes.

    Raises
    ------
    ValueError if the unit component of the string cannot be
    parsed as a known/supported unit or if the value component
    cannot be coerced to a float.
    """
    duration_stripped = duration_raw.strip()
    unit = duration_stripped[-3:]
    value = duration_stripped[:-3]

    conversions = {"min": 1, "sec": 1 / 60.0}

    conversion = conversions.get(unit, None)
    if conversion is None:
        raise ValueError(
            "Unknown string format for timepoint; could not parse {}"
            "as a time unit in string {}".format(
                unit, duration_raw
            )
        )
    return float(value) * conversion


def parse_temperature_to_celsius(
    temperature_raw: str,
) -> float:
    """
    Parse a temperature string in the form
    '<number><1 letter unit>', e.g. '30C',
    '35 C', stripping away whitespace, and
    return the temperature value as a float
    in units of Celsius. As this function is
    specific to cleaning a particular dataset,
    'C' is the only accepted input unit.


    Parameters
    ----------
    temperature_raw : str
        temperature string in the form
        '<number><1 letter unit>',potentially with leading
        and/or trailing whitespace, and potentially with
        whitespace between the duration and the unit.
        Valid examples include '30C ','35 C', ' -4.25   C '

    Returns
    -------
    The parsed temperature as a float in units of Celsius.

    Raises
    ------
    ValueError if the unit component of the string cannot be
    parsed as a known/supported unit or if the value component
    cannot be coerced to a float.
    """
    temp_stripped = temperature_raw.strip()
    unit = temp_stripped[-1:]
    value = temp_stripped[:-1]

    conversions = {
        "C": 1,
    }

    conversion = conversions.get(unit, None)
    if conversion is None:
        raise ValueError(
            "Unknown string format for temperature; could not parse {}"
            "as a temperature unit.".format(unit)
        )
    return float(value) * conversion


def parse_titration_data(
    excel_file: str,
    plate_header_rows: Iterable[int],
    metadata_row_offset: int,
    sample_id_prefix: str = "sample",
    temperature: str = None,
    verbose: bool = False,
    **kwargs
) -> pl.DataFrame:
    """
    Parse an entire Excel sheet of titration data
    to a single tidy polars DataFrame, with validation.

    Parameters
    ----------
    excel_file : str
        Path to an excel file to parse.

    plate_header_rows : Iterable[int]
        Iterable set of integers representing
        the header rows within the excel sheet
        of individual 96-well plates.

    metadata_row_offset : int
        Where is the plate metadata row
        relative to the header row?
        (This differs among datasheets,
        but it is typically 0 (same row)
        or 1 (one row above)

    sample_id_prefix : str
        Prefix for the string sample unique ids. Default
        "sample".

    verbose : bool
        Print which plate is currently being
        parsed? Default False.

    temperature : str
        Temperature for all data in the
        sheet, if shared among all data and known,
        as a string parseable by parse_temperature_to_celsius().
        Otherwise, will attempt to parse temperature
        metadata from plate header rows.

    **kwargs :
        Additional keyword arguments passed to
        pandas.read_excel()

    Returns
    -------
    The parsed data, as a long-form tidy
    polars DataFrame.

    Raises
    ------
    A ValueError if validation fails.
    """
    results = [None] * len(plate_header_rows)

    for i_plate, header_row in enumerate(
        plate_header_rows
    ):
        if verbose:
            print("Parsing plate {}".format(i_plate))
        metadata_row = header_row + metadata_row_offset
        plate, meta = parse_plate(
            excel_file,
            plate_header_row=header_row,
            plate_metadata_row=metadata_row,
            **kwargs
        )

        if verbose:
            print("Parsed plate:")
            print(plate)
            print("Parsed metadata:")
            print(meta)

        if temperature is None:
            timepoint = meta.iloc[0][0].strip()
            temp = meta.iloc[0][1].strip()
        else:
            timepoint = meta.iloc[0][0].strip()
            temp = temperature
            # some sheets do not explicitly name
            # the temparture, others do.
        plate = clean_single_plate(plate).with_columns(
            timepoint_minutes=pl.lit(
                parse_duration_to_minutes(timepoint)
            ),
            temperature_celsius=pl.lit(
                parse_temperature_to_celsius(temp)
            ),
            sample_id=(
                pl.lit(
                    "{}-t{}-{}-rep".format(
                        sample_id_prefix,
                        temp.replace(" ", "-"),
                        timepoint.replace(" ", "-"),
                    )
                )
                + pl.col("replicate").cast(pl.Utf8)
            ),
        )
        results[i_plate] = plate

    all_results = pl.concat(results)
    validate_longform_data(all_results)
    return all_results


def main(
    excel_file_path_63C_and_pilot: str,
    excel_file_path_72C: str,
    save_path: str,
    separator="\t",
) -> None:
    """
    Read in two files worth of Excel formatted
    titration data, clean them, and save them as a
    single tidy delimited text file (default .tsv).

    Parameters
    ----------
    excel_file_path_63C_and_pilot: str
        Path to the excel file to read in with 63C data
        and 72C pilot

    excel_file_path_72C: str
        Path to excel file to read in with 72C data

    save_path : str
        path to save the output.

    separator : str
        Separator string for the delimited
        text file. Default '\t'
        (tab-delimited / .tsv).

    Returns
    -------
    None
    """
    # constants not encoded in the raw data sheet
    virus_name = "H5N1_mountain_lion_isolate"
    medium_name = "milk"
    well_volume = 0.1
    sample_id_prefix = "{}-{}".format(
        virus_name, medium_name
    )
    usecols = "A:N"

    parsed_63 = (
        parse_titration_data(
            excel_file_path_63C_and_pilot,
            [7 + 11 * x for x in range(13)],
            -1,
            sample_id_prefix=sample_id_prefix,
            usecols=usecols,
        )
        .filter(
            pl.col("timepoint_minutes")
            > 0
            # do not use 0 timepoint for 63C
            # as it is before ramp-up
        )
        .with_columns(is_pilot=pl.lit(False))
    )

    parsed_72_pilot = parse_titration_data(
        excel_file_path_63C_and_pilot,
        [
            153 + 11 * x for x in range(8)
        ],  # 72C pilot experiment
        -1,
        sample_id_prefix=sample_id_prefix + "-pilot",
        usecols=usecols,
    ).with_columns(is_pilot=pl.lit(True))

    parsed_72 = parse_titration_data(
        excel_file_path_72C,
        [19 + 11 * x for x in range(7)],
        -1,
        sample_id_prefix=sample_id_prefix,
        usecols=usecols,
        temperature="72C",
    ).with_columns(is_pilot=pl.lit(False))

    parsed = pl.concat(
        [parsed_63, parsed_72, parsed_72_pilot]
    )

    # filter out wells that weren't used
    # and add metadata
    dat = (
        parsed.drop_nulls(["well_status"])
        .with_columns(
            virus=pl.lit(virus_name),
            medium=pl.lit(medium_name),
            well_volume_ml=pl.lit(well_volume),
        )
        .with_columns(
            condition_id=(
                pl.col("virus")
                + pl.lit("-")
                + pl.col("medium")
                + pl.lit("-")
                + pl.col("temperature_celsius").cast(
                    pl.Utf8
                )
                + pl.lit("C")
            )
        )
        .with_columns(
            condition_id=pl.when(pl.col("is_pilot"))
            .then(pl.col("condition_id") + "-pilot")
            .otherwise(pl.col("condition_id"))
        )
        .select(
            "virus",
            "medium",
            "temperature_celsius",
            "timepoint_minutes",
            "replicate",
            "log10_dilution",
            "well_status",
            "well_volume_ml",
            "condition_id",
            "sample_id",
            "is_pilot",
        )
    )

    dat.write_csv(save_path, separator=separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in Excel formatted titration data, "
            "clean it, and save it as a tidy delimited "
            "text file (default .tsv)."
        )
    )
    parser.add_argument(
        "excel_file_path_63C_and_pilot",
        type=str,
        help=(
            "Path to the Excel file to read in and clean "
            "containing 63C experiment data and pilot "
            "72C data"
        ),
    )
    parser.add_argument(
        "excel_file_path_72C",
        type=str,
        help=(
            "Path to the Excel file to read in and clean "
            "containing 72C experiment data"
        ),
    )
    parser.add_argument(
        "save_path",
        type=str,
        help="Path to save the cleaned data",
    )
    parser.add_argument(
        "--separator",
        type=str,
        help="Separator for the delimited text file",
        default="\t",
    )
    parsed = vars(parser.parse_args())
    main(
        parsed["excel_file_path_63C_and_pilot"],
        parsed["excel_file_path_72C"],
        parsed["save_path"],
        separator=parsed["separator"],
    )
