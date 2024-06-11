#!/usr/bin/env python3

import argparse

import polars as pl

import analyze as ana


def main(
    data_path: str,
    titer_mcmc_path: str,
    halflife_mcmc_path: str,
    output_path: str,
    separator: str = "\t",
) -> None:
    """
    Create a tab-separated table of inferred
    titers for all titers with at least one positive
    well.

    Parameters
    ----------
    data_path : str
        Path to the data used to fit the model,
        as a delimited text file
        (default .tsv: tab-delimited, change this
        with the separator argument)

    titer_mcmc_path : str
        Path to the MCMC output for individual
        titer inference, saved as a .pickle archive.

    halflife_mcmc_path : str
        Path to the MCMC output for virus half-life
        inference, saved as a .pickle archive.

    output_path : str
        Path to which to save the table.

    separator : str
        Delimiter for the delimited text
        file specified in data_path. Default
        `\t` (tab-delimited).
    """

    tidy_results = ana.get_tidy_results(
        data_path,
        titer_mcmc_path,
        halflife_mcmc_path,
        include_pilot=True,
    )
    titers = tidy_results["titers"]

    # titers with 0 positive wells do not
    # have meaningful quantitative estimates
    titers = titers.filter(
        pl.col("total_positive_wells") > 0
    )
    tab = ana.median_qi_table(
        titers,
        ["log_titer"],
        [
            "temperature_celsius",
            "medium",
            "timepoint_minutes",
            "total_positive_wells",
            "sample_id",
            "condition_id",
            "is_pilot",
        ],
    ).sort(
        "medium",
        "is_pilot",
        "temperature_celsius",
        "timepoint_minutes",
        "sample_id",
    )

    tab.write_csv(output_path, separator="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in MCMC output and produce a table of "
            "titers to be reported in and/or accompany "
            "the manuscript"
        )
    )
    parser.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to the data used for fitting, "
            "formatted as "
            "a delimited text file"
        ),
    )
    parser.add_argument(
        "titer_mcmc_path",
        type=str,
        help=(
            "Path to the MCMC output for individual "
            "titer inference, saved as a .pickle archive."
        ),
    )
    parser.add_argument(
        "halflife_mcmc_path",
        type=str,
        help=(
            "Path to the MCMC output for virus "
            "half-life inference, saved as a .pickle archive."
        ),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Path to save the generated table."),
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        help=(
            "Separator for the delimited text file containing "
            "the data (specified in data_path)"
        ),
        default="\t",
    )
    parsed = vars(parser.parse_args())
    main(
        parsed["data_path"],
        parsed["titer_mcmc_path"],
        parsed["halflife_mcmc_path"],
        parsed["output_path"],
        separator=parsed["separator"],
    )
