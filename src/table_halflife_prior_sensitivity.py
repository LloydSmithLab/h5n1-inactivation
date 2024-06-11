#!/usr/bin/env python3

import argparse
import os
import re

import polars as pl


def main(
    table_paths: list[str],
    output_path: str,
    separator: str = "\t",
) -> None:
    """
    Summarize an MCMC diagnostic table to get the extrema
    and associated parameters.

    Parameters
    ----------
    table_paths : list[str]
        List of paths to halflife tables for individual prior
        parameter sets.

    output_path : str
        Path to save the output table.

    separator : str
        Delimiter for the delimited text
        files specified in diagnostic_table_path and
        output_path. Default
        `\t` (tab-delimited).
    """

    all_tabs = []
    pattern = r"table_halflives_(.*).tsv"

    for tab_path in table_paths:
        tab = pl.read_csv(tab_path, separator=separator)
        prior_set_name = re.search(
            pattern, os.path.basename(tab_path)
        ).group(1)

        tab = tab.with_columns(
            prior_parameter_set=pl.lit(prior_set_name)
        )

        all_tabs.append(tab)

    result = pl.concat(all_tabs)

    result.write_csv(output_path, separator=separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in halflife tables and row bind, "
            "them, adding metadata on which prior "
            "parameter set was used"
        )
    )

    parser.add_argument(
        "table_paths",
        type=str,
        help=(
            "Whitespace-separated list of one or more paths "
            "to the halflife tables, "
            "(saved as a delimited text files)."
        ),
        nargs="+",
    )

    parser.add_argument(
        "-o",
        "--output-path",
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
        parsed["table_paths"],
        parsed["output_path"],
        separator=parsed["separator"],
    )
