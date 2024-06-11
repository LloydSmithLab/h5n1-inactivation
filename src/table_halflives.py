#!/usr/bin/env python3

import argparse

import numpy as np
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
    halflives for all experimental conditions

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
    hl_model = ana.load_mcmc(halflife_mcmc_path)[
        0
    ].run_model
    prior_dists = ana.extract_distribution_params(hl_model)
    hl_exp_loc = round(
        np.exp(
            prior_dists["log_halflife_distribution"]["loc"]
        ),
        1,
    )
    hl_exp_scale = round(
        np.exp(
            prior_dists["log_halflife_distribution"][
                "scale"
            ]
        ),
        1,
    )
    int_mode_loc = prior_dists["log_intercept_loc_prior"][
        "loc"
    ]
    int_mode_scale = prior_dists[
        "log_intercept_loc_prior"
    ]["scale"]
    int_sd_base = prior_dists["log_intercept_scale_prior"][
        "base_dist"
    ]
    int_sd_loc = int_sd_base.loc
    int_sd_scale = int_sd_base.scale

    tidy_results = ana.get_tidy_results(
        data_path,
        titer_mcmc_path,
        halflife_mcmc_path,
        include_pilot=True,
    )
    hls = tidy_results["halflives"]

    hls = ana.with_halflife_derived_quantities(hls)

    # filter out pilot and biphasic experiments
    hls = (
        hls.filter(pl.col("temperature_celsius") < 72)
        .with_columns(halflife_minutes=pl.col("halflife"))
        .with_columns(
            halflife_seconds=pl.col("halflife_minutes")
            * 60.0
        )
        .with_columns(
            time_to_lose_10_logs_10=10
            / pl.col("decay_rate")
        )
    )

    tab = (
        ana.median_qi_table(
            hls,
            [
                "halflife_minutes",
                "halflife_seconds",
                "time_to_lose_10_logs_10",
            ],
            [
                "virus",
                "temperature_celsius",
                "medium",
                "condition_id",
            ],
        )
        .sort(
            "virus",
            "medium",
            "temperature_celsius",
            "condition_id",
        )
        .with_columns(
            log_halflife_prior_exp_loc=pl.lit(
                hl_exp_loc
            ).cast(pl.Float32),
            log_halflife_prior_exp_scale=pl.lit(
                hl_exp_scale
            ).cast(pl.Float32),
            intercept_mode_prior_loc=pl.lit(
                int_mode_loc
            ).cast(pl.Float32),
            intercept_mode_prior_scale=pl.lit(
                int_mode_scale
            ).cast(pl.Float32),
            intercept_sd_prior_loc=pl.lit(int_sd_loc).cast(
                pl.Float32
            ),
            intercept_sd_prior_scale=pl.lit(
                int_sd_scale
            ).cast(pl.Float32),
        )
    )

    tab.write_csv(output_path, separator="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in MCMC output and produce a table of "
            "halflives reported in the manuscript"
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
