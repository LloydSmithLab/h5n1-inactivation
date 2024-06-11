#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import analyze as ana
import plotting as plot


def main(
    data_path: str,
    titer_mcmc_path: str,
    halflife_mcmc_path: str,
    output_path: str,
    separator: str = "\t",
) -> None:
    """
    Create a prior predictive check display figure,
    that mirrors the primary main text figure.
    with one panel showing inferred halflives
    and another showing model fits to data, but

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
        Path to which to save the figure.

    separator : str
        Delimiter for the delimited text
        file specified in data_path. Default
        `\t` (tab-delimited).
    """
    print(
        f"Creating figure {os.path.basename(output_path)}..."
    )

    hl_model = ana.load_mcmc(halflife_mcmc_path)[
        0
    ].run_model

    tidy_results = ana.get_tidy_results(
        data_path,
        titer_mcmc_path,
        halflife_mcmc_path,
        include_pilot=False,
    )
    titers = tidy_results["titers"]
    hls_int = tidy_results[
        "halflives_with_intercepts_prior_check"
    ]
    titers = titers.with_columns(
        display_titer=pl.when(pl.col("detected"))
        .then(10 ** pl.col("log_titer"))
        .otherwise(10 ** pl.col("log10_approx_lod"))
    )

    hls_reg = ana.downsample_draws(
        hls_int, 10, id_column="sample_id"
    ).with_columns(
        initial_titer=10 ** pl.col("log_titer_intercept")
    )

    reg_plot = plot.titer_regression(
        titers.filter(pl.col("temperature_celsius") < 70),
        hls_reg.filter(pl.col("temperature_celsius") < 70),
        facet={
            "col": "temperature_celsius",
            "sharex": False,
            "label_cols": False,
        },
    )
    # do not plot exp decay lines for 72C as
    # raw data pattern does not follow a fixed-rate
    # exponential

    fig, ax = plt.subplots(1, 1, figsize=[5, 4])

    reg_plot.render(fig=fig, ax=ax)
    fig.supxlabel(None)
    fig.supylabel(None)
    ax.set_title("63C")
    ax.set_ylim([1e-1, 1e8])
    ax.set_xlim([-0.1, 5.2])
    ax.set_xlabel(
        "Time (min since target temperature reached)"
        + "\n\n"
        + plot.get_annotation_string(hl_model)
    )
    ax.set_ylabel("Virus titer (TCID$_{50}$/mL)")

    print(f"Saving figure to {output_path}...")
    fig.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in MCMC output and produce a main "
            "text figure showing fits to the data "
            "and inferred halflives"
        )
    )
    parser.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to the data used for fitting, formatted as "
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
        help=("Path to save the generated figure."),
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
    # set seed for reproducibility
    # (since we use random draws)
    np.random.seed(52367)
    main(
        parsed["data_path"],
        parsed["titer_mcmc_path"],
        parsed["halflife_mcmc_path"],
        parsed["output_path"],
        separator=parsed["separator"],
    )
