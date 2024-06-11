#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from grizzlyplot.scales import ScaleXCategorical
from matplotlib.ticker import ScalarFormatter

import analyze as ana
import plotting as plot


def main(
    data_path: str,
    titer_mcmc_path: str,
    halflife_mcmc_path: str,
    output_path: str,
    separator: str = "\t",
    prior_annotate: bool = True,
) -> None:
    """
    Create the main text display figure,
    with one panel showing inferred halflives
    and another showing model fits to the raw data

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

    prior_annotate : bool
       Annotate the plot with key prior values?
       Boolean, default True.
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
    hls = tidy_results["halflives"]
    hls_int = tidy_results["halflives_with_intercepts"]
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
        titers,
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

    hls = hls.with_columns(
        halflife_seconds=pl.col("halflife") * 60.0
    ).filter(pl.col("temperature_celsius") < 70)
    # do not plot an estimated half-life for 72C since
    # raw data pattern does not follow a fixed-rate
    # exponential

    hl_plot = plot.halflife_violins(
        hls,
        x_column="condition_id",
        halflife_column="halflife_seconds",
        additional_mappings=dict(
            fillcolor="condition_id",
            markerfacecolor="condition_id",
        ),
        scales=dict(
            fillcolor=plot.condition_color_scale,
            markerfacecolor=plot.condition_color_scale,
            x=ScaleXCategorical(),
        ),
        facet=dict(
            col="temperature_celsius",
            sharex=False,
            label_cols=False,
        ),
        markeredgewidth=3,
    )

    fig, ax = plt.subplots(
        2, 2, figsize=[10, 8], sharex=None, sharey="row"
    )

    reg_plot.render(fig=fig, ax=ax[0, ::])
    hl_plot.render(fig=fig, ax=ax[1, ::])
    fig.supxlabel(None)
    fig.supylabel(None)

    title_63C = "63C"
    title_72C = "72C"

    ax[0, 0].set_title(title_63C)
    ax[0, 1].set_title(title_72C)
    ax[0, 0].set_ylim([1e-1, 1e8])
    ax[0, 0].set_xlim([-0.1, 5.2])
    ax[0, 1].set_xlim([-0.01, 0.52])
    ax[0, 0].set_xlabel(
        "Time (min since target temperature reached)", x=1
    )
    ax[0, 0].set_ylabel("Virus titer (TCID$_{50}$/mL)")

    ax[1, 0].yaxis.set_major_formatter(ScalarFormatter())
    ax[1, 0].set_ylabel("Half-life (sec)")
    ax[1, 0].set_ylim([0, 8])
    ax[1, 0].set_yticks(range(9))

    if prior_annotate:
        ax[1, 0].set_xlabel(
            plot.get_annotation_string(hl_model)
        )

    ax[1, 0].set_xticklabels("")
    ax[1, 1].set_xticklabels("")
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
    # do not annotate main text figure
    prior_annotate = "default" not in parsed["output_path"]
    main(
        parsed["data_path"],
        parsed["titer_mcmc_path"],
        parsed["halflife_mcmc_path"],
        parsed["output_path"],
        separator=parsed["separator"],
        prior_annotate=prior_annotate,
    )
