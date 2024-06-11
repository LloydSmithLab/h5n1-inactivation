#!/usr/bin/env python3

import argparse

import arviz as az

import analyze as ana


def main(
    mcmc_path: str,
    output_path: str,
    separator: str = "\t",
) -> None:
    """
    Create tab-separated tables of
    MCMC diagnostics

    Parameters
    ----------
    mcmc_path : str
        Path to the MCMC output, saved as a .pickle archive.

    output_path : str
        Path to save the output table.

    separator : str
        Delimiter for the delimited text
        file specified in output_path. Default
        `\t` (tab-delimited).
    """

    mcmc = ana.load_mcmc(mcmc_path)[0].mcmc_runner

    az_mcmc = az.from_numpyro(mcmc)

    tab = az.summary(
        az_mcmc, kind="diagnostics", round_to=5
    )
    tab.to_csv(output_path, sep=separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in MCMC output and produce a table of "
            "MCMC diagnostics using Arviz"
        )
    )

    parser.add_argument(
        "mcmc_path",
        type=str,
        help=(
            "Path to the MCMC output , "
            "saved as a .pickle archive."
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
        parsed["mcmc_path"],
        parsed["output_path"],
        separator=parsed["separator"],
    )
