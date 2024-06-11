#!/usr/bin/env python3

"""
Fit titer inference and half-life
inference models to titratration
well data.
"""

import argparse
import pickle

import jax
import numpy as np
import numpyro
import polars as pl
import toml
from numpyro.infer import Predictive
from pyter.infer import Inference

from config import get_model_parameter
from model_factory import model_factory


def main(
    data_path: str,
    mcmc_config_path: str,
    prior_param_path: str,
    model_name: str,
    output_path: str = None,
    separator="\t",
    strict: bool = True,
):
    """
    Perform inference from a dataset,
    using Pyter models and a No-U-Turn sampler,
    saving the result to disk as a Python .pickle
    file. Also performs prior and posterior
    predictive checks.

    Parameters
    ----------
    data_path : str
        Path to the data file to fit to. Data
        should be in tidy tabular format in a
        delimited text file (default .tsv, see
        separator)

    mcmc_config_path : str
        Path to a TOML-formatted configuration
        file specifying parameters for the MCMC.

    prior_param_path : str
        Path to a TOML-formatted configuration
        file specifying hyperparameter values
        for prior distributions.

    model_name : str
        Name of the model to fit. One of 'individual_titer'
        and 'halflife'.

    output_path : str
        Path to save the output.
        If None, a save path will be constructed
        from the model_name: '{model_name}.pickle'.
        Default None.

    separator : str
        Separator for the delimited data
        text file. Default '\t' (tab / .tsv
        format)

    strict : bool
        Raise an error if there are divergent transitions
        after warmup? Default True.

    Return
    ------
    None, saving the result to disk as a side effect

    Raises
    ------
    An error there are divergent transitions after
    warmup and strict is set to true.
    """
    data = pl.read_csv(data_path, separator=separator)
    mcmc_config = toml.load(mcmc_config_path)
    prior_params = toml.load(prior_param_path)
    seed = get_model_parameter(
        mcmc_config, model_name, "seed"
    )
    n_chains = get_model_parameter(
        mcmc_config, model_name, "n_chains"
    )
    n_prior_pred_samples = get_model_parameter(
        mcmc_config, model_name, "n_prior_predictive"
    )
    n_cores = get_model_parameter(
        mcmc_config, model_name, "n_cores"
    )
    if n_cores is None:
        n_cores = 1
    numpyro.set_host_device_count(n_cores)

    m_data, model = model_factory(
        model_name, data, prior_params
    )

    infer = Inference(
        target_accept_prob=get_model_parameter(
            mcmc_config, model_name, "target_accept_prob"
        ),
        max_tree_depth=get_model_parameter(
            mcmc_config, model_name, "max_tree_depth"
        ),
    )
    if n_cores < n_chains:
        chain_method = "sequential"
    else:
        chain_method = "parallel"

    infer.infer(
        data=m_data,
        model=model,
        random_seed=seed,
        num_chains=n_chains,
        chain_method=chain_method,
    )
    infer.mcmc_runner.print_summary()

    if strict:
        print("Checking for MCMC convergence problems...")
        if np.any(
            infer.mcmc_runner.get_extra_fields()[
                "diverging"
            ]
        ):
            raise ValueError(
                "At least one divergent transition after "
                "warmup. Exiting without saving results "
                "because `strict` was set to `True`. "
                "If you want to save results anyway for "
                "diagnosis, set `strict = False`"
            )
        else:
            print("No divergent transitions.\n")

    print("Performing predictive checks...")
    prior_predictive = Predictive(
        model.model, num_samples=n_prior_pred_samples
    )

    prior_preds = prior_predictive(
        rng_key=jax.random.PRNGKey(seed + 1),
        data=infer.run_data,
    )

    posterior_predictive = Predictive(
        model.model,
        posterior_samples=infer.mcmc_runner.get_samples(),
        return_sites=[
            "log_titer",
            "log_titer_intercept",
            "well_status",
            "log_halflife",
        ],
    )

    posterior_preds = posterior_predictive(
        rng_key=jax.random.PRNGKey(seed + 2),
        data=infer.run_data,
    )

    output = (infer, prior_preds, posterior_preds)

    if output_path is None:
        output_path = f"{model_name}.pickle"

    print(f"Saving output to {output_path}...")
    with open(output_path, "wb") as file:
        pickle.dump(output, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read in cleaned data as a tsv or other delimited, "
            "file, fit to it, and save the result as a .pickle "
            "archive."
        )
    )
    parser.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to the data to fit, formatted as "
            "a delimited text file"
        ),
    )
    parser.add_argument(
        "mcmc_config_path",
        type=str,
        help=(
            "Path to a TOML-formatted configuration file "
            "specifying configuration for the mcmc."
        ),
    )
    parser.add_argument(
        "prior_config_path",
        type=str,
        help=(
            "Path to a TOML-formatted configuration file "
            "specifying hyperparameter values for prior "
            "distributions."
        ),
    )

    parser.add_argument(
        "model_name",
        type=str,
        help=(
            "Name of the model to fit. One of 'individual_titer' "
            "and 'halflife'"
        ),
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help=(
            "Path to save the fit model object. "
            "If not specified, it will be constructed "
            "from the model_name"
        ),
        default=None,
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
        parsed["mcmc_config_path"],
        parsed["prior_config_path"],
        parsed["model_name"],
        output_path=parsed["output_path"],
        separator=parsed["separator"],
        strict=True,
    )
