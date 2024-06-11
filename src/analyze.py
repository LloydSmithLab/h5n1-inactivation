import pickle

import numpy as np
import numpyro.distributions as dist
import polars as pl
from numpy.typing import ArrayLike
from pyter.infer import Inference
from pyter.models import AbstractModel


def spread_draws(
    posteriors: dict,
    variable_names: str | tuple | list[str | tuple],
) -> pl.DataFrame:
    """
    Given a dictionary of posteriors,
    return a long-form polars dataframe
    indexed by draw, with variable
    values (equivalent of tidybayes
    spread_draws() function).

    Parameters
    ----------
    posteriors : str
        A dictionary of posteriors
        with variable names as keys and
        numpy ndarrays as values (with
        the first axis corresponding
        to the posterior draw number,
        as the output of the get_samples()
        method of a numpyro MCMC object.

    variable_names: list[tuple[str]] | list[str]
        List of variables to retrieve. Array-valued
        variables should be specified as tuples of two or
        more strings, where the first string gives
        the variable name and subsequent ones give
        names for the array indices.

    Returns
    -------
    A tidy polars DataFrame with variable values associated
    to draw numbers and to variable array indices, where
    appropriate.
    """

    for i_var, v in enumerate(variable_names):
        if isinstance(v, str):
            v_dims = None
        else:
            v_dims = v[1:]
            v = v[0]

        post = posteriors.get(v)
        long_post = post.flatten()[..., np.newaxis]

        indices = np.array(list(np.ndindex(post.shape)))
        n_dims = indices.shape[1] - 1
        if v_dims is None:
            dim_names = [
                ("{}_dim_{}_index".format(v, k), pl.Int64)
                for k in range(n_dims)
            ]
        elif len(v_dims) != n_dims:
            raise ValueError(
                "incorrect number of "
                "dimension names "
                "provided for variable "
                "{}".format(v)
            )
        else:
            dim_names = [
                (v_dim, pl.Int64) for v_dim in v_dims
            ]

        p_df = pl.DataFrame(
            np.concatenate([indices, long_post], axis=1),
            schema=(
                [("draw", pl.Int64)]
                + dim_names
                + [(v, pl.Float64)]
            ),
        )

        if i_var == 0:
            df = p_df
        else:
            df = df.join(
                p_df,
                on=[
                    col
                    for col in df.columns
                    if col in p_df.columns
                ],
            )
        pass

    return df


def spread_and_recover_ids(
    posteriors: dict,
    variable_names: str | tuple | list[str | tuple],
    id_mappers: dict[ArrayLike] = None,
    id_datatype: str = "str",
    keep_internal: bool = False,
) -> pl.DataFrame:
    """
    Wraps the spread_draws function but automatically
    converts numerical internal ids (array indices)
    to categorical values, following a mapping. This can be
    useful for converting integer category ids back to human
    readable category values.

    Parameters
    ----------
    posteriors : dict
        See spread_draws()

    variable_names: list | tuple | str
        See spread_draws()

    id_mappers: dict
        Dictionary whose keys are the id dimension names
        and whose values are arrays whose k-th entries are
        the ID values we wish to associate with internal
        integer id k.

    id_datatype : str
        String specifying a numpy dtype to which
        the array of recovered id values will be cast
        Default "str" (cast to an array of strings).

    keep_internal : bool
       Retain the original internal ids? Default False.

    Returns
    -------
    A tidy polars dataframe of the same form as the output
    of spread_draws(), but with id values remapped via
    id_mappers.
    """

    temp_spread = spread_draws(posteriors, variable_names)

    if id_mappers is None:
        id_mappers = {}

    new_cols = []

    for dim_name, mapper in id_mappers.items():
        if dim_name in temp_spread.columns:
            map_vals = temp_spread.get_column(
                dim_name
            ).to_numpy()
            new_cols.append(
                pl.lit(
                    mapper[map_vals].astype(id_datatype)
                ).alias(dim_name)
            )

            if keep_internal:
                new_cols.append(
                    temp_spread.get_column(dim_name).alias(
                        dim_name + "_internal"
                    )
                )
    return temp_spread.with_columns(new_cols)


def load_mcmc(path: str) -> Inference:
    """
    Load a pickled MCMC chains object
    and return it.

    Parameters
    ----------
    path : str
       Path to pickled object to load.

    Returns
    -------
    The loaded object
    """
    with open(path, "rb") as file:
        infer = pickle.load(file)

    return infer


def get_sample_index(
    data: pl.DataFrame, variable_lods: bool = False
) -> pl.DataFrame:
    """
    Get a polars DataFrame of titer metadata
    from the overall long-form cleaned data.

    This is useful for joining to MCMC output.

    Parameters
    ----------
    data : pl.DataFrame
        Long-form tidy data, usually with more
        than one row (observation) per sample.

    variable_lods : boolean
        Are titer limits of detection expected to
        vary in this experiment? If not, will raise
        a value error if the data imply multiple distinct
        LODs.

    Returns
    -------
    A polars DataFrame of sample metadata with
    one row per sample.
    """
    sample_metadata = data.unique("sample_id").select(
        "sample_id",
        "condition_id",
        "timepoint_minutes",
        "medium",
        "temperature_celsius",
        "is_pilot",
    )

    sample_index = (
        data.group_by("sample_id")
        .agg(
            total_positive_wells=pl.col(
                "well_status"
            ).sum(),
            least_log10_dilution=pl.col(
                "log10_dilution"
            ).max(),
            lod_well_volume_ml=pl.col("well_volume_ml")
            .filter(
                pl.col("log10_dilution")
                == pl.col("log10_dilution").max()
            )
            .max()
            # lod is determined by the volume of the
            # largest well at the least extreme dilution
        )
        .with_columns(
            detected=pl.col("total_positive_wells") > 0,
            log10_approx_lod=(
                -0.5
                - pl.col("least_log10_dilution")
                - pl.col("lod_well_volume_ml").log10()
            )
            # convert the Spearman-Karber
            # LOD of (-0.5 - least_log10_dilution)
            # log10 TCID/(well volume) to units of
            # mL using the mL volume of the largest
            # well used at the least dilution
        )
        .join(sample_metadata, on="sample_id")
    )

    if not (
        variable_lods
        or sample_index["log10_approx_lod"].unique().len()
        == 1
    ):
        raise ValueError(
            "Got more than one log10 approximate LOD; "
            "this is not expected in this experiment"
        )

    return sample_index


def get_condition_index(
    data: pl.DataFrame,
) -> pl.DataFrame:
    """
    Get a polars DataFrame of experimental condition metadata
    from the overall long-form cleaned data DataFrame.

    This is useful for joining to MCMC output.

    Parameters
    ----------
    data : pl.DataFrame
        Long-form tidy data, usually with more
        than one row (observation) per experimental condition.

    Returns
    -------
    A polars DataFrame of experimental condition metadata with
    one row per experimental condition.
    """

    condition_index = data.unique(
        subset="condition_id"
    ).select(
        "condition_id",
        "virus",
        "medium",
        "temperature_celsius",
    )

    return condition_index


def spread_titers(
    inference_object: Inference, samples: dict = None
) -> pl.DataFrame:
    """
    Convenience method for calling spread_and_recover_ids()
    on the output of TiterModel inference.

    Parameters
    ----------
    inference_object : Inference
        An inference object with samples for
        a parameter named "log_titer".

    samples: dict
        Dictionary of samples to spread. If None,
        use the output of inference_object.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    A tidy polars dataframe of log_titer samples, obtained
    as the output of spread_and_recover_ids()
    """
    if samples is None:
        samples = (
            inference_object.mcmc_runner.get_samples()
        )
    return spread_and_recover_ids(
        samples,
        [("log_titer", "sample_id")],
        {
            "sample_id": inference_object.run_data[
                "unique_external_ids"
            ]["titer"]
        },
        keep_internal=False,
    )


def spread_halflives(
    inference_object: Inference, samples: dict = None
) -> pl.DataFrame:
    """
    Convenience method for calling spread_and_recover_ids
    on the output of HalfLifeModel inference.

    Parameters
    ----------
    inference_object : Inference
        An pyter.Inference object with samples for
        a parameter named "log_halflife".

    samples: dict
        Dictionary of samples to spread. If None,
        use the output of inference_object.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    A tidy polars dataframe of log_titer samples, obtained
    as the output of spread_and_recover_ids()
    """
    if samples is None:
        samples = (
            inference_object.mcmc_runner.get_samples()
        )

    return spread_and_recover_ids(
        samples,
        [("log_halflife", "condition_id")],
        id_mappers={
            "condition_id": inference_object.run_data[
                "unique_external_ids"
            ]["halflife"]
        },
    )


def spread_halflives_with_intercepts(
    inference_object: Inference, samples: dict = None
) -> pl.DataFrame:
    """
    Convenience method for calling spread_and_recover_ids
    on the output of HalfLifeModel inference, including
    draws for the inferred intercepts for the individual titers.

    Parameters
    ---------
    inference_object: Inference
        pyter.Inference object to query for id mappers
        for the halflives and intercepts

    samples: dict
        Dictionary of samples to spread. If None,
        use the output of inference_object.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    A tidy polars dataframe of halflife samples, and
    associated titer intercept samples, obtained
    as the output of spread_and_recover_ids()
    """
    if samples is None:
        samples = (
            inference_object.mcmc_runner.get_samples()
        )

    return spread_and_recover_ids(
        samples,
        [
            ("log_halflife", "condition_id"),
            ("log_titer_intercept", "sample_id"),
        ],
        id_mappers={
            "condition_id": inference_object.run_data[
                "unique_external_ids"
            ]["halflife"],
            "sample_id": inference_object.run_data[
                "unique_external_ids"
            ]["titer"],
        },
    )


def with_halflife_derived_quantities(
    halflife_df: pl.DataFrame,
):
    """
    Augment a polars dataframe with
    columns derived from the log halflife

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to augment

    Returns
    -------
    The augmented DataFrame
    """

    df = (
        halflife_df.with_columns(
            halflife=pl.col("log_halflife").exp()
        )
        .with_columns(
            decay_rate=(
                pl.lit(np.log10(2)) / pl.col("halflife")
            )
        )
        .with_columns(
            exp_rate=(-1.0 * pl.col("decay_rate"))
        )
    )

    return df


def downsample_draws(
    df: pl.DataFrame,
    n_draws_to_sample: int,
    id_column: str | None = None,
    draw_column: str = "draw",
) -> pl.DataFrame:
    """
    Downsample a tidy dataframe to only a
    certain set number of unique draws for
    each unique value of a given id column

    Parameter
    ---------
    df : pl.DataFrame
        Tidy DataFrame to downsample.

    n_draws_to_sample: int
        Number of random draws to sample (
        in total, or per unique ID column
        value if an id_column is specified).

    id_column : str | None
       ID column. If specified, choose
       n_draws_to_sample independently for each
       unique value of the ID column.

    draw_column : str
       Name of the column identifying individual
       draws. Default 'draw'
    """
    if id_column is not None:
        to_sample = (
            df.unique(id_column)
            .select(
                pl.col(id_column).repeat_by(
                    n_draws_to_sample
                )
            )
            .explode(id_column)
        )
        join_cols = [draw_column, id_column]
    else:
        to_sample = pl.DataFrame()
        join_cols = [draw_column]

    sampled_draws = to_sample.with_columns(
        pl.lit(
            np.random.randint(
                df.select("draw").min(),
                df.select("draw").max(),
                size=to_sample.shape[0],
            )
        ).alias(draw_column)
    )

    return df.join(sampled_draws, on=join_cols)


def get_tidy_titers(
    titer_infer: Inference,
    data: pl.DataFrame,
    samples: dict = None,
) -> pl.DataFrame:
    """
    Convenience function to wrap spread_titer_draws()
    but also call get_sample_index()
    and join to the results.

    Parameters
    ----------
    titer_infer : Inference
        MCMC results for titer inference, passed
        to spread_titer_draws()

    data : pl.DataFrame
        Input data as a polars DataFrame,
        passed to get_sample_index()

    samples: dict
        Dictionary of MCMC samples to spread. If None,
        use the output of titer_infer.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    The joined tidy results, as a polars DataFrame.
    """
    sample_index = get_sample_index(data)
    return spread_titers(
        titer_infer, samples=samples
    ).join(sample_index, on="sample_id")


def get_tidy_hls(
    hl_infer: Inference,
    data: pl.DataFrame,
    samples: dict = None,
) -> pl.DataFrame:
    """
    Convenience function to wrap spread_halflives()
    but also call get_condition_index(), join it to the
    results, and apply with_halflife_derived_quantities()
    to the resultant dataframe.

    Parameters
    ----------
    hl_infer : Inference
        MCMC results for halflife inference, passed
        to spread_halflives()

    data : pl.DataFrame
        Input data as a polars DataFrame,
        passed to get_condition_index()

    samples: dict
        Dictionary of MCMC samples to spread. If None,
        use the output of hl_infer.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    The resulting tidy results, as a polars DataFrame.
    """
    condition_index = get_condition_index(data)

    tidy_hls = (
        spread_halflives(
            hl_infer,
            samples=samples,
        )
        .join(condition_index, on="condition_id")
        .pipe(with_halflife_derived_quantities)
    )

    return tidy_hls


def get_tidy_hls_with_intercepts(
    hl_infer: Inference,
    data: pl.DataFrame,
    samples: dict = None,
):
    """
    Convenience function to wrap spread_halflives_with_intercepts()
    but also call get_sample_index(), join it to the
    results,and apply with_halflife_derived_quantities()
    to the resultant dataframe.

    Parameters
    ----------
    hl_infer : Inference
        MCMC results for halflife inference, passed
        to spread_halflives_with_intercepts()

    data : pl.DataFrame
        Input data as a polars DataFrame,
        passed to get_sample_index().

    samples: dict
        Dictionary of MCMC samples to spread. If None,
        use the output of hl_infer.mcmc_runner.get_samples().
        Default None.

    Returns
    -------
    The resulting tidy results, as a polars DataFrame.
    """

    sample_index = get_sample_index(data)

    tidy_hls_with_intercepts = (
        spread_halflives_with_intercepts(
            hl_infer, samples=samples
        )
        .join(
            sample_index, on=["sample_id", "condition_id"]
        )
        .pipe(with_halflife_derived_quantities)
    )

    return tidy_hls_with_intercepts


def get_tidy_results(
    data_path: str,
    titer_infer_path: str,
    hl_infer_path: str,
    include_pilot: bool = False,
    separator: str = "\t",
) -> dict[pl.DataFrame]:
    """
    Get a dictionary of data and tidy MCMC results
    for dowstream analysis, including the results of
    get_tidy_titers(), get_tidy_hls(), and
    get_tidy_hls_with_intercepts()

    Parameters
    ----------
    data_path : str
        Path to the input data, as a delimited
        text file.

    titer_infer_path : str
        Path to pickled MCMC results for raw titers.

    hl_infer_path : str
        Path to pickled MCMC results for halflives.

    include_pilot : bool
        Include pilot experiments? Boolean, default False.

    separator: str
        Delimiter for the input data text file.

    Returns
    -------
    A dictionary with the results of calling get_tidy_titers(),
    get_tidy_hls(), and get_tidy_hls_with_intercepts() on the
    data and MCMC results {parameter inference results, prior checks,
    posterior checks}, plus the data itself
    """

    data = pl.read_csv(data_path, separator=separator)

    if not include_pilot:
        data = data.filter(~pl.col("is_pilot"))

    (
        titer_infer,
        titer_prior_check,
        titer_post_check,
    ) = load_mcmc(titer_infer_path)
    titer_mapping = {
        "titers": titer_infer.mcmc_runner.get_samples(),
        "titers_prior_check": titer_prior_check,
        "titer_posterior_check": titer_post_check,
    }

    tidy_titer_dict = {
        key: get_tidy_titers(
            titer_infer, data, samples=val
        )
        for key, val in titer_mapping.items()
    }

    hl_infer, hl_prior_check, hl_post_check = load_mcmc(
        hl_infer_path
    )
    hl_mapping = {
        "": hl_infer.mcmc_runner.get_samples(),
        "_prior_check": hl_prior_check,
        "_posterior_check": hl_post_check,
    }

    tidy_hl_dict = {
        "halflives"
        + key: get_tidy_hls(hl_infer, data, samples=val)
        for key, val in hl_mapping.items()
    }

    tidy_hl_int_dict = {
        "halflives_with_intercepts"
        + key: get_tidy_hls_with_intercepts(
            hl_infer, data, samples=val
        )
        for key, val in hl_mapping.items()
    }

    result = {
        "data": data,
        **tidy_titer_dict,
        **tidy_hl_dict,
        **tidy_hl_int_dict,
    }

    return result


def expression_format_point_interval(
    point_estimate_column: str,
    left_endpoint_column: str,
    right_endpoint_column: str,
    format_string: str = "{point:.2f} [{left:.2f}, {right:.2f}]",
) -> pl.Expr:
    """
    Get a Polars expression formatting
    posterior estimates in the form
    "point_estimate [interval_left,
    interval_right]" for use in
    written results sections.

    Parameters
    ----------
    point_estimate_column: str
        Name of the column containing the point estimate(s)
    left_endpoint_column: str
        Name of the column containing the left interval endpoint(s)
    right_endpoint_column: str
        Name of the column containing the right interval endpoint(s)
    format_string: str
        Format string to format with the point estimate and endpoints,
        when these are passed as a dict of the form
            ```{"point": x_1, "left": x_2, "right": x_3}```
        Default: "{point:.2f} [{left:.2f}, {right:.2f}]",
        which would yield something like "2.52 [1.88, 3.51]"

    Returns
    -------
    format_expr: pl.Expr
        A polars expression that will yield
        appropriately formatted strings when
        evaluated.
    """
    return pl.struct(
        [
            pl.col(point_estimate_column).alias("point"),
            pl.col(left_endpoint_column).alias("left"),
            pl.col(right_endpoint_column).alias("right"),
        ]
    ).apply(
        lambda x: format_string.format(**x),
        return_dtype=pl.Utf8,
    )


def median_qi_table(
    df: pl.DataFrame,
    columns: list,
    group_columns: list = None,
    rename: dict = None,
) -> pl.DataFrame:
    """
    Given a tidy polars DataFrame, generate
    a table of formatted medians
    and quantile intervals for the given
    columns, optionally grouped.

    Parameters
    ----------

    df : pl.DataFrame
        The dataframe on which to perform the summary operation
    columns : list
        The columns to get estimates for
    group_columns: list, optional
        Columns to group by, if any
    rename : dict, optional
        Optional dictionary to rename
        the estimate columns in the output
        table, e.g. {"col1": "newname"}
        would lead estimates based on column
        "col1" to be called "foo_median"
        "newname_q025", "newname_q795", and "
        "newname_formatted" in the output table.

    Returns
    -------
    A polars data frame with the summary estimates
    """
    if group_columns is None:
        df = df.with_columns(pl.lit(1).alias("group_id"))
        group_columns = ["group_id"]

    if rename is None:
        rename = {}

    tab = df.groupby(group_columns)
    estimates = ["median", "q025", "q975", "formatted"]

    summary_table = (
        tab.agg(
            [
                col
                for x in columns
                for col in [
                    pl.col(x)
                    .median()
                    .alias(x + "_median"),
                    pl.col(x)
                    .quantile(0.025)
                    .alias(x + "_q025"),
                    pl.col(x)
                    .quantile(0.975)
                    .alias(x + "_q975"),
                ]
            ]
        )
        .with_columns(
            [
                expression_format_point_interval(
                    x + "_median", x + "_q025", x + "_q975"
                ).alias(x + "_formatted")
                for x in columns
            ]
        )
        .select(
            group_columns
            + [
                col
                for x in columns
                for col in [
                    pl.col(x + "_" + est).alias(
                        rename.get(x, x) + "_" + est
                    )
                    for est in estimates
                ]
            ]
        )
        .sort(group_columns)
    )

    return summary_table


def extract_distribution_params(model: AbstractModel):
    """
    Extract distribution names and parameters
    for distributions instantiated in
    a pyter Model.

    Parameters
    ----------
    model : pyter.AbstractModel
        Model from which to extract distributions

    Returns
    -------
    dict
       Dict of dicts, indexed by the distributions'
       names within the Pyter model. Each dict contains
       an entry for the distribution's class as well
       as all other class attributes (the outputs of
       a call to the distribution's __getstate__()
       method)
    """
    model_state = {
        k: v
        for k, v in model.__getstate__().items()
        if isinstance(v, dist.Distribution)
    }
    return {
        key: {
            "class": value.__class__,
            **value.__getstate__(),
        }
        for key, value in model_state.items()
    }
