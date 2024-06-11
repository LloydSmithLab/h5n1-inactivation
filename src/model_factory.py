import numpy as np
import numpyro.distributions as dist
import polars as pl
from pyter.data import (
    AbstractData,
    HalfLifeData,
    TiterData,
)
from pyter.models import (
    AbstractModel,
    HalfLifeModel,
    TiterModel,
)


def model_factory(
    model_name: str,
    data: pl.DataFrame,
    prior_params: dict,
) -> tuple[AbstractData, AbstractModel]:
    """
    Instantiate an appropriate pyter Model and Data pair
    given the requested output.

    Parameters
    ----------
    model_fit_name : str
        String that uniquely identifies a fitting
        problem

    data : pl.DataFrame
        Tidy polars DataFrame of data to fit to.
        Its entries will populate the pyter Data
        object.

    prior_params : dict
        dictionary of hyperparameter values for
        model prior distributions.

    Returns
    -------
    A tuple (data, model) of the instantiated pyter Data object
    and the instantiated pyter Model object.
    """

    if model_name == "individual_titer":
        m_data = TiterData(
            well_status=data["well_status"].to_numpy(),
            well_dilution=data[
                "log10_dilution"
            ].to_numpy(),
            well_titer_id=data["sample_id"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=0,
            log_base=10,
        )
        titer_params = prior_params[
            "individually_inferred_log10_titers"
        ]
        model = TiterModel(
            log_titer_prior=dist.Normal(
                loc=titer_params["loc"],
                scale=titer_params["scale"],
            ),
            assay="tcid",
        )
    elif model_name == "halflife":
        # only infer half-life for certain conditions
        halflife_conditions = [
            "H5N1_mountain_lion_isolate-milk-63.0C"
        ]

        data = data.filter(
            pl.col("condition_id").is_in(
                halflife_conditions
            )
        )

        m_data = HalfLifeData(
            well_status=data["well_status"].to_numpy(),
            well_dilution=data[
                "log10_dilution"
            ].to_numpy(),
            well_titer_id=data["sample_id"].to_numpy(),
            well_titer_error_scale_id=data[
                "condition_id"
            ].to_numpy(),
            well_halflife_id=data[
                "condition_id"
            ].to_numpy(),
            well_intercept_id=data["sample_id"].to_numpy(),
            well_intercept_loc_id=data[
                "condition_id"
            ].to_numpy(),
            well_intercept_scale_id=data[
                "condition_id"
            ].to_numpy(),
            well_time=data["timepoint_minutes"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=0,
            log_base=10,
        )

        hl = prior_params["log_halflife_minutes"]
        t0_mode = prior_params["t0_log10_titer_mode"]
        t0_sd = prior_params["t0_log10_titer_sd"]

        model = HalfLifeModel(
            log_halflife_distribution=dist.Normal(
                loc=np.log(hl["exp_loc"]),
                scale=np.log(hl["exp_scale"]),
            ),
            log_intercept_distribution=dist.Normal,
            log_intercept_loc_prior=dist.Normal(
                loc=t0_mode["loc"], scale=t0_mode["scale"]
            ),
            log_intercept_scale_prior=dist.TruncatedNormal(
                low=0.0,
                loc=t0_sd["loc"],
                scale=t0_sd["scale"],
            ),
            assay="tcid",
            intercepts_hier=True,
            halflives_hier=False,
            titers_overdispersed=False,
        )
    else:
        raise ValueError("Unknown model to fit")

    return m_data, model
