"""
Helper functions for working with
configuration files
"""


def get_model_parameter(
    config_dict: dict,
    model_name,
    parameter_name,
    strict=True,
):
    """
    Given a parameter dictionary, model name,
    and parameter name, retrieve a model-specific
    parameter value if one exists, and otherwise
    fall back on the default value.

    Parameters
    ----------
    config_dict : dict
       Configuration dictionary to use.

    model_name : str
       Name of the model.

    parameter_name : str
       Name of the parameter value
       to retrieve

    strict : bool
       Error if a result cannot be found (True), or return
       None (False). Default True.

    Return
    ------
    The parameter value, preferring a model-specific
    value if one is found.
    """
    model_dict = config_dict.get(model_name, {})
    default_dict = config_dict.get("default", {})

    result = model_dict.get(
        parameter_name,
        default_dict.get(parameter_name, None),
    )

    if strict and result is None:
        raise ValueError(
            f"No value found for parameter_name '{parameter_name}' "
            f"for model_name '{model_name}'."
        )
    return result
