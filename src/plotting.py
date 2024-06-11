#!/usr/bin/env python3

"""
Style and helper functions for plotting
"""

import string

import grizzlyplot as gp
import grizzlyplot.geoms as geoms
import grizzlyplot.scales as scales
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec
from numpy.typing import ArrayLike
from pyter.models import HalfLifeModel

import analyze

# custom project matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.size"] = 15
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["axes.formatter.limits"] = (-3, 3)
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.bottom"] = False
mpl.rcParams["legend.fancybox"] = True
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.framealpha"] = 1
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["legend.fontsize"] = "small"
mpl.rcParams["legend.title_fontsize"] = "small"
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.2


# orderings, defaults, and custom scales for Grizzlyplot

default_xspan = [0, 20]
default_xlen = default_xspan[1] - default_xspan[0]
default_xmargin = 0.15
default_xdiff = default_xlen * default_xmargin
default_xlim = [
    default_xspan[0] - default_xdiff,
    default_xspan[1] + default_xdiff,
]
default_xticks = [0, 5, 10, 15, 20]
default_ylim = [0.1, 10**7]
default_yticks = [1, 10**2, 10**4, 10**6]
default_ylab_pad = 20

marker_scale = scales.ScaleDiscreteManual(
    mapping={True: "o", False: "v"}
)
condition_color_scale = scales.ScaleColorManual(
    mapping={
        "H5N1_mountain_lion_isolate-milk-63.0C": "orange",
        "H5N1_mountain_lion_isolate-milk-72.0C": "red",
        "k": "k",
    }
)


def get_figsize(
    aspect: float = 1.6, width: float = 6.5
) -> list[float, float]:
    """
    Return a list of the form
    [width, height] with the requested
    aspect ratio and width in inches.

    If aspect and/or width are not
    specified, use project defaults.

    Parameters
    ----------
    aspect: float
        Aspect ratio for the figure. Default 1.6.

    width: float
        Width of the figure in inches.
        Default 8.5 (width of a US letter page
        with 1 in margins on each side).

    Returns
    -------

    figsize: list[float, float]
       A list of the form [width, height] that
       can be passed as a figsize parameter to
       :func:`matplotlib.pyplot.figure()`` and
       similar functions.
    """
    return [width, width / aspect]


marker_scale = scales.ScaleDiscreteManual(
    mapping={True: "o", False: "v"}
)


# Helper functions to avoid code repetition
# and maintain standard look and feel across
# plots


def ordered_subfigures(
    figure: mpl.figure.Figure,
    nrows: int = 1,
    ncols: int = 1,
    order: str | ArrayLike = "lrtb",
    squeeze: bool = True,
    wspace: float = None,
    hspace: float = None,
    width_ratios: ArrayLike = None,
    height_ratios: ArrayLike = None,
    **kwargs
):
    """
    Add a set of subfigures to this figure or subfigure, with order

    This is a simple modification of the built in matplotlib
    Figure.subfigures() method designed as a workaround for the
    fact that figures created that way necessarily draw upper
    rows first and lower rows later, which can be a problem
    if subfigures are to overlap with upper figure on top.

    A subfigure has the same artist methods as a figure,
    and is logically the same as a figure,
    but cannot print itself.
    See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        The figure to which to add the ordered subfigures.

    nrows, ncols : int, default: 1
        Number of rows/columns of the subfigure grid.

    order: str | array_like, default "lrtb"
        Ordering for the rows and columns. Default
        is as in Figure.subfigures(): left to right and then
        top to bottom.

    squeeze : bool, default: True
        If True, extra dimensions are squeezed out from the returned
        array of subfigures.

    wspace, hspace : float, default: None
        The amount of width/height reserved for space between subfigures,
        expressed as a fraction of the average subfigure width/height.
        If not given, the values will be inferred from a figure or
        rcParams when necessary.

    width_ratios : array-like of length *ncols*, optional
        Defines the relative widths of the columns. Each column gets a
        relative width of ``width_ratios[i] / sum(width_ratios)``.
        If not given, all columns will have the same width.

    height_ratios : array-like of length *nrows*, optional
        Defines the relative heights of the rows. Each row gets a
        relative height of ``height_ratios[i] / sum(height_ratios)``.
        If not given, all rows will have the same height.

    **kwargs:
        Other keyword arguments passed to
        :meth:`matplotlib.Figure.figure.add_subfigure()`.

    Returns
    ----------
    Array of subfigures

    """
    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=figure,
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )

    if isinstance(order, (list, np.ndarray, tuple)):
        order_list = order
    elif not isinstance(order, str):
        raise ValueError("Order must be list/array or str")
    else:
        if all([x in order for x in ["l", "r", "t", "b"]]):
            if order.index("l") < order.index("r"):
                colrange = range(ncols)
            else:
                colrange = reversed(range(ncols))
            if order.index("t") < order.index("b"):
                rowrange = range(nrows)
            else:
                rowrange = reversed(range(nrows))
            if min(
                order.index("l"), order.index("r")
            ) < min(order.index("t"), order.index("b")):
                order_list = [
                    [j, i]
                    for j in rowrange
                    for i in colrange
                ]
            else:
                order_list = [
                    [j, i]
                    for i in colrange
                    for j in rowrange
                ]
        else:
            raise ValueError("Invalid order string")
    sfarr = np.empty((nrows, ncols), dtype=object)
    for ind in range(nrows * ncols):
        j, i = order_list[ind]
        sfarr[j, i] = figure.add_subfigure(
            gs[j, i], **kwargs
        )

    if squeeze:
        # Discarding unneeded dimensions that equal 1.
        # If we only have one
        # subfigure, just return
        # it instead of a 1-element array.
        return (
            sfarr.item()
            if sfarr.size == 1
            else sfarr.squeeze()
        )
    else:
        # Returned axis array will be
        # always 2-d, even if nrows=ncols=1.
        return sfarr


def label_subfigures(
    figure,
    labels=string.ascii_uppercase,
    x=0,
    y=0.95,
    style="oblique",
    fontsize="xx-large",
    fontweight="bold",
    ha="center",
    va="center",
    **kwargs
):
    """
    Convenience function to autolabel
    subfigures within a figure with
    nicely formatted text, including
    some standard defaults
    """
    for i, subfig in enumerate(figure.subfigs):
        subfig.text(
            x=x,
            y=y,
            s=labels[i],
            style=style,
            fontsize="xx-large",
            fontweight=fontweight,
            **kwargs
        )
    return figure


def titer_regression(
    titers: pl.DataFrame,
    regression_lines: pl.DataFrame,
    facet: dict = None,
    xmin=0,
    xmax=20,
    linealpha=0.25,
    **kwargs
):
    """
    GrizzlyPlot specification for basic
    titer regression, single phase.
    """
    return gp.GrizzlyPlot(
        data=titers,
        mapping=dict(
            y="titer",
            x="timepoint_minutes",
            group="sample_id",
        ),
        facet=facet,
        geoms=[
            geoms.GeomAxHLines(
                yintercept=(
                    10 ** titers["log10_approx_lod"][0]
                ),
                lw=2,
                ls="dashed",
                color="k",
                name="LOD line",
            ),
            geoms.GeomExponentialX(
                data=regression_lines,
                base=10,
                xmin=xmin,
                xmax=xmax,
                lw=0.5,
                alpha=linealpha,
                mapping=dict(
                    yintercept="initial_titer",
                    rate="exp_rate",
                    group="sample_id",
                    color="condition_id",
                ),
                name="Predicted decay",
            ),
            geoms.GeomPointInterval(
                mapping=dict(
                    y="display_titer", marker="detected"
                ),
                name="Titer posterior estimates",
                markersize=10,
                markerfacecolor="#abb0ae",
                markeredgewidth=1,
                lw=3,
                alpha=0.7,
                color="k",
            ),
        ],
        scales=dict(
            y=scales.ScaleY("log"),
            color=condition_color_scale,
            marker=marker_scale,
        ),
        alpha=0.75,
        **kwargs
    )


def halflife_violins(
    halflife_data: pl.DataFrame,
    x_column: str,
    halflife_column: str = "halflife",
    additional_mappings: dict = None,
    alpha: float = 0.75,
    fillalpha: float = 0.5,
    lw: float = 1,
    **kwargs
) -> gp.GrizzlyPlot:
    """
    Generate default halflife GrizzlyPlot
    for this project.

    Parameters
    ----------
    halflife_data : pl.DataFrame
        Polars DataFrame containing
        data to plot.
    x_column : str
        Which column of the dataframe contains
        x-values for the violins to plot.
    halflife_column : str
        Which column of the dataframe contains
        posterior draws for the halflives (
        which will go into the density estimates
        for the violins to plot)
    additional_mappings : dict
        Additional user-specified aesthetic mappings
        beyond the required x and y mappings. Default
        None.
    alpha : float
        Transparency of the points and lines.
        Default 0.75
    fillalpha : float
        Transparency of the violins. Default 0.5
    lw : float
        Line width of the points and lines.
        Default 1.
    **kwargs
        Other keyword arguments passed to
        GrizzlyPlot constructor

    Returns
    -------
    halflife_plot : gp.GrizzlyPlot
        A defined GrizzlyPlot of halflife estimates
    """
    if additional_mappings is None:
        additional_mappings = dict()
    halflife_plot = gp.GrizzlyPlot(
        data=halflife_data,
        mapping=dict(
            x=x_column,
            y=halflife_column,
            **additional_mappings
        ),
        geoms=[
            geoms.GeomViolin(
                name="halflives",
                violinwidth=5,
                linecolor="none",
                norm="max",
                color="k",
                trimtails=0.005,
            ),
            geoms.GeomPointIntervalY(
                markersize=14, lw=6, color="k", alpha=1
            ),
        ],
        alpha=alpha,
        fillalpha=fillalpha,
        lw=lw,
        **kwargs
    )

    return halflife_plot


def left_align_yticklabels(
    ax: mpl.axes.Axes, ylab_pad: float = default_ylab_pad
) -> mpl.axes.Axes:
    """
    Left align y tick labels for a given
    axis.

    Particularly useful to prevent different length
    numerals from changing horizontal positioning of
    the actual x=0 gridline.

    Parameters
    ----------

    ax: mpl.axes.Axes:
        the Matplotlib axes object for which to
        align the ticklabels

    ylabl_pad: float
        how much to pad the labels from
        the ticks. Defaults to the overall
        module default value.

    Returns
    -------

    ax:
        The manipulated axes object.

    """
    # this makes the tick locator fixed,
    # if it isn't already
    # avoiding a warning message
    ax.set_yticks(ax.get_yticks())

    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontdict={"horizontalalignment": "left"},
    )
    ax.tick_params(axis="y", pad=ylab_pad)

    return ax


def rotate_xticklabels(
    ax: mpl.axes.Axes, rotation: float = 45, **kwargs
) -> mpl.axes.Axes:
    """
    Rotate x ticklabels to a given angle.

    Performs rotation with defaults chosen
    so that long labels don't overlap but
    still point clearly to their respective
    ticks.

    Parameters
    ----------

    ax: mpl.axes.Axes:
        the Matplotlib axes object for which to rotate the ticklabels

    rotation: float
         the rotation angle in degrees. 0
         degrees is parallel to the x-axis,
         with the label reading left to
         right. Default 45.

    **kwargs:
         other keyword arguments passed
         to ax.set_ticklabels()

    Returns
    -------

    ax:
        The manipulated axes object.
    """

    # this makes the tick locator fixed,
    # if it isn't already,
    # avoiding a warning message
    ax.set_xticks(ax.get_xticks())

    # this performs the rotation
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation_mode="anchor",
        ha="right",
        rotation=rotation,
        **kwargs
    )

    return ax


def multiunit_time_ticks(
    values: ArrayLike, units: ArrayLike, axis_unit: str
) -> tuple[list, list]:
    """
    Produce a set of tick labels for a
    time axis with variable units,
    and place them at the appropriate
    points given the fundamental unit
    of the axis.

    Parameters
    ----------
    values : ArrayLike
        Tick label time values in the units in which they will be
        displayed
    units : ArrayLike
        Corresponding tick label time units, which will be displayed
        alongside the values to avoid ambiguity
    axis_unit : str
        Fundamental unit of the axis (i.e. time unit of the actual
        provided data)

    Returns
    -------
    tick_values, tick_labels : tuple[list, list]
        where to place the ticks on the axis
        and how to label them
    """

    tick_values = [
        analyze.convert_time_to(val, unit_from, axis_unit)
        for val, unit_from in zip(values, units)
    ]

    tick_labels = [
        "${:d}${}".format(val, unit)
        for val, unit in zip(values, units)
    ]

    return tick_values, tick_labels


def get_annotation_string(
    model: HalfLifeModel,
    log_halflife_name: str = "\\ln(h_j)",
    intercept_mode_name: str = "<v_0>_j",
    intercept_sd_name: str = "\\sigma_j",
):
    """
    Get a string for annotating a plot with
    a math-formatted listing of the key prior
    values, extracting those from a given
    Pyter halflife model assuming all priors
    are Normal or postiive-constrained
    TruncatedNormal. Specific to this paper.

    Parameters
    ----------
    axis :
       Axis to annotate.

    model : pyter.models.AbstractModel
       Pyter model from which to draw annotation
       prior values

    log_halflife_name : str
       Name to give the log halflife parameter in
       the annotation, in TeX-like math notation.
       Backslashes should be escaped.
       Default '\\ln(h_j)'.

    intercept_mode_name : str
       Name to give the titer intercept mode parameter
       in the annotation, in TeX-like math notation.
       Backslashes should be escaped.
       Default '$<v_0>_j$'.

    intercept_sd_name : str
       Name to give the titer intercept standard deviation
       parameter in the annotation, in TeX-like math
       notation. Backslashes should be escaped.
       Default '$\\sigma_j$'.

    Return
    ------
    matplotlib.axes._axes.Axes
       The annotated axis.
    """

    prior_dists = analyze.extract_distribution_params(
        model
    )
    hl_loc = np.exp(
        prior_dists["log_halflife_distribution"]["loc"]
    )
    hl_scale = np.exp(
        prior_dists["log_halflife_distribution"]["scale"]
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

    log_halflife_string = (
        "${} \\sim \\mathcal{{N}}("
        "\\ln({:g}), \\ln({:g}))$".format(
            log_halflife_name, hl_loc, hl_scale
        )
    )
    intercept_mode_string = (
        "${} \\sim \\mathcal{{N}}("
        "{:g}, {:g})$".format(
            intercept_mode_name,
            int_mode_loc,
            int_mode_scale,
        )
    )
    intercept_sd_string = (
        "${} \\sim \\mathcal{{N}}^+("
        "{:g}, {:g})$".format(
            intercept_sd_name, int_sd_loc, int_sd_scale
        )
    )

    return (
        log_halflife_string
        + ",\n"
        + intercept_mode_string
        + ",\n"
        + intercept_sd_string
    )
