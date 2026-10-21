"""
File: plotting.py
Author: Patrick Hannawald (patrick.hannawald@dlr.de)
Date: 2026-01-13
Description: Utility functions for plotting
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skspatial.objects import Vector
import numpy as np
import skimage as ski
from typing import Literal, Sequence

def plot_comparison(image1, image2, n_tiles: tuple[int, int]=(8, 8), zoom_in: int=0, figsize: tuple[float, float]=(6, 6), cmap: str="gray"):
    """Compare two images (of equal size) by making a checkerboard between those two images.
    
    You can zoom in to the middle of the images with zoom_in >= 2
    """
    compared = ski.util.compare_images(image1, image2, method='checkerboard', n_tiles=n_tiles)
    if image1.shape != image2.shape:
        raise ValueError(f"Shape of image1 ({image1.shape}) not equal to image2 ({image2.shape})!")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = figsize)
    ax.imshow(compared, cmap = cmap)
    ax.figure.colorbar(ax.images[0], ax=ax)
    if zoom_in >= 2:
        zoom_factor = zoom_in * 2
        ax.set_xlim(image1.shape[1]//2 - image1.shape[1]//zoom_factor, 
                    image1.shape[1]//2 + image1.shape[1]//zoom_factor)
        ax.set_ylim(image1.shape[0]//2 - image1.shape[0]//zoom_factor, 
                    image1.shape[0]//2 + image1.shape[0]//zoom_factor)


def plot_image(ax, img, x=None, y=None, 
               west=None, east=None, north=None, south=None, 
               title='', cmap='gray', 
               xlabel='km along longitude', ylabel='km along latitude',
               adjust_contrast_factor: float|None = None,
               xlim: tuple[float, float]|None = None,
               ylim: tuple[float, float]|None = None,
               title_fontsize: int|None = None,
               label_fontsize: int|None = None,
               colorbar_label: str|None=None):
        """
        Plot an image on a given axes with custom extent.

        Parameters:
        ax (Axes): The axes to plot on.
        img (numpy array): The image to plot.
        x (list[float], optional): The x-coordinates of the image. Defaults to None.
        y (list[float], optional): The y-coordinates of the image. Defaults to None.
        west (float, optional): The westernmost longitude of the bounding box. Defaults to None.
        east (float, optional): The easternmost longitude of the bounding box. Defaults to None.
        north (float, optional): The northernmost latitude of the bounding box. Defaults to None.
        south (float, optional): The southernmost latitude of the bounding box. Defaults to None.
        title (str, optional): The title of the plot. Defaults to ''.
        cmap (str, optional): The colormap to use. Defaults to 'gray'.
        adjust_contrast_factor (float, optional): Scale image values between MEAN+-adjust_contrast_factor*SD. Lower values => higher contrast
        """
        if x is not None and y is not None:
            extent = [min(x), max(x), min(y), max(y)]
        elif west is not None and east is not None and north is not None and south is not None:
            extent = [west[0], east[0], south[1], north[1]]
        else:
            extent = None
        if adjust_contrast_factor is not None:
            mean = np.mean(img)
            sd = np.std(img)
            vmin = mean - adjust_contrast_factor*sd
            vmax = mean + adjust_contrast_factor*sd
        else:
            vmin, vmax = None, None
        ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.figure.colorbar(ax.images[0], ax=ax, label=colorbar_label)

def arrow(ax, origin: tuple[float, float], angle_degrees: float, arrow_length: float, color = 'red', shrink: float=0.05):
    """Adds an arrow a (image) plot from origin to arrow_length distance in angle_degrees direction."""
    center_x, center_y = origin
    
    # Calculate the end point of the arrow
    angle_radians = np.deg2rad(angle_degrees)    
    end_x = center_x + arrow_length * np.cos(angle_radians)
    end_y = center_y + arrow_length * np.sin(angle_radians)
    
    ax.annotate("", xy=(end_x, end_y), xytext=(center_x, center_y), 
                arrowprops=dict(facecolor=color, shrink=shrink))

def plot_histogram(
    ax: plt.Axes, img: np.ndarray, *, bins: int | Sequence[float] = 50,
    range: tuple[float, float] | None = None, density: bool = False, log: bool = False,
    title: str = "", xlabel: str = "Pixel value", ylabel: str = "Count",
    color: str = "steelblue", edgecolor: str = "black", alpha: float = 0.8, 
    show_stats: bool = True,
    stats_color: str = "red",
    stats_linestyle: Literal["-", "--", "-.", ":"] = "--",
    stats_linewidth: float = 1.5,
    title_fontsize: int | None = None, label_fontsize: int | None = None, tick_fontsize: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot a histogram of the pixel values of an image (or any 1-D numeric array).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the histogram will be drawn.
    img : np.ndarray
        Image (or 1-D array) whose pixel/value distribution is to be plotted.
        ``NaN`` and ``inf`` values are ignored automatically.
    bins : int or sequence of float, default 50
        Number of histogram bins or explicit bin edges.
    range : tuple(float, float) or None, default None
        Lower and upper range of the bins.  Values outside the range are ignored.
    density : bool, default False
        If True, the histogram is normalized to form a probability density,
        i.e. the integral of the histogram is 1.
    log : bool, default False
        If True, the y-axis is plotted on a log scale.
    title, xlabel, ylabel : str, optional
        Axis titles and labels.
    color, edgecolor, alpha : str / float, optional
        Appearance of the bars.
    show_stats : bool, default True
        If True, vertical lines for the mean and ±1 sigma are drawn.
    stats_color, stats_linestyle, stats_linewidth : optional
        Appearance of the statistical guide lines.
    title_fontsize, label_fontsize, tick_fontsize : int or None, optional
        Font sizes for the title, axis labels and tick labels.
    Returns
    -------
    hist, bin_edges, patches : tuple(np.ndarray, np.ndarray, np.ndarray)
        The histogram counts (or density), the bin edges and the list of
        `matplotlib.patches.Rectangle` objects that were created by ``ax.hist``.
        Returning them makes it easy to further customise the plot after the
        call (e.g. add a second y-axis).

    Notes
    -----
    * ``NaN`` and infinite values are removed before the histogram is computed.
    * The function does **not** call ``plt.show()`` - the caller decides when
      to render or save the figure.
    * The signature is deliberately similar to ``plot_image`` so you can
      replace calls in a loop without touching the surrounding code.
    """

    flat = img.ravel()
    flat = flat[np.isfinite(flat)]          # removes NaN and ±inf

    hist, bin_edges, patches = ax.hist(
        flat, bins=bins, range=range, density=density,
        log=log, color=color, edgecolor=edgecolor, alpha=alpha
    )

    # Add optional statistical guide lines (mean ± 1sigma)
    if show_stats and flat.size > 0:
        mean = flat.mean()
        std = flat.std()
        for i, (val, label) in enumerate([(mean, "mean, mean±sigma"), (mean - std, "mean-sigma"), (mean + std, "mean+sigma")]):
            ax.axvline(
                val,
                color=stats_color,
                linestyle=stats_linestyle,
                linewidth=stats_linewidth,
                label=label
            )
        # Show a legend only for the stats lines (avoid duplicate entries)
        handles, labels = ax.get_legend_handles_labels()
        # Keep only the three stats entries
        stats_idx = [i for i, l in enumerate(labels) if l in {"mean, mean±sigma"}]
        if stats_idx:
            ax.legend([handles[i] for i in stats_idx],
                      [labels[i] for i in stats_idx],
                      loc="upper right",
                      fontsize=label_fontsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if tick_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    return hist, bin_edges, patches



def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    From here: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib by 'stackoverflowuser2010'

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()
