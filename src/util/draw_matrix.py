import matplotlib.pyplot as plt
import numpy as np


def update_matrix(fig, ax, dims, point, threshold=-50):
    dims = np.copy(dims)
    dims[point] = -65

    textcolors = ["white", "black"]
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    im = heatmap(dims, ax=ax, cmap="RdYlGn")
    kw.update(color=textcolors[im.norm(dims[point[1], point[0]]) > threshold])
    fig.canvas.draw()
    fig.canvas.flush_events()


def draw_matrix(dims, point, data=None, threshold=-50):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    dims = np.copy(dims)
    dims[point] = -65
    textcolors = ["white", "black"]

    fig, ax = plt.subplots()
    im = heatmap(dims, ax=ax, cmap="RdYlGn")

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(dims.shape[0]):
        for j in range(dims.shape[1]):
            kw.update(color=textcolors[im.norm(dims[i, j]) > threshold])
            im.axes.text(j, i, data[i, j], **kw)

    fig.tight_layout()

    plt.ion()
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return (fig, ax)


def heatmap(data, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im
