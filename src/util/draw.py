import matplotlib.pyplot as plt
import numpy as np


def update_matrix(fig, ax, dims, point, threshold=-50, state=None, prev_point=None):
    dims[prev_point] = -1
    dims[point] = -65

    textcolors = ["white", "black"]
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    im = heatmap(dims, ax=ax, cmap="RdYlGn")
    # kw.update(color=textcolors[im.norm(dims[point[1] - 1, point[0] - 1]) > threshold])

    j = prev_point[1]
    i = prev_point[0]
    index = j * dims.shape[0] + i

    next(x for x in im.axes._axes.texts if x._y == i - 0.4 and x._x == j)._text = str(round(state[index, 0], 2))
    next(x for x in im.axes._axes.texts if x._y == i and x._x == j + 0.3)._text = str(round(state[index, 1], 2))
    next(x for x in im.axes._axes.texts if x._y == i + 0.4 and x._x == j)._text = str(round(state[index, 2], 2))
    next(x for x in im.axes._axes.texts if x._y == i and x._x == j - 0.3)._text = str(round(state[index, 3], 2))

    # fig.canvas.draw()
    fig.canvas.flush_events()


def draw_matrix(dims, point, data=None, threshold=-50, state=None):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    dims = np.copy(dims)
    buff = dims[point]
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
            if (i == point[0] and j == point[1]):
                dims[i, j] = buff
            im.axes.text(j, i, dims[i, j], horizontalalignment='center', verticalalignment='center')

            index = j * dims.shape[0] + i
            im.axes.text(j, i - 0.4, str(round(state[index, 0], 2)), horizontalalignment='center',
                         verticalalignment='center')
            im.axes.text(j + 0.3, i, str(round(state[index, 1], 2)), horizontalalignment='center',
                         verticalalignment='center')
            im.axes.text(j, i + 0.4, str(round(state[index, 2], 2)), horizontalalignment='center',
                         verticalalignment='center')
            im.axes.text(j - 0.3, i, str(round(state[index, 3], 2)), horizontalalignment='center',
                         verticalalignment='center')

    fig.tight_layout()

    plt.ion()
    plt.show()

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


def draw_plot(log):
    plt.plot(log)
    plt.xlabel('iteration')
    plt.ylabel('profit')
    plt.show()
