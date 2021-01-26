from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)


def histo_3d_multi(decoded_weight_matrix, decoded_score_matrix, longest_beam_array, decoding_layer=0):
    fig = plt.figure(figsize=(20, 10))

    examples, beam_size, number_steps, _, _, number_paragraphs = decoded_weight_matrix.shape

    for i in range(1, examples+1):

        ax = fig.add_subplot(2, examples/2, i, projection='3d')
        example = i-1

        xs = np.arange(0, number_paragraphs+2, 1)
        verts = []
        zs = np.arange(0, number_steps, 1)
        z_max_lim = 0
        for z in zs:
            ys = []
            ys = np.append(np.append(0, decoded_weight_matrix[example, np.argmax(
                decoded_score_matrix[example, :, longest_beam_array[example]]), z, decoding_layer, 0, :]), 0)
            verts.append(list(zip(xs, ys)))
            if ys.max() > z_max_lim:
                z_max_lim = ys.max()

        poly = PolyCollection(
            verts, facecolors=[plt.cm.jet(x) for x in np.random.rand(300)])
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        ax.set_title(r"Example: $%d$" % (example))
        ax.set_xlabel('Paragraph')
        ax.set_xlim3d(0, number_paragraphs)
        ax.set_ylabel('step')
        ax.set_ylim3d(0, number_steps)
        ax.set_zlabel('Attention')
        ax.set_zlim3d(0, z_max_lim)

    plt.show()


def histo_3d_simple(decoded_weight_matrix, decoded_score_matrix, number_of_textual_units, longest_beam_array, decoding_layer=0, example=0):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    examples, beam_size, number_steps, _, _, number_paragraphs = decoded_weight_matrix.shape

    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.6)

    xs = np.arange(0, number_paragraphs + 2, 1)
    verts = []
    zs = np.arange(0, number_steps, 1)
    z_max_lim = 0
    for z in zs:
        ys = []
        ys = np.append(np.append(0, decoded_weight_matrix[example, np.argmax(
            decoded_score_matrix[example, :, longest_beam_array[example]]), z, decoding_layer, 0, :]), 0)
        verts.append(list(zip(xs, ys)))
        if ys.max() > z_max_lim:
            z_max_lim = ys.max()

    poly = PolyCollection(
        verts, facecolors=[plt.cm.jet(x) for x in np.random.rand(300)])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('Document End')
    ax.set_xlim3d(0, number_paragraphs)
    ax.set_ylabel('step')
    ax.set_ylim3d(0, number_steps)
    ax.set_zlabel('Attentiomn')
    ax.set_zlim3d(0, z_max_lim)

    text_units = number_of_textual_units[example]
    text_units = np.cumsum(text_units[text_units != 0])
    # ax.set_xticks(list(range(number_paragraphs+2)))
    ax.set_xticks(text_units)
    ax.set_xticklabels(
        range(1, len(text_units)+1))

    plt.show()
