from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.gridspec as gridspec

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
    text_units = np.cumsum(text_units[text_units != 0]) + 1
    # ax.set_xticks(list(range(number_paragraphs+2)))
    ax.set_xticks(text_units)
    ax.set_xticklabels(range(1, len(text_units)+1))

    plt.show()

    
    
def heatmap_simp(result_dict,decoded_weight_matrix,decoded_score_matrix, example=0, decoding_layer=0,num_multi_head=0,plot=True, ax=[]):

    step=result_dict['longest_beam_array'][example]

    number_of_textual_units=result_dict["number_of_textual_units"][example]
    text_units = np.cumsum(number_of_textual_units[number_of_textual_units != 0])

    longest_beam_array=(result_dict["longest_beam_array"]-1).astype("int")


    aux=decoded_weight_matrix[example, np.argmax( decoded_score_matrix[example, :, longest_beam_array[example]]), :, decoding_layer, num_multi_head, :]
    aux=aux[:, np.max(aux, axis=0) > 1e-10]
    aux=aux[np.max(aux, axis=1) > 1e-10,:]
    # Calculating the output and storing it in the array Z
    x = np.arange(0, aux.shape[0], 1)
    y = np.arange(0, aux.shape[1], 1)
    X, Y = np.meshgrid(x, y)


    Z = aux[X, Y]

    if plot:
        fig, ax = plt.subplots(figsize=(20,10))

    for p in text_units:
        ax.plot(x,np.repeat(p, aux.shape[0]))

    im = ax.imshow(Z, cmap='hot', extent=(0, aux.shape[0], aux.shape[1], 0), aspect='auto')#, interpolation='bilinear')



    bar= plt.colorbar(im);
    bar.set_label('Attention')


    ax.set_yticks(y[::2], minor=True)
    ax.set_yticklabels(y[::2], minor=True )
    
    ax.set_yticks(np.cumsum(number_of_textual_units[number_of_textual_units != 0])-number_of_textual_units[number_of_textual_units != 0]/2)
    ax.set_yticklabels(["Doc " +str(i) for i in range(0, len(text_units))])
    ax.tick_params(axis='y', which='major', length=20)

    ax.set_ylabel('Paragraph')
    ax.set_xlabel('step')
    if plot:
        plt.show()
    else:
        return ax
    
def heatmap_multi(result_dict,decoded_weight_matrix,decoded_score_matrix, decoding_layer=0,num_multi_head=0,size=(40,8)):    
    fig = plt.figure(figsize=size)

    examples, beam_size, number_steps, _, _, number_paragraphs = decoded_weight_matrix.shape
    spec=gridspec.GridSpec(2,examples//2, wspace=0.6, hspace=0.3)

    for i in range(examples):

        ax = fig.add_subplot(spec[i//(examples//2),i%(examples//2)])

        example = i
        ax=heatmap_simp(result_dict,decoded_weight_matrix,decoded_score_matrix, example, decoding_layer,num_multi_head,False,ax)
        ax.set_title(r"Example: $%d$" % (example))

    plt.show()
    
    
def heatmap_dec_layer(result_dict,decoded_weight_matrix,decoded_score_matrix,num_multi_head=0, size=(40, 70),save=False):
    fig = plt.figure(figsize=size)
    examples, beam_size, number_steps, num_decoding_layer, _ , number_paragraphs = decoded_weight_matrix.shape

    outer = gridspec.GridSpec(examples, 1, wspace=0.2, hspace=0.4)

    for i in range(examples):
        inner = gridspec.GridSpecFromSubplotSpec(1, num_decoding_layer, subplot_spec=outer[i], wspace=0.7, hspace=0.2)
        for j in range(num_decoding_layer):
            ax = fig.add_subplot( inner[j])
            ax=heatmap_simp(result_dict,decoded_weight_matrix,decoded_score_matrix, i, j,num_multi_head,False,ax)
            if j==0:
                ax.set_title(r"Example $%d$"% (i) +" \n\n" + r" Dec_layer: $%d$" % (j))
            else:
                ax.set_title(r"Dec_layer: $%d$" % (j))
    
    fig.suptitle("Attentions for all examples over all decoding layers and attetion head {}".format(num_multi_head))   
    if save:
        plt.savefig('saved_figs/dec_layer.svg',facecolor="white")
    plt.show()
    
def heatmap_att_head(result_dict,decoded_weight_matrix,decoded_score_matrix,example=0, size=(40, 70),save=False):
    fig = plt.figure(figsize=size)
    examples, beam_size, number_steps, num_decoding_layer, num_multi_head , number_paragraphs = decoded_weight_matrix.shape

    outer = gridspec.GridSpec(num_multi_head, 1, wspace=0.2, hspace=0.4)

    for i in range(num_multi_head):
        inner = gridspec.GridSpecFromSubplotSpec(1, num_decoding_layer, subplot_spec=outer[i], wspace=0.7, hspace=0.2)
        for j in range(num_decoding_layer):
            ax = fig.add_subplot( inner[j])
            ax=heatmap_simp(result_dict,decoded_weight_matrix,decoded_score_matrix, example, j,i,False,ax)
            if j==0:
                ax.set_title(r"Multi head $%d$"% (i) +" \n\n" + r" Dec_layer: $%d$" % (j))
            else:
                ax.set_title(r"Dec_layer: $%d$" % (j))
            
    plt.suptitle("Attentions for Example number {} over all decoding layers and attetion heads".format(example)) 
    if save:
        plt.savefig('saved_figs/att_head.svg',facecolor="white")
    plt.show()