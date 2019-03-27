import os
import glob
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn import manifold 
from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "input_dir", None,
    "Directory for the generations")
tf.app.flags.DEFINE_string(
    "projection", "PCA",
    "Type of projection")
tf.app.flags.DEFINE_string(
    "out", None, "dest for out")
tf.app.flags.DEFINE_bool(
    "mean", False, "Whether to run mean dist")


def main(unused_args):
    labels = ["bach", "beethoven", "debussy", "chopin"]
    composers = {x:[] for x in labels}
    for composer in composers:
        path = os.path.join(FLAGS.input_dir, composer)
        print(path)
        for file in glob.glob(path+"/*.dump"):
            with open(file, "rb") as f:
                composers[composer].append(pickle.load(f))

    all_vectors = []
    composer_lengths = [0] * 4
    num_samples = 800
    for i, composer in enumerate(composers):
        # tf.logging.info(len(composers[composer]))
        # tf.logging.info(composers[composer][0].shap
        if not composers[composer]:
            print("composer not found: " + composer)
            continue
        if type(composers[composer][0]) is list:
            cur_vectors = list(np.array([x for l in composers[composer] for x in l]))
            composers[composer] = cur_vectors
            all_vectors += cur_vectors
            composer_lengths.append(len(cur_vectors))
        else:
            all_vectors += composers[composer][:num_samples]
            composers[composer] = composers[composer][:num_samples]
            composer_lengths[i] = len(composers[composer][:num_samples])

    all_vectors = np.array(all_vectors) 
    [print(len(composers[x]), x) for x in composers]
    print('nan', print(np.isnan(all_vectors)))
    all_targets = np.array([y for i, x in enumerate(composers) for y in [x]*composer_lengths[i]])
    if FLAGS.out:
        file = FLAGS.out + '_vectors.dump'
        with open(file, 'wb') as fp:
            pickle.dump(all_vectors, fp)
            tf.logging.info('Wrote vector file to %s', file)
        file = FLAGS.out + '_labels.dump'
        with open(file, 'wb') as fp:
            pickle.dump(all_targets, fp)
            tf.logging.info('Wrote labels file to %s', file)
        exit()
    elif FLAGS.mean:
        # orderings = [("bach", "bach"), ("bach", "beethoven"), ("bach", "debussy"), ("bach", "chopin"), 
        #              ("beethoven", "beethoven"), ("beethoven","debussy"), ("beethoven", "chopin")]
        values, composer_vals = [], []
        intra = []
        for i, composeri in enumerate(labels):
            curi = composers[composeri]
            # curi = all_vectors[(i*num_samples):(i+1)*num_samples]
            print(len(curi))
            # print(len(curi), composeri, "i")
            for j, composerj in enumerate(labels):
                curj = composers[composerj]
                # curj = all_vectors[(j*num_samples):(j+1)*num_samples]
                # print("distances", len(curj), len(curi), cdist(curi, curj).shape)
                # exit()
                
                # print(cdist(curi, curj))
                x = np.mean(cdist(curi, curj))
                print(composeri, composerj, x)
                values.append(x)
                if composeri == composerj:
                    print('appending')
                    intra.append(x)
                composer_vals.append((composeri, composerj))


        values = np.array(values)
        # composer_vals = np.array(composer_vals)
        new_vals = values.reshape(4, 4)
        # composer_vals = composer_vals.reshape(4, 4)
        plt.imshow(new_vals, cmap='hot', interpolation='nearest')
        plt.xticks(range(4), labels)
        plt.yticks(range(4), labels)
        # plt.show()
        
        # norm_values = [100*float(val)/sum(values) for val in values]
        norm_values = (values - values.min()) / (values.max() - values.min())
        intra = np.array(intra)
        # intra = [100*float(val)/sum(intra) for val in intra]
        intra = (intra - intra.min()) / (intra.max() - intra.min())
        print('mean', np.mean(norm_values))
        print('intramean', np.mean(intra))
        print('std', np.std(norm_values))
        print('intrastd', np.std(intra))
        print('diff', np.mean(norm_values)-np.mean(intra))
        print('diffvar', np.var(norm_values)-np.var(intra))
        # norm_values = values
        # norm_values = normalize(values.reshape(1, -1), norm='max').T
        for val, comp in zip(norm_values, composer_vals):
            # val = val[0]

            print(comp[0], comp[1], round(val, 4))
        exit()

    projections = {"PCA": PCA, "TSNE": TSNE}
    tf.logging.info("Using " + FLAGS.projection)

    components = None
    if FLAGS.projection == "TSNE":
        components = PCA(n_components=256).fit_transform(all_vectors)
        components = TSNE(n_components=2, perplexity=40, n_iter=6000, early_exaggeration=100, verbose=1).fit_transform(all_vectors)
        # components = manifold.Isomap(n_neighbors=20, n_components=2, n_jobs=-1).fit_transform(all_vectors)
    else:
        components = PCA(n_components=2).fit_transform(all_vectors)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)#, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_zlabel('Principal Component 3', fontsize = 15)
    # ax.set_title(os.path.dirname(FLAGS.input_dir) + ': 2 component ' + FLAGS.projection, fontsize = 20)
    ax.set_title("2 component t-SNE on CondRNN hidden states")
    tf.logging.info(len(components))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    cmap = cm.get_cmap('Set1', 9)
    cmap = cmap.colors
    colours = [cmap[i] for i in [1, 2, 4, 7]]
    curIndex = [0, composer_lengths[0]]
    for label, colour, index in zip(["Bach", "Beethoven", "Debussy", "Chopin"], colours, composer_lengths):
         cindices = components[curIndex[0]:curIndex[1]]
         curIndex[0] += index
         curIndex[1] += index
         ax.scatter(cindices[:,0], cindices[:,1], c=[colour], label=label, alpha=0.45, s=40, edgecolor='')

    ax.grid()
    ax.legend()
    plt.show()


def console_entry_point():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
