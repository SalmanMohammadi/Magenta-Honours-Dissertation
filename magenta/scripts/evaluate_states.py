import os
import glob
import pickle

import magenta as mg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import manifold 
from sklearn.manifold import TSNE

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "input_dir", None,
    "Directory for the generations")
tf.app.flags.DEFINE_string(
    "projection", "PCA",
    "Type of projection")


def main(unused_args):
    labels = ["bach", "beethoven", "debussy", "chopin"]
    composers = {x:[] for x in labels}
    labels_indices = []
    for composer in composers:
        path = os.path.join(FLAGS.input_dir, composer)
        for file in glob.glob(path+"/*.dump"):
            with open(file, "rb") as f:
                composers[composer].append(pickle.load(f))
                labels_indices.append(composer)

    all_vectors = []
    composer_lengths = []
    for composer in composers:
        # tf.logging.info(len(composers[composer]))
        # tf.logging.info(composers[composer][0].shape)
        cur_vectors = list(np.array([x for l in composers[composer] for x in l]))
        all_vectors += cur_vectors
        composer_lengths.append(len(cur_vectors))

    all_vectors = np.array(all_vectors) 
    all_targets = [y for i, x in enumerate(composers) for y in [x]*composer_lengths[i]]

    projections = {"PCA": PCA, "TSNE": TSNE}
    tf.logging.info("Using " + FLAGS.projection)

    components = None
    if FLAGS.projection == "TSNE":
        components = PCA(n_components=256).fit_transform(all_vectors)
        components = TSNE(n_components=2, perplexity=200, n_iter=10000, verbose=1).fit_transform(components)
        # components = manifold.Isomap(n_neighbors=20, n_components=2, n_jobs=-1).fit_transform(all_vectors)
    else:
        components = PCA(n_components=2).fit_transform(all_vectors)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)#, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title(os.path.dirname(FLAGS.input_dir) + ': 2 component ' + FLAGS.projection, fontsize = 20)


    colours = ['y', 'r', 'm', 'b']
    indices = [len(composers[composer]) for composer in composers]
    curIndex = [0, indices[0]]
    for label, colour, index in zip(labels, colours, indices):
         cindices = components[curIndex[0]:curIndex[1]]
         curIndex[0] += index
         curIndex[1] += index
         ax.scatter(cindices[:,0], cindices[:,1], c=colour, label=label, alpha=0.5, s=10)

    ax.grid()
    ax.legend()
    plt.show()


def console_entry_point():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
