import os
import glob
import pickle

import magenta as mg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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
    for composer in composers:
        path = os.path.join(FLAGS.input_dir, composer)

        for file in glob.glob(path+"/*.dump"):
            with open(file, "rb") as f:
                composers[composer].append(pickle.load(f))

    all_vectors = []
    for composer in composers:
        all_vectors += [np.array(x).reshape((-1 , np.array(x).shape[2])).mean(axis=0) for x in composers[composer]]

    all_vectors = np.array(all_vectors) 
    all_targets = [y for x in composers for y in [x]*len(composers[x])]

    projections = {"PCA": PCA, "TSNE": TSNE}
    tf.logging.info("Using " + FLAGS.projection)
    components = None
    if FLAGS.projection == "TSNE":
        components = PCA(n_components=50).fit_transform(all_vectors)
        components = TSNE(n_components=2).fit_transform(components)
    else:
        components = PCA(n_components=2).fit_transform(all_vectors)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component ' + FLAGS.projection, fontsize = 20)


    colours = ['r', 'g', 'b', 'y']
    indices = [(max(0, x-10), x) for x in range(10, 50, 10)]
    for label, colour, index in zip(labels, colours, indices):
         cindices = all_vectors[index[0]:index[1]]
         ax.scatter(cindices[:,0], cindices[:,1], c=colour, label=label)

    ax.grid()
    ax.legend()
    plt.show()


def console_entry_point():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
