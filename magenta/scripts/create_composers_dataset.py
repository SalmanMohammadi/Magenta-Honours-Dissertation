# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from shutil import copyfile
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None,
        'Root directory for midi files.')
tf.app.flags.DEFINE_string('output_dir', None,
        'Output directory')
tf.app.flags.DEFINE_string('csv', None,
        'CSV metadata file.')
tf.app.flags.DEFINE_integer('files', None,
        'Number of files to include in the dataset.')
tf.app.flags.DEFINE_bool('eval', False,
        'Whether we want to create a dataset for evaluating on.')
tf.app.flags.DEFINE_bool('validate', False,
        'Whether we want to create a dataset for validating on.')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    if None in [FLAGS.input_dir, FLAGS.output_dir, FLAGS.csv, FLAGS.files]:
        tf.logging.fatal('--input_dir --output_dir --csv --files required.')
        return

    composers = ['Frédéric Chopin',
        'Johann Sebastian Bach', 
        'Claude Debussy', 
        'Ludwig van Beethoven'
        ]
    
    num_files = FLAGS.files
    split = 'train'
    output_dir = os.path.expanduser(FLAGS.output_dir)
    input_dir = os.path.expanduser(FLAGS.input_dir)

    if FLAGS.eval:
        tf.logging.info("Creating a dataset for eval.")
        split ='test'
    elif FLAGS.validate:
        tf.logging.info("Creating a dataset for validation.")
        split ='validation'
    
    csv = os.path.expanduser(FLAGS.csv)
    df = pd.read_csv(csv)
    df = df[df.split.str.contains(split)]
    df = df[df.canonical_composer.isin(composers)]

    files_per_composer = int(num_files/len(composers))
    filenames = []
    composers_out = []
    for composer in composers:
        filenames += list(df[df.canonical_composer.str.contains(composer)].sort_values('year')[-files_per_composer:].midi_filename.values)
        composers_out += [composer for x in range(files_per_composer)] 

    for filename, composer in zip(filenames, composers_out):
        output_filename = os.path.basename(filename)
        if FLAGS.eval:
            output_filename = composer + output_filename
            tf.logging.info('Copying ' + output_filename + " composer: " + composer)
        else:
            tf.logging.info('Copying ' + output_filename)
        copyfile(input_dir+filename, output_dir + '/' + output_filename)

def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()
