# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from shutil import copyfile
import magenta as mg
import os
import random

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
tf.app.flags.DEFINE_bool('test', False,
        'Whether we want to create a dataset for testing on.')
tf.app.flags.DEFINE_integer('file_slice', None,
        'Number of samples to collect from each file')
tf.app.flags.DEFINE_integer('length', None,
        'Length of each sample')


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

    composers_paths = dict(zip(composers, ['chopin', 'bach', 'debussy', 'beethoven']))
    
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
    elif FLAGS.test:
        tf.logging.info("Creating a dataset for testing.")
        split = ['test', 'validation']

    csv = os.path.expanduser(FLAGS.csv)
    df = pd.read_csv(csv)
    if FLAGS.test:
        df = df[df.split.isin(split)]
    else:
        df = df[df.split.str.contains(split)]
    df = df[df.canonical_composer.isin(composers)] 

    files_per_composer = int(num_files/len(composers))
    filenames = []
    composers_out = []
    for composer in composers:
        curnames = list(df[df.canonical_composer.str.contains(composer)].sort_values('year')[-files_per_composer:].midi_filename.values)
        tf.logging.info(composer + ": " + str(len(curnames)))
        filenames += curnames
        composers_out += [composer for x in range(len(curnames))] 

    file_slice = FLAGS.file_slice
    total_files = len(composers_out)
    if file_slice:
        total_files *= file_slice

    tf.logging.info("Total files: " + str(total_files))

    for filename, composer in zip(filenames, composers_out):
        output_filename = os.path.basename(filename)
        cur_dir = output_dir
        if FLAGS.eval or FLAGS.test:
            composer_dir = composers_paths[composer] + '/'
            if not os.path.exists(os.path.join(cur_dir, composer_dir)):
                os.mkdir(os.path.join(cur_dir, composer_dir))
            cur_dir = os.path.join(cur_dir, composer_dir)
            tf.logging.info('Copying ' + output_filename + " composer: " + composer)
        else:
            tf.logging.info('Copying ' + output_filename)

        sequence = mg.music.midi_file_to_sequence_proto(input_dir+filename)
        
        if file_slice:
            subseqs = mg.music.split_note_sequence(sequence, FLAGS.length)
            for i in range(file_slice):
                subseq = random.choice(subseqs)
                tf.logging.info("Copied slice " + str(i) + '_' + output_filename)
                mg.music.sequence_proto_to_midi_file(subseq, cur_dir +'/'+ str(i) + '_' + output_filename)
        elif FLAGS.length:
            sequence = mg.music.extract_subsequence(sequence, 0, FLAGS.length)
            mg.music.sequence_proto_to_midi_file(sequence, cur_dir+'/'+output_filename)
        else:
            copyfile(input_dir+filename, output_dir + '/' + output_filename)


def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()
