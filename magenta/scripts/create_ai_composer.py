# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from shutil import copyfile
import magenta as mg
import numpy as np
import os
import csv
import copy
import random

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None,
        'Root directory for midi files.')
tf.app.flags.DEFINE_string('output_dir', None,
        'Output directory')
tf.app.flags.DEFINE_string('base_dir', None,
        'Base directory')
tf.app.flags.DEFINE_string('cond_dir', None,
        'Cond directory')
tf.app.flags.DEFINE_string('csv', None,
        'CSV metadata file.')
tf.app.flags.DEFINE_integer('length', 30,
        'Length of each sample')
tf.app.flags.DEFINE_integer('participants', 6,
        'Num experiments')
tf.app.flags.DEFINE_integer('tests', 8,
        'Number of comparisons to make')

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    if None in [FLAGS.input_dir, FLAGS.output_dir, FLAGS.csv, FLAGS.base_dir, FLAGS.cond_dir]:
        tf.logging.fatal('--input_dir --output_dir --csv --files --base_dir --cond_dir required.')
        return

    composers = ['Frédéric Chopin',
        'Johann Sebastian Bach', 
        'Claude Debussy', 
        'Ludwig van Beethoven'
        ]
    
    split = ['test', 'validation']
    output_dir = os.path.expanduser(FLAGS.output_dir)
    input_dir = os.path.expanduser(FLAGS.input_dir)

    os.mkdir(FLAGS.base_dir)
    os.mkdir(FLAGS.cond_dir)

    df_csv = os.path.expanduser(FLAGS.csv)
    df = pd.read_csv(df_csv)
    df = df[df.split.isin(split)]

    start_times = [x for x in range(5, 600, 25)]
    df = df[df.canonical_composer.isin(composers)]
    print(df.groupby("canonical_composer").count())

    orderings = np.tile(['AXY', 'AYX', 'YAX', 'YXA', 'XYA', 'XAY'], int(FLAGS.participants / 6))
    orderings = np.tile(orderings, (FLAGS.tests, 1))
    for x in orderings:
      random.shuffle(x)

    composer_abb = np.tile(['C', 'J', 'D', 'B'], int(FLAGS.tests / 4))

    output_filename = "participant"
    composer_dict = {"C": 'Frédéric Chopin', "J":'Johann Sebastian Bach', "D":'Claude Debussy', "B":'Ludwig van Beethoven'}
    output_dict = {'A': FLAGS.output_dir, 'X': FLAGS.cond_dir, 'Y': FLAGS.base_dir}
    new_orderings = []
    for i in range(FLAGS.participants):
        composer_ordering = copy.copy(composer_abb)
        random.shuffle(composer_ordering)
        composer_filenames = [random.choice(df[df.canonical_composer.str.contains(composer_dict[c])].midi_filename.values) for c in composer_abb]
        new_orderings.append(np.array(composer_ordering))
        for j, fname in enumerate(composer_filenames):
            cname = "{}_{}_test_{}".format(output_filename, i, j)
            start_time = random.choice(start_times)
            sequence = mg.music.midi_file_to_sequence_proto(input_dir+fname)

            while not start_time + 5 + (FLAGS.length*2) < sequence.total_time:
              start_time = random.choice(start_times)

            sequence = mg.music.extract_subsequence(sequence, start_time, start_time+(FLAGS.length*2))
            subseqs = mg.music.split_note_sequence(sequence, FLAGS.length)

            primer, output = subseqs[0], subseqs[1]
            file_dict = {'A': output, 'X': primer, 'Y': primer}
            for x, order in enumerate(orderings[j, i]):
              cur_dir = output_dict[order]
              midi_filename = "{}_{}.midi".format(cname, order)
              print("{}/{}".format(cur_dir, midi_filename))
              sequence = file_dict[order]
              mg.music.sequence_proto_to_midi_file(sequence, cur_dir+'/'+midi_filename)

    new_orderings = np.array(new_orderings).T
    with open(os.path.join(output_dir, "orderings.csv"),"w+") as f:
      csvWriter = csv.writer(f,delimiter=',')
      csvWriter.writerows(orderings)
      csvWriter.writerows(new_orderings)

def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()
