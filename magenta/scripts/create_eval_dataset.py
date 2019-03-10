# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from shutil import copyfile
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', None,
        'Output directory')
tf.app.flags.DEFINE_string('csv', None,
        'CSV metadata file.')

def main(unused_argv):

	composers = ['Frédéric Chopin',
        'Johann Sebastian Bach', 
        'Claude Debussy', 
        'Ludwig van Beethoven'
        ]
	split = 'test'

def console_entry_point():
    tf.app.run(main)

if __name__ == '__main__':
    console_entry_point()