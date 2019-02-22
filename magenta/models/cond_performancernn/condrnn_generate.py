# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate polyphonic performances from a trained checkpoint.

Uses flags to define operation.
"""

import pickle
import ast
import os
import time
import tempfile

import tensorflow as tf
import magenta

from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn import performance_sequence_generator

from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
from magenta.common import state_util
import models
from models import LSTMModel, LSTMAE, LSTMConfig

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
  'checkpoint_dir', None,
  'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
  'model', 'LSTM',
  'Model: LSTMAE, LSTM')
tf.app.flags.DEFINE_string(
  'checkpoint_number',  None,
  'The checkpoint number to restore from')
tf.app.flags.DEFINE_boolean(
  'gpu',  False,
  'Whether to use CudNN')
tf.app.flags.DEFINE_integer(
  'layers', 2,
  'Number of layers to use')
tf.app.flags.DEFINE_integer(
  'batch_size', 64,
  'Batch size')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.checkpoint_number:
    tf.logging.fatal('--checkpoint_number required')
    return

  model_dict = {'LSTM': LSTMModel, 'LSTMAE': LSTMAE}
  checkpoint_dir = FLAGS.checkpoint_dir

  data_config = models.default_configs['conditional_performance_with_dynamics']
  data_config.hparams.parse(FLAGS.hparams)
  encoder_decoder = data_config.encoder_decoder

  layers = FLAGS.layers
  batch_size = FLAGS.batch_size

  config = LSTMConfig(
    encoder_decoder=encoder_decoder,
    gpu=FLAGS.gpu,
    layers=layers,
    batch_size=batch_size)

  model = model_dict[FLAGS.model](config, 'generate')
  number = FLAGS.checkpoint_number
  bundle_file = checkpoint_dir + 'bundle.mag'
  checkpoint_filename = os.path.join(checkpoint_dir, 'model.ckpt-' + str(number))
  metagraph_filename = os.path.join(checkpoint_dir, 'model.ckpt-' + str(number) + '.meta')

  tf.logging.info(checkpoint_dir)
  with tf.Graph().as_default() as g:
    
    model.build_graph_fn()
    sess = tf.Session(graph=g)
    saver = tf.train.Saver()
    
    tf.logging.info('Checkpoint used: %s', checkpoint_filename)
    saver.restore(sess, checkpoint_filename)
     
    try:
      tempdir = tempfile.mkdtemp()
      checkpoint_filename = os.path.join(tempdir, 'model.ckpt')
      saver = tf.train.Saver(sharded=False, write_version=tf.train.SaverDef.V1)
      saver.save(sess, checkpoint_filename, meta_graph_suffix='meta',
                 write_meta_graph=True)
      metagraph_filename = checkpoint_filename + '.meta'
      bundle = generator_pb2.GeneratorBundle()
        
      details = generator_pb2.GeneratorDetails(
                id='performance_with_dynamics',
                description='Performance RNN with dynamics (compact input)')
      bundle.generator_details.CopyFrom(details)

      with tf.gfile.Open(checkpoint_filename, 'rb') as f:
        bundle.checkpoint_file.append(f.read())
      with tf.gfile.Open(metagraph_filename, 'rb') as f:
        bundle.metagraph_file = f.read()

      tf.logging.info('Writing to: ' + bundle_file)
      with tf.gfile.Open(bundle_file, 'wb') as f:
        f.write(bundle.SerializeToString())   

    finally:
      if tempdir is not None:
        tf.gfile.DeleteRecursively(tempdir)
        


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
