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
"""Train and evaluate a performance RNN model."""

import os

import magenta
import models
from models import LSTMModel, LSTMAE
from models import LSTMConfig
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train
import tensorflow as tf
import pandas as pd
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
  'csv', None,
  'CSV containing metadata.')
tf.app.flags.DEFINE_float(
  'composer_weighting', None,
  'Weighting for composer loss')
tf.app.flags.DEFINE_integer(
  'log_steps', 10,
  'When to output log')
tf.app.flags.DEFINE_integer(
  'checkpoint_secs', 60,
  'When to save checkpoint')
tf.app.flags.DEFINE_integer(
  'decay_steps', 2000,
  'When to introduce composer loss')
tf.app.flags.DEFINE_boolean(
  'gpu', False,
  'Whether to use CudNN')
tf.app.flags.DEFINE_string(
  'optimizer', 'AdamOptimizer',
  'Optimizer: RMSPropOptimizer, AdamOptimizer')
tf.app.flags.DEFINE_string('run_dir', '/tmp/performance_rnn/logdir/run1',
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation. Separate subdirectories for training '
                           'events and eval events will be created within '
                           '`run_dir`. Multiple runs can be stored within the '
                           'parent directory of `run_dir`. Point TensorBoard '
                           'to the parent directory of `run_dir` to see all '
                           'your runs.')
tf.app.flags.DEFINE_string('config', 'conditional_performance_with_dynamics', 'The config to use')
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training or '
                           'evaluation.')
tf.app.flags.DEFINE_integer('num_training_steps', 100,
                            'The the number of global training steps your '
                            'model should take before exiting training. '
                            'Leave as 0 to run until terminated manually.')
tf.app.flags.DEFINE_integer('num_eval_examples', 0,
                            'The number of evaluation examples your model '
                            'should process for each evaluation step.'
                            'Leave as 0 to use the entire evaluation set.')
tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            'A summary statement will be logged every '
                            '`summary_frequency` steps during training or '
                            'every `summary_frequency` seconds during '
                            'evaluation.')
tf.app.flags.DEFINE_integer('num_checkpoints', 3,
                            'The number of most recent checkpoints to keep in '
                            'the training directory. Keeps all if 0.')
tf.app.flags.DEFINE_boolean('eval', False,
                            'If True, this process only evaluates the model '
                            'and does not update weights.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if not FLAGS.run_dir:
    tf.logging.fatal('--run_dir required')
    return
  if not FLAGS.sequence_example_file:
    tf.logging.fatal('--sequence_example_file required')
    return
  
  sequence_example_file_paths = tf.gfile.Glob(
      os.path.expanduser(FLAGS.sequence_example_file))
  run_dir = os.path.expanduser(FLAGS.run_dir)

  mode = 'eval' if FLAGS.eval else 'train'
  
  composers, units = None, None
  if FLAGS.csv:
    csv = os.path.expanduser(FLAGS.csv)
    tf.logging.info("CSV file provided, populating metadata")
    composers, units = models.get_composers(csv)
  
  data_config = models.get_config_with_csv(composers)
  data_config.hparams.parse(FLAGS.hparams)

  steps = FLAGS.num_training_steps
  encoder_decoder = data_config.encoder_decoder
  label_classifier_weight = FLAGS.composer_weighting
  decay_steps = FLAGS.decay_steps

  optimizers = {'AdamOptimizer': tf.train.AdamOptimizer,
                'RMSPropOptimizer': tf.train.RMSPropOptimizer}
  optimizer = optimizers[FLAGS.optimizer]
  config = LSTMConfig(
    encoder_decoder=encoder_decoder,
    label_classifier_units=units,
    label_classifier_weight=label_classifier_weight,
    decay_steps=decay_steps,
    gpu=FLAGS.gpu,
    optimizer=optimizer)
  model = LSTMModel(config, mode, sequence_example_file_paths)
  
  if FLAGS.eval:
    eval_dir = os.path.join(run_dir, 'eval')
    tf.gfile.MakeDirs(eval_dir)
    tf.logging.info('Eval dir: %s', eval_dir)
    num_batches = (
        (FLAGS.num_eval_examples or
         magenta.common.count_records(sequence_example_file_paths)) //
        config.hparams.batch_size)
    events_rnn_train.run_eval(build_graph_fn, train_dir, eval_dir, num_batches)
  
  else:
    model.train(logdir=run_dir, steps=steps,
      save_summaries_steps=FLAGS.summary_frequency,
      log_steps=FLAGS.log_steps)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
