# -*- coding: utf-8 -*-

from tensorflow.python.util import nest as tf_nest
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
import tensorflow as tf
import magenta as mg
import pandas as pd
import numpy as np
from magenta.models.performance_rnn.performance_model import PerformanceRnnConfig

class BaseModel:

    def __init__(self, config, mode, examples_path=None):
        self.build_graph_fn = self.get_build_graph_fn(config, mode, examples_path)

    def get_build_graph_fn(self, config, mode, examples_path=None):
        pass

    def train(self, logdir='log/', save_checkpoint_secs=60, save_summaries_steps=10, steps=1000, log_steps=10):
        with tf.Graph().as_default():
          tf.logging.info('Building graph.')
          self.build_graph_fn()

          global_step = tf.get_collection('global_step')[0]
          loss = tf.get_collection('loss')[0]
          train_op = tf.get_collection('train_op')[0]
          logging_dict = {
              'global_step': global_step,
              'loss': loss
          }
          for key in ['lstm_loss', 'composer_loss', 'composer_weighting']:
            if tf.get_collection(key):
                logging_dict[key] = tf.get_collection(key)[0]

          hooks = [
              tf.train.NanTensorHook(loss),
              tf.train.LoggingTensorHook(
                  logging_dict, every_n_iter=log_steps),
              tf.train.StopAtStepHook(steps)
          ]

          tf.logging.info('Starting training loop...')
          tf.contrib.training.train(train_op=train_op,
                                    logdir=logdir,
                                    hooks=hooks,
                                    save_checkpoint_secs=save_checkpoint_secs,
                                    save_summaries_steps=save_summaries_steps)
          tf.logging.info('Training loop complete.')

    def evaluate(self):
        with tf.Graph().as_default():
            self.get_build_graph_fn('eval')()

            # global_step = tf.train.get_or_create_global_step()
            global_step = tf.get_collection('global_step')[0]
            loss = tf.get_collection('loss')[0]
            perplexity = tf.get_collection('metrics/perplexity')[0]
            accuracy = tf.get_collection('metrics/accuracy')[0]
            eval_ops = tf.get_collection('eval_ops')

            logging_dict = {
                'Global Step': global_step,
                'Loss': loss,
                'Perplexity': perplexity,
                'Accuracy': accuracy
            }
            hooks = [
                EvalLoggingTensorHook(logging_dict, every_n_iter=num_batches),
                tf.contrib.training.StopAfterNEvalsHook(num_batches),
                tf.contrib.training.SummaryAtEndHook(eval_dir),
            ]

            tf.contrib.training.evaluate_repeatedly(
                train_dir,
                eval_ops=eval_ops,
                hooks=hooks,
                eval_interval_secs=60,
                timeout=timeout_secs)

class LSTMModel(BaseModel):

    def get_build_graph_fn(self, config, mode, examples_path=None):

        def build_graph():

            encoder_decoder = config.encoder_decoder
            input_size = encoder_decoder.input_size
            num_classes = encoder_decoder.num_classes
            default_event_label = encoder_decoder.default_event_label

            batch_size = config.batch_size
            label_shape = []
            learning_rate = config.learning_rate
            inputs, labels, lengths, composers = None, None, None, None
            if mode == 'train' or mode == 'eval':
                if isinstance(encoder_decoder, mg.music.OneHotEventSequenceMetaDataEncoderDecoder):
                    inputs, labels, lengths, composers = mg.common.get_padded_batch_metadata(
                        examples_path, batch_size, input_size,
                        label_shape=label_shape, shuffle=mode == 'train', 
                        composer_shape=config.label_classifier_units, num_enqueuing_threads=config.threads)
                    inputs = tf.debugging.check_numerics(inputs, "Inputs invalid")
                    labels = tf.cast(tf.debugging.check_numerics(tf.cast(labels, tf.float32), "Labels invalid"), tf.int64)
                    lengths = tf.cast(tf.debugging.check_numerics(tf.cast(lengths, tf.float32), "Lengths invalid"), tf.int32)
                    composers = tf.cast(tf.debugging.check_numerics(tf.cast(composers, tf.float32), "Composers invalid"), tf.int64)
                else:
                    inputs, labels, lengths = mg.common.get_padded_batch(
                            examples_path, batch_size, input_size,
                            label_shape=label_shape, shuffle=mode == 'train')
                    # assert not tf.debugging.is_nan(inputs)
                    # assert not tf.debugging.is_nan(labels)
                    # assert not tf.debugging.is_nan(lengths)
            else:
              inputs = tf.placeholder(tf.float32, [batch_size, None,
                                                   input_size])
            config.dropout = 1.0 if mode == 'generate' else config.dropout

            outputs, initial_state, final_state = None, None, None
            if config.gpu:
                tf.logging.info("Using CudNN")
                outputs, initial_state, final_state = get_cudnn(
                    inputs, config.rnn_layers, config.dropout, batch_size, mode)
            else:
                cell = get_deep_lstm(config.rnn_layers, config.dropout)
                initial_state = cell.zero_state(batch_size, tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(
                    cell, inputs, sequence_length=lengths, initial_state=initial_state,
                    swap_memory=True)

            outputs_flat = mg.common.flatten_maybe_padded_sequences(
                    outputs, lengths)

            num_logits = num_classes
            
            logits_flat = tf.contrib.layers.linear(outputs_flat, num_logits)

            if mode == 'train' or mode == 'eval':
              labels_flat = mg.common.flatten_maybe_padded_sequences(
                      labels, lengths)
              softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_flat, logits=logits_flat)
              softmax_cross_entropy = tf.debugging.check_numerics(softmax_cross_entropy, "softmax_cross_entropy invalid")

              loss = None
              global_step = tf.Variable(-1, trainable=False)
              # Predict our composer to enforce structure on embeddings
              if config.label_classifier_weight:

                tf.logging.info('****Building classifier graph.')

                composer_logits = tf.layers.dense(final_state[-1].h, config.label_classifier_units)
                composer_softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=composers, logits=composer_logits)

                composer_logits = tf.debugging.check_numerics(composer_logits, "composer_logits invalid")
                composer_loss = tf.reduce_mean(composer_softmax_cross_entropy)

                lstm_loss = tf.reduce_mean(softmax_cross_entropy)

                # tf.add_to_collection('composers', composers)
                tf.add_to_collection('composer_loss', composer_loss)
                tf.add_to_collection('lstm_loss', lstm_loss)
                tf.summary.scalar('composer_loss', composer_loss)
                tf.summary.scalar('lstm_loss', lstm_loss)

                decay_steps = config.decay_steps
                classifier_weight =  config.label_classifier_weight - tf.train.polynomial_decay(
                                            config.label_classifier_weight, global_step,
                                            decay_steps, 0.0,
                                            power=0.2)
                composer_loss = classifier_weight * composer_loss
                lstm_loss = (1 - classifier_weight) * lstm_loss

                tf.add_to_collection('composer_weighting', classifier_weight)
                tf.summary.scalar('composer_weight', classifier_weight)
                composer_loss = tf.maximum(tf.Variable(1e-07), composer_loss)
                
                composer_loss = tf.debugging.check_numerics(composer_loss, "composer_loss invalid")
                lstm_loss = tf.debugging.check_numerics(lstm_loss, "lstm_loss invalid")

                loss = tf.add(lstm_loss, composer_loss)
                loss = tf.debugging.check_numerics(loss, "loss invalid")

                tf.add_to_collection('loss', loss)
                tf.summary.scalar('loss', loss)

              else:
                loss = tf.reduce_mean(softmax_cross_entropy)
                tf.add_to_collection('loss', loss)
                tf.summary.scalar('loss', loss)

              optimizer = config.optimizer(learning_rate=learning_rate, momentum=config.momentum)
              train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer, global_step=global_step, clip_gradient_norm=config.norm)

              tf.add_to_collection('global_step', global_step)
              tf.add_to_collection('train_op', train_op)
              tf.add_to_collection('optimizer', optimizer)
            elif mode == 'generate':
              temperature = tf.placeholder(tf.float32, [])
              softmax_flat = tf.nn.softmax(
                    tf.div(logits_flat, tf.fill([num_classes], temperature)))
              softmax = tf.reshape(
                    softmax_flat, [batch_size, -1, num_classes])

              tf.add_to_collection('inputs', inputs)
              tf.add_to_collection('temperature', temperature)
              tf.add_to_collection('softmax', softmax)

              for state in tf_nest.flatten(initial_state):
                tf.add_to_collection('initial_state', state)
              for state in tf_nest.flatten(final_state):
                tf.add_to_collection('final_state', state)
        return build_graph

class LSTMAE(BaseModel):

    def get_build_graph_fn(self, config, mode, examples_path):

        def build_graph():

            encoder_decoder = config.encoder_decoder
            input_size = encoder_decoder.input_size
            num_classes = encoder_decoder.num_classes
            default_event_label = encoder_decoder.default_event_label

            batch_size = config.batch_size
            label_shape = []
            learning_rate = config.learning_rate
            inputs, labels, lengths, composers = None, None, None, None
            if mode == 'train' or mode == 'eval':
                if isinstance(encoder_decoder, mg.music.OneHotEventSequenceMetaDataEncoderDecoder):
                    inputs, labels, lengths, composers = mg.common.get_padded_batch_metadata(
                        examples_path, batch_size, input_size,
                        label_shape=label_shape, shuffle=mode == 'train', 
                        composer_shape=config.label_classifier_units)
                    assert not tf.debugging.is_nan(inputs)
                    assert not tf.debugging.is_nan(labels)
                    assert not tf.debugging.is_nan(lengths)
                    assert not tf.debugging.is_nan(composers)
                else:
                    inputs, labels, lengths = mg.common.get_padded_batch(
                            examples_path, batch_size, input_size,
                            label_shape=label_shape, shuffle=mode == 'train')
                    assert not tf.debugging.is_nan(inputs)
                    assert not tf.debugging.is_nan(labels)
                    assert not tf.debugging.is_nan(lengths)
            else:
              inputs = tf.placeholder(tf.float32, [batch_size, None,
                                                   input_size])
            config.dropout = 1.0 if mode == 'generate' else config.dropout

            encoder_cell = get_deep_lstm(config.rnn_layers, config.dropout)
            initial_state = encoder_cell.zero_state(batch_size, tf.float32)
            outputs_enc, final_state_enc = tf.nn.dynamic_rnn(
                encoder_cell, inputs, sequence_length=lengths, initial_state=initial_state,
                swap_memory=True)

            #Perhaps use a dense layer here?
            z = tf.contrib.layers.linear(outputs_enc[-1], config.z_size)
        
            decoder_cell = get_deep_lstm(config.rnn_layers, config.dropout)
            decoder_initial_state = final_state_enc[-1].h
            if isinstance(decoder_cell.state_size, tuple):
                decoder_initial_state = tuple([final_state_enc[-1].h for s in decoder_cell.state_size])

            outputs, final_state = tf.nn.dynamic_rnn(
                decoder_cell, inputs, sequence_length=lengths, initial_state=decoder_initial_state,
                swap_memory=True)

            outputs_flat = mg.common.flatten_maybe_padded_sequences(
                    outputs, lengths)
            num_logits = num_classes
            logits_flat = tf.contrib.layers.linear(outputs_flat, num_logits)

            if mode == 'train' or mode == 'eval':
              labels_flat = mg.common.flatten_maybe_padded_sequences(
                      labels, lengths)
              softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_flat, logits=logits_flat)

              loss = None
              global_step = tf.Variable(-1, trainable=False)
              # Predict our composer to enforce structure on embeddings
              if config.label_classifier_weight:

                tf.logging.info('****Building classifier graph.')

                composer_logits = tf.layers.dense(final_state[-1].h, config.label_classifier_units)
                composer_softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=composers, logits=composer_logits)
                composer_loss = tf.reduce_mean(composer_softmax_cross_entropy)

                lstm_loss = tf.reduce_mean(softmax_cross_entropy)

                decay_steps = config.decay_steps
                classifier_weight =  config.label_classifier_weight - tf.train.polynomial_decay(
                                            config.label_classifier_weight, global_step,
                                            decay_steps, 0.0,
                                            power=0.2)
                composer_loss = classifier_weight * composer_loss
                lstm_loss = (1 - classifier_weight) * lstm_loss

                composer_loss = tf.maximum(tf.Variable(0.0), composer_loss)
                
                loss = tf.add(lstm_loss, composer_loss)

                tf.add_to_collection('loss', loss)
                tf.add_to_collection('composer_loss', composer_loss)
                tf.add_to_collection('lstm_loss', lstm_loss)
                tf.summary.scalar('composer_loss', composer_loss)
                tf.summary.scalar('lstm_loss', lstm_loss)
                tf.summary.scalar('loss', loss)
              else:
                loss = tf.reduce_mean(softmax_cross_entropy)
                tf.add_to_collection('loss', loss)
                tf.summary.scalar('loss')

              optimizer = config.optimizer(learning_rate=learning_rate)
              train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer, global_step=global_step)

              tf.add_to_collection('global_step', global_step)
              tf.add_to_collection('train_op', train_op)
              tf.add_to_collection('optimizer', optimizer)
            elif mode == 'generate':
              temperature = tf.placeholder(tf.float32, [])
              softmax_flat = tf.nn.softmax(
                    tf.div(logits_flat, tf.fill([num_classes], temperature)))
              softmax = tf.reshape(
                    softmax_flat, [batch_size, -1, num_classes])

              tf.add_to_collection('inputs', inputs)
              tf.add_to_collection('temperature', temperature)
              tf.add_to_collection('softmax', softmax)

              for state in tf_nest.flatten(initial_state):
                tf.add_to_collection('initial_state', state)
              for state in tf_nest.flatten(final_state):
                tf.add_to_collection('final_state', state)
        return build_graph

class BaseConfig():
    def __init__(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate

class LSTMConfig(BaseConfig):
    def __init__(self, encoder_decoder, optimizer=tf.train.RMSPropOptimizer, learning_rate=0.01,
        rnn_layers=[512, 512], dropout=0.7, label_classifier_weight=None, 
        label_classifier_units=None, label_classifier_dict=None, decay_steps=2000, gpu=False, layers=None, batch_size=None, threads=None,
        momentum=0.0, norm=2):
        if layers:
            self.rnn_layers = [512 for x in range(layers)]
        else:
            self.rnn_layers = rnn_layers
        self.encoder_decoder = encoder_decoder
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.label_classifier_weight = label_classifier_weight
        self.label_classifier_units = label_classifier_units
        self.decay_steps = decay_steps
        self.gpu = gpu
        self.batch_size=batch_size
        self.threads = threads
        self.momentum = momentum
        self.norm = norm

        super(LSTMConfig, self).__init__(optimizer, learning_rate)

class LSTMAEConfig(LSTMConfig):
    def __init__(self, z_size=256):
        self.z_size = 256

        super(LSTMAEConfig, self).__init__()

def state_tuples_to_cudnn_lstm_state(lstm_state_tuples):
  """Convert LSTMStateTuples to CudnnLSTM format."""
  h = tf.stack([s.h for s in lstm_state_tuples])
  c = tf.stack([s.c for s in lstm_state_tuples])
  return (h, c)

def cudnn_lstm_state_to_state_tuples(cudnn_lstm_state):
  """Convert CudnnLSTM format to LSTMStateTuples."""
  h, c = cudnn_lstm_state
  return tuple(
      tf.contrib.rnn.LSTMStateTuple(h=h_i, c=c_i)
      for h_i, c_i in zip(tf.unstack(h), tf.unstack(c)))

def get_cudnn(inputs, layers, dropout, batch_size, mode):
    cudnn_inputs = tf.transpose(inputs, [1, 0, 2])
    initial_state = tuple(
        tf.contrib.rnn.LSTMStateTuple(
            h=tf.zeros([batch_size, num_units], dtype=tf.float32),
            c=tf.zeros([batch_size, num_units], dtype=tf.float32))
        for num_units in layers)
    outputs, final_state = None, None
    if mode != 'generate': 
        # We can make a single call to CudnnLSTM since all layers are the same
        # size and we aren't using residual connections.
        cudnn_initial_state = state_tuples_to_cudnn_lstm_state(initial_state)
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=len(layers),
            num_units=layers[0],
            direction='unidirectional',
            dropout=1.0 - dropout)
        outputs, final_state = cell(
          cudnn_inputs, initial_state=cudnn_initial_state,
          training=mode == 'train')
        final_state = cudnn_lstm_state_to_state_tuples(final_state)
    else:
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
              for num_units in layers])

        outputs, final_state = tf.nn.dynamic_rnn(
            cell, cudnn_inputs, initial_state=initial_state, time_major=True,
            scope='cudnn_lstm/rnn')

    outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs, tuple(initial_state), tuple(final_state)

def get_deep_lstm(layers, dropout):
  cells = [tf.nn.rnn_cell.LSTMCell(size, reuse=tf.AUTO_REUSE) for size in layers]
  cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout) for cell in cells]
  return tf.contrib.rnn.MultiRNNCell(cells)

def get_composers(csv):
    df = pd.read_csv(csv)
    composers = df[df.midi_filename.str.contains('2017')].groupby('canonical_composer').count().index.values
    composers = [composer.split(' ')[1] for composer in composers]
    composers = np.sort(np.array(df.groupby('canonical_composer').count().index.values, dtype=np.str))
    return composers, len(composers)

def get_composers_constrained():
    composers = [u'Frédéric Chopin',u'Johann Sebastian Bach', u'Claude Debussy', u'Ludwig van Beethoven']
    composers = [x.encode('utf-8') for x in composers]
    return composers, 4

def get_config_with_csv(composers):
    return PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics',
            description='Conditional Performance RNN with dynamics'),
        mg.music.OneHotEventSequenceMetaDataEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32), composers),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32)

default_configs = {
    'performance': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance',
            description='Performance RNN'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding()),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001)),

    'performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics',
            description='Performance RNN with dynamics'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),

    'performance_with_dynamics_compact': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics',
            description='Performance RNN with dynamics (compact input)'),
        mg.music.OneHotIndexEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),

    'performance_with_dynamics_and_modulo_encoding': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics_and_modulo_encoding',
            description='Performance RNN with dynamics and modulo encoding'),
        mg.music.ModuloPerformanceEventSequenceEncoderDecoder(
            num_velocity_bins=32),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),

    'performance_with_dynamics_and_note_encoding': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics_and_note_encoding',
            description='Performance RNN with dynamics and note encoding'),
        mg.music.NotePerformanceEventSequenceEncoderDecoder(
            num_velocity_bins=32),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        note_performance=True),

    'density_conditioned_performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='density_conditioned_performance_with_dynamics',
            description='Note-density-conditioned Performance RNN + dynamics'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            mg.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        ]),

    'pitch_conditioned_performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='pitch_conditioned_performance_with_dynamics',
            description='Pitch-histogram-conditioned Performance RNN'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            mg.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ]),

    'multiconditioned_performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='multiconditioned_performance_with_dynamics',
            description='Density- and pitch-conditioned Performance RNN'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            mg.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            mg.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ]),

    'optional_multiconditioned_performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='optional_multiconditioned_performance_with_dynamics',
            description='Optionally multiconditioned Performance RNN'),
        mg.music.OneHotEventSequenceEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32,
        control_signals=[
            mg.music.NoteDensityPerformanceControlSignal(
                window_size_seconds=3.0,
                density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            mg.music.PitchHistogramPerformanceControlSignal(
                window_size_seconds=5.0)
        ],
        optional_conditioning=True),

    'conditional_performance_with_dynamics': PerformanceRnnConfig(
        mg.protobuf.generator_pb2.GeneratorDetails(
            id='performance_with_dynamics',
            description='Conditional Performance RNN with dynamics'),
        mg.music.OneHotEventSequenceMetaDataEncoderDecoder(
            mg.music.PerformanceOneHotEncoding(
                num_velocity_bins=32)),
        tf.contrib.training.HParams(
            batch_size=64,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=1.0,
            clip_norm=3,
            learning_rate=0.001),
        num_velocity_bins=32),
}
