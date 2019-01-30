from tensorflow.python.util import nest as tf_nest
import tensorflow as tf
import magenta as mg
import pandas as pd
import numpy as np
from magenta.models.performance_rnn.performance_model import PerformanceRnnConfig

class BaseModel:

    def __init__(self, config, mode, examples_path):
        self.build_graph_fn = self.get_build_graph_fn(config, mode, examples_path)

    def get_build_graph_fn(self, config, mode, examples_path):
        pass

    def train(self, logdir='log/', save_checkpoint_secs=60, save_summaries_steps=60, steps=1000):
        with tf.Graph().as_default():
          self.build_graph_fn()

          global_step = tf.train.get_or_create_global_step()
          loss = tf.get_collection('loss')[0]
          optimizer = tf.get_collection('optimizer')[0].minimize(loss)
          train_op = tf.get_collection('train_op')[0]
          logging_dict = {
              'global_step': global_step,
              'loss': loss
          }
          hooks = [
              tf.train.NanTensorHook(loss),
              tf.train.LoggingTensorHook(
                  logging_dict, every_n_iter=100),
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
            global_step = tf.train.get_or_create_global_step()
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

    def get_build_graph_fn(self, config, mode, examples_path):

        def build_graph():

            encoder_decoder = config.encoder_decoder
            input_size = encoder_decoder.input_size
            num_classes = encoder_decoder.num_classes
            default_event_label = encoder_decoder.default_event_label

            batch_size = 64
            label_shape = []
            learning_rate = config.learning_rate
            inputs, labels, lengths = None, None, None
            if mode == 'train' or mode == 'eval':
                if isinstance(encoder_decoder, mg.music.OneHotEventSequenceMetaDataEncoderDecoder):
                    inputs, labels, lengths, composers = mg.common.get_padded_batch_metadata(
                        examples_path, batch_size, input_size,
                        label_shape=label_shape, shuffle=mode == 'train')
                else:
                    inputs, labels, lengths = mg.common.get_padded_batch(
                            examples_path, batch_size, input_size,
                            label_shape=label_shape, shuffle=mode == 'train')
            else:
              inputs = tf.placeholder(tf.float32, [batch_size, None,
                                                   input_size])
            config.dropouts = [1.0 if mode == 'generate' else dropout for dropout in config.dropouts]

            cell = get_deep_lstm(config.rnn_layers, config.dropouts)
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

              # Predict our composer to enforce structure on embeddings
              if config.label_classifier_weight:
                  composer_classifier = tf.layers.dense(logits_flat, config.label_classifier_units)

              loss = tf.scalar_mul(tf.Variable(1 - config.label_classifier_weight), tf.reduce_mean(softmax_cross_entropy))
              optimizer = config.optimizer(learning_rate=learning_rate)
              train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer)

              tf.summary.scalar('loss', loss)
              tf.add_to_collection('loss', loss)
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
        self.optimizer = optimizerlabel_classifier_units
        self.learning_rate = learning_rate

class LSTMConfig(BaseConfig):
    def __init__(self, encoder_decoder, optimizer=tf.train.RMSPropOptimizer, learning_rate=0.01,
        rnn_layers=[512,512], dropouts=[0.7, 0.7], label_classifier_weight=None, 
        label_classifier_units=None, label_classifier_dict=None):
        self.encoder_decoder = encoder_decoder
        self.rnn_layers = rnn_layers
        self.dropouts = dropouts
        self.label_classifier_weight = label_classifier_weight
        if label_classifier_weight and (not label_classifier_units or not label_classifier_dict):
            tf.logging.fatal('label_classifier_units and label_classifier_dict required if label_classifier_weight')
            return
        self.label_classifier_units = label_classifier_units
        self.label_classifier_dict = label_classifier_dict

        super(LSTMConfig, self).__init__(optimizer, learning_rate)

def get_deep_lstm(layers, dropouts):
  cells = [tf.nn.rnn_cell.LSTMCell(size, reuse=tf.AUTO_REUSE) for size in layers]
  cells = [tf.contrib.rnn.DropoutWrapper(cell, prob) for prob, cell in zip(dropouts, cells)]
  return tf.contrib.rnn.MultiRNNCell(cells)


def get_composers(csv):
    df = pd.read_csv(csv)
    composers = df[df.midi_filename.str.contains('2017')].groupby('canonical_composer').count().index.values
    composers = [composer.split(' ')[1] for composer in x]
    composers = np.sort(np.array(df.groupby('canonical_composer').count().index.values, dtype=np.str))
    return dict(zip(composers, range(len(composers)))), len(composers)


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
