import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import tensorflow as tf

# TensorFlow 2.x moved the `app` module under `compat.v1`. Use the
# compat alias so this script remains compatible with both TF1 and TF2.
tf = tf.compat.v1
import scripts.target as tgt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir and checkpoint_file, unless save_generator_bundle is True, in '
    'which case both this flag and either run_dir or checkpoint_file are '
    'required')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')

def _resolve_bundle_filename(bundle_file_flag, timestamp):
    """Return final bundle filename including the .mag extension."""
    base = bundle_file_flag if bundle_file_flag else "magenta_" + timestamp
    if not base.endswith(".mag"):
        base += ".mag"
    return base


def main(unused_argv):
    from magenta.models.melody_rnn import melody_rnn_config_flags
    from magenta.models.melody_rnn import melody_rnn_model
    from magenta.models.melody_rnn import melody_rnn_sequence_generator

    tf.logging.set_verbosity(FLAGS.log)
    train_dir = os.path.join(tgt.MODEL_DIR, "logdir/train")
    hparams_path = os.path.join(train_dir, "hparams")
    with tf.gfile.Open(hparams_path) as f:
        config, hparams = f.readline().split("\t")

    melody_rnn_config_flags.FLAGS.config = config
    melody_rnn_config_flags.FLAGS.hparams = hparams
    config = melody_rnn_config_flags.config_from_flags()

    generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
        model=melody_rnn_model.MelodyRnnModel(config),
        details=config.details,
        steps_per_quarter=config.steps_per_quarter,
        checkpoint=train_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_file = _resolve_bundle_filename(FLAGS.bundle_file, timestamp)
    bundle_path = os.path.join(tgt.MODEL_DIR, bundle_file)

    if FLAGS.bundle_description is None:
        tf.logging.warning('No bundle description provided.')
    tf.logging.info('Saving generator bundle to %s', bundle_path)
    generator.create_bundle_file(bundle_path, FLAGS.bundle_description)
    hparams_dest = os.path.join(
        tgt.MODEL_DIR, os.path.splitext(bundle_file)[0] + ".hparams")
    tf.gfile.Copy(hparams_path, hparams_dest, overwrite=True)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
