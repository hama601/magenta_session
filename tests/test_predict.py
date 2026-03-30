import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pathlib
import pytest

# Ensure the repository root is on the path so that ``server`` can be imported.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def _create_stubs():
    modules = {}
    magenta = types.ModuleType('magenta')
    music = types.ModuleType('magenta.music')
    midi_io = types.ModuleType('magenta.music.midi_io')
    midi_io.midi_to_sequence_proto = MagicMock()
    midi_io.sequence_proto_to_midi_file = MagicMock()
    music.midi_io = midi_io
    music.read_bundle_file = MagicMock()
    magenta.music = music

    models = types.ModuleType('magenta.models')
    melody_rnn_module = types.ModuleType('magenta.models.melody_rnn')
    melody_rnn_config_flags = types.ModuleType('melody_rnn_config_flags')
    melody_rnn_config_flags.FLAGS = types.SimpleNamespace()
    melody_rnn_model = types.ModuleType('melody_rnn_model')
    melody_rnn_model.default_configs = {
        'attention_rnn': types.SimpleNamespace(details=None)
    }
    melody_rnn_model.MelodyRnnModel = MagicMock()
    melody_rnn_sequence_generator = types.ModuleType('melody_rnn_sequence_generator')
    class DummyGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return MagicMock()
    melody_rnn_sequence_generator.MelodyRnnSequenceGenerator = DummyGenerator
    melody_rnn_module.melody_rnn_config_flags = melody_rnn_config_flags
    melody_rnn_module.melody_rnn_model = melody_rnn_model
    melody_rnn_module.melody_rnn_sequence_generator = melody_rnn_sequence_generator
    models.melody_rnn = melody_rnn_module
    magenta.models = models

    protobuf = types.ModuleType('magenta.protobuf')
    generator_pb2 = types.ModuleType('generator_pb2')
    class DummyGenerateSections(list):
        def add(self, start_time, end_time):
            self.append(types.SimpleNamespace(start_time=start_time, end_time=end_time))
    class DummyOptions:
        def __init__(self):
            self.generate_sections = DummyGenerateSections()
    generator_pb2.GeneratorOptions = DummyOptions
    music_pb2 = types.ModuleType('music_pb2')
    protobuf.generator_pb2 = generator_pb2
    protobuf.music_pb2 = music_pb2
    magenta.protobuf = protobuf

    modules.update({
        'magenta': magenta,
        'magenta.music': music,
        'magenta.music.midi_io': midi_io,
        'magenta.models': models,
        'magenta.models.melody_rnn': melody_rnn_module,
        'magenta.models.melody_rnn.melody_rnn_config_flags': melody_rnn_config_flags,
        'magenta.models.melody_rnn.melody_rnn_model': melody_rnn_model,
        'magenta.models.melody_rnn.melody_rnn_sequence_generator': melody_rnn_sequence_generator,
        'magenta.protobuf': protobuf,
        'magenta.protobuf.generator_pb2': generator_pb2,
        'magenta.protobuf.music_pb2': music_pb2,
    })
    tf = types.ModuleType('tensorflow')
    tf.gfile = types.SimpleNamespace(Exists=lambda path: False)
    modules['tensorflow'] = tf
    return modules


@pytest.fixture
def predict_module():
    modules = _create_stubs()
    with patch.dict(sys.modules, modules):
        if 'server.predict' in sys.modules:
            del sys.modules['server.predict']
        predict = importlib.import_module('server.predict')
        yield predict, modules


def test_generate_midi_high_tempo(predict_module):
    predict, modules = predict_module
    note = types.SimpleNamespace(end_time=0.5)
    sequence = types.SimpleNamespace(notes=[note] * 5, tempos=[types.SimpleNamespace(qpm=0)])
    modules['magenta.music.midi_io'].midi_to_sequence_proto.return_value = sequence
    with patch('pretty_midi.PrettyMIDI') as pm:
        midi = pm.return_value
        midi.estimate_tempo.return_value = 300
        predict.generate_midi(midi)
    assert sequence.tempos[0].qpm == 150


def test_generate_midi_default_qpm(predict_module):
    predict, modules = predict_module
    note = types.SimpleNamespace(end_time=0.5)
    sequence = types.SimpleNamespace(notes=[note] * 3, tempos=[types.SimpleNamespace(qpm=0)])
    modules['magenta.music.midi_io'].midi_to_sequence_proto.return_value = sequence
    with patch('pretty_midi.PrettyMIDI') as pm:
        midi = pm.return_value
        midi.estimate_tempo.return_value = 200
        predict.generate_midi(midi)
    assert sequence.tempos[0].qpm == 120


def test_steps_to_seconds(predict_module):
    predict, _ = predict_module
    assert predict._steps_to_seconds(1, 150) == pytest.approx(0.1)
