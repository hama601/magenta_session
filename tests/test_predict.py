import importlib
import sys
import types
import unittest


class _FakeGenerateSections:
    def __init__(self):
        self.sections = []

    def add(self, start_time, end_time):
        self.sections.append({"start_time": start_time, "end_time": end_time})


class _FakeGeneratorOptions:
    def __init__(self):
        self.generate_sections = _FakeGenerateSections()


class _FakeGenerator:
    def __init__(self):
        self.last_primer = None
        self.last_options = None

    def generate(self, primer_sequence, generator_options):
        self.last_primer = primer_sequence
        self.last_options = generator_options
        return "generated-sequence"


class _FakeMelodyRnnModelModule(types.ModuleType):
    class MelodyRnnModel:
        def __init__(self, config):
            self.config = config


class _FakeTempFile:
    def __init__(self):
        self.name = "fake_output.mid"

    def seek(self, position):
        self.position = position


class _FakeMidiData:
    def __init__(self, sequence, tempo):
        self._sequence = sequence
        self._tempo = tempo

    def estimate_tempo(self):
        return self._tempo


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.fake_generator = _FakeGenerator()
        self.saved_modules = sys.modules.copy()

        tf_module = types.ModuleType("tensorflow")
        tf_module.gfile = types.SimpleNamespace(Exists=lambda path: False, Open=lambda path: open(path))

        midi_io_module = types.ModuleType("magenta.music.midi_io")
        midi_io_module.midi_to_sequence_proto = lambda midi_data: midi_data._sequence
        midi_io_module.sequence_proto_to_midi_file = lambda generated_sequence, path: None

        music_module = types.ModuleType("magenta.music")
        music_module.read_bundle_file = lambda path: "fake-bundle"
        music_module.midi_io = midi_io_module

        magenta_module = types.ModuleType("magenta")
        magenta_module.music = music_module

        flags_module = types.ModuleType("magenta.models.melody_rnn.melody_rnn_config_flags")
        flags_module.FLAGS = types.SimpleNamespace(config=None, hparams=None)
        flags_module.config_from_flags = lambda: types.SimpleNamespace(details="flag-details")

        model_module = _FakeMelodyRnnModelModule("magenta.models.melody_rnn.melody_rnn_model")
        model_module.default_configs = {"attention_rnn": types.SimpleNamespace(details="fake-details")}

        seq_gen_module = types.ModuleType("magenta.models.melody_rnn.melody_rnn_sequence_generator")
        seq_gen_module.MelodyRnnSequenceGenerator = lambda **kwargs: self.fake_generator

        melody_rnn_module = types.ModuleType("magenta.models.melody_rnn")
        melody_rnn_module.melody_rnn_config_flags = flags_module
        melody_rnn_module.melody_rnn_model = model_module
        melody_rnn_module.melody_rnn_sequence_generator = seq_gen_module

        models_module = types.ModuleType("magenta.models")
        models_module.melody_rnn = melody_rnn_module

        generator_pb2_module = types.ModuleType("magenta.protobuf.generator_pb2")
        generator_pb2_module.GeneratorOptions = _FakeGeneratorOptions

        music_pb2_module = types.ModuleType("magenta.protobuf.music_pb2")
        protobuf_module = types.ModuleType("magenta.protobuf")
        protobuf_module.generator_pb2 = generator_pb2_module
        protobuf_module.music_pb2 = music_pb2_module

        pretty_midi_module = types.ModuleType("pretty_midi")
        tempfile_module = types.ModuleType("tempfile")
        tempfile_module.NamedTemporaryFile = lambda: _FakeTempFile()

        sys.modules.update({
            "tensorflow": tf_module,
            "magenta": magenta_module,
            "magenta.music": music_module,
            "magenta.music.midi_io": midi_io_module,
            "magenta.models": models_module,
            "magenta.models.melody_rnn": melody_rnn_module,
            "magenta.models.melody_rnn.melody_rnn_config_flags": flags_module,
            "magenta.models.melody_rnn.melody_rnn_model": model_module,
            "magenta.models.melody_rnn.melody_rnn_sequence_generator": seq_gen_module,
            "magenta.protobuf": protobuf_module,
            "magenta.protobuf.generator_pb2": generator_pb2_module,
            "magenta.protobuf.music_pb2": music_pb2_module,
            "pretty_midi": pretty_midi_module,
            "tempfile": tempfile_module,
        })

        sys.modules.pop("server.predict", None)
        self.predict = importlib.import_module("server.predict")

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.saved_modules)

    def _make_sequence(self, note_count):
        notes = [types.SimpleNamespace(end_time=i + 1) for i in range(note_count)]
        tempos = [types.SimpleNamespace(qpm=0)]
        return types.SimpleNamespace(notes=notes, tempos=tempos)

    def test_generate_midi_halves_too_fast_tempo_with_enough_notes(self):
        sequence = self._make_sequence(note_count=5)
        midi_data = _FakeMidiData(sequence=sequence, tempo=300)

        self.predict.generate_midi(midi_data, total_seconds=10)

        self.assertEqual(sequence.tempos[0].qpm, 150)
        section = self.fake_generator.last_options.generate_sections.sections[0]
        self.assertAlmostEqual(section["start_time"], 5 + (60.0 / 150 / 4))
        self.assertEqual(section["end_time"], 10)

    def test_generate_midi_uses_default_qpm_when_note_count_is_three_or_less(self):
        sequence = self._make_sequence(note_count=3)
        midi_data = _FakeMidiData(sequence=sequence, tempo=300)

        self.predict.generate_midi(midi_data, total_seconds=8)

        self.assertEqual(sequence.tempos[0].qpm, 120)
        section = self.fake_generator.last_options.generate_sections.sections[0]
        self.assertAlmostEqual(section["start_time"], 3 + (60.0 / 120 / 4))
        self.assertEqual(section["end_time"], 8)

    def test_steps_to_seconds(self):
        self.assertAlmostEqual(self.predict._steps_to_seconds(8, 120), 1.0)
        self.assertAlmostEqual(self.predict._steps_to_seconds(4, 60), 1.0)


if __name__ == "__main__":
    unittest.main()
