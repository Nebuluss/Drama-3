import unittest
from unittest.mock import patch

import numpy as np

from run_drama3_full import build_speaker_prompt


class SpeakerDialogueTests(unittest.TestCase):
    def test_each_turn_uses_only_its_own_references(self):
        voices = [
            {'speaker_id': 0, 'text': 'First voice.', 'codes': np.full((10, 4), 11, dtype=np.int16)},
            {'speaker_id': 1, 'text': 'Second voice.', 'codes': np.full((10, 5), 22, dtype=np.int16)},
            {'speaker_id': 1, 'text': 'Another clip.', 'codes': np.full((10, 3), 33, dtype=np.int16)},
        ]
        with patch('run_drama3_full.build_drama3_prompt') as builder:
            for speaker, expected_values in [(0, [11]), (1, [22, 33]), (0, [11]), (1, [22, 33])]:
                text = f'<|speaker:{speaker}|> A new reply.'
                build_speaker_prompt(text, 'tokenizer', 'config', voices)
                args, kwargs = builder.call_args
                self.assertEqual(args[0], text)
                self.assertTrue(all(t.startswith(f'<|speaker:{speaker}|>') for t in args[3]))
                self.assertEqual([t[0, 0].item() for t in args[4]], expected_values)
                self.assertEqual(kwargs, {})  # No previous character audio in the prompt.

    def test_unreferenced_speaker_still_allows_tts(self):
        with patch('run_drama3_full.build_drama3_prompt') as builder:
            build_speaker_prompt('<|speaker:2|> Hello.', 'tokenizer', 'config', [])
            self.assertEqual(builder.call_args.args[3:], ([], []))


if __name__ == '__main__':
    unittest.main()
