import unittest
import sets


class TokenBoundaryTests(unittest.TestCase):
    def test_whitespace_and_long_expression(self):
        for tail in ('', ' ', '\t\r\n', '\u3000'):
            self.assertEqual(sets.tokens(' A & (~B | C_1)' + tail),
                             ['A', '&', '(', '~', 'B', '|', 'C_1', ')'])
            self.assertEqual(sets.tokens(tail), [])
        self.assertEqual(sets.tokens('A &' * 10000 + ' B   '),
                         ['A', '&'] * 10000 + ['B'])

    def test_invalid_suffix_is_reported_without_offset_change(self):
        with self.assertRaisesRegex(sets.SetError, "式を読めない") as caught:
            sets.tokens('A & $bad  ')
        self.assertIn("' $bad  '", str(caught.exception))
