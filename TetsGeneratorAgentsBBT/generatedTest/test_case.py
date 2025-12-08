import unittest

class TestAddFunction(unittest.TestCase):
    def test_simple_addition(self):
        self.assertEqual(add(2, 3), 5)

    def test_large_numbers(self):
        self.assertEqual(add(1000, 2000), 3000)

    def test_negative_numbers(self):
        self.assertEqual(add(-2, -3), -5)

    def test_zero(self):
        self.assertEqual(add(0, 0), 0)

    def test_float_numbers(self):
        self.assertAlmostEqual(add(2.5, 3.7), 6.2)

    def test_non_numeric_input(self):
        with self.assertRaises(TypeError):
            add('a', 'b')

    def test_non_integer_input(self):
        with self.assertRaises(TypeError):
            add(2.5, 'b')

    def test_missing_argument(self):
        with self.assertRaises(TypeError):
            add(2)

    def test_invalid_argument_type(self):
        with self.assertRaises(TypeError):
            add([1, 2], 3)

def add(a, b):
    return a + b

if __name__ == '__main__':
    unittest.main()