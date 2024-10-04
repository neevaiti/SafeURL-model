from django.test import TestCase

class AdvancedSanityTests(TestCase):
    def test_tuple_unpacking(self):
        a, b, c = (1, 2, 3)
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)

    def test_lambda_functions(self):
        square = lambda x: x * x
        self.assertEqual(square(5), 25)

    def test_map_function(self):
        numbers = [1, 2, 3, 4]
        squared = list(map(lambda x: x ** 2, numbers))
        self.assertListEqual(squared, [1, 4, 9, 16])

    def test_filter_function(self):
        numbers = range(10)
        even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
        self.assertListEqual(even_numbers, [0, 2, 4, 6, 8])

    def test_zip_function(self):
        names = ['Alice', 'Bob', 'Charlie']
        scores = [85, 90, 95]
        combined = list(zip(names, scores))
        self.assertEqual(combined[1], ('Bob', 90))

    def test_enumerate_function(self):
        items = ['apple', 'banana', 'cherry']
        indexed_items = list(enumerate(items))
        self.assertEqual(indexed_items[2], (2, 'cherry'))

    def test_sorted_function(self):
        unsorted_list = [3, 1, 4, 1, 5, 9]
        sorted_list = sorted(unsorted_list)
        self.assertListEqual(sorted_list, [1, 1, 3, 4, 5, 9])

    def test_all_function(self):
        conditions = [True, True, True]
        self.assertTrue(all(conditions))

    def test_any_function(self):
        conditions = [False, False, True]
        self.assertTrue(any(conditions))

    def test_dict_update(self):
        original = {'a': 1, 'b': 2}
        update = {'b': 3, 'c': 4}
        original.update(update)
        self.assertDictEqual(original, {'a': 1, 'b': 3, 'c': 4})

    def test_list_extend(self):
        list_a = [1, 2, 3]
        list_b = [4, 5, 6]
        list_a.extend(list_b)
        self.assertListEqual(list_a, [1, 2, 3, 4, 5, 6])

    def test_string_formatting(self):
        name = "World"
        greeting = f"Hello, {name}!"
        self.assertEqual(greeting, "Hello, World!")

    def test_list_slicing(self):
        numbers = [0, 1, 2, 3, 4, 5]
        slice = numbers[1:4]
        self.assertListEqual(slice, [1, 2, 3])

    def test_set_difference(self):
        set_a = {1, 2, 3, 4}
        set_b = {3, 4, 5, 6}
        difference = set_a - set_b
        self.assertSetEqual(difference, {1, 2})

    def test_frozenset(self):
        frozen = frozenset([1, 2, 3])
        self.assertIn(2, frozen)

    def test_complex_numbers(self):
        complex_num = 1 + 2j
        self.assertEqual(complex_num.real, 1)
        self.assertEqual(complex_num.imag, 2)

    def test_list_comprehension_with_condition(self):
        numbers = [x for x in range(10) if x % 3 == 0]
        self.assertListEqual(numbers, [0, 3, 6, 9])

    def test_dict_comprehension_with_condition(self):
        keys = ['a', 'b', 'c', 'd']
        values = [1, 2, 3, 4]
        dictionary = {k: v for k, v in zip(keys, values) if v % 2 == 0}
        self.assertDictEqual(dictionary, {'b': 2, 'd': 4})

    def test_nested_loops(self):
        result = [(x, y) for x in range(3) for y in range(3)]
        self.assertIn((2, 2), result)