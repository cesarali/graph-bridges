import unittest

class AbstractTestCase(unittest.TestCase):

    @unittest.skip("Abstract test case")
    def test_abstract_method(self):
        pass

class ConcreteTestCase(AbstractTestCase):

    def test_concrete_method(self):
        self.assertEqual(2 + 2, 4)

if __name__ == "__main__":
    unittest.main()
