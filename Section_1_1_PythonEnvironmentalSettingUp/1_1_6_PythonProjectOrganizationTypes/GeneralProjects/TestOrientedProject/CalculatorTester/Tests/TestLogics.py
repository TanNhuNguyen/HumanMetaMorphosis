# Supporting libraries
import unittest; # This is the library for testing
import os;
import sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); # Add the project folder to the system path so that the system can see the Apps folder
from Apps import add, divide; # The Apps directory is upper from the TestLogics.py file

# Define the class for testing the functions in the apps
class TestCalculatorLogics(unittest.TestCase):
    # Testing adding the two positive numbers
    def test_addingPositiveNumbers(self):
        self.assertEqual(add(2, 3), 5);
    # Testing adding the two negative numbers
    def test_addingNegativeNumbers(self):
        self.assertEqual(add(-2, -3), -5);
    # Testing dividing the normal cases
    def test_dividingNormalCase(self):
        self.assertEqual(divide(10, 2), 5);
    # Testing dividing with the zero case. In this case, the testing is not equal but the with condition
    def test_dividingRaiseZeroDivision(self):
        with self.assertRaises(ZeroDivisionError):
            divide(5, 0);

# Main function for starting the tests
if __name__ == '__main__':
    unittest.main()
