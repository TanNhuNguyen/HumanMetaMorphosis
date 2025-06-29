#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os, sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));

import pytest;
from Core.Vitals import calculateBMI, interpretBMI;

#*********************************************************************************************************#
#********************************************* TESTING FUNCTIONS *****************************************#
#*********************************************************************************************************#
def test_calculateBMI():
    assert calculateBMI(70, 175) == 22.86;
def test_interpret_bmi():
    assert interpretBMI(17) == "Underweight";
    assert interpretBMI(22) == "Normal";
    assert interpretBMI(27) == "Overweight";
    assert interpretBMI(32) == "Obese";