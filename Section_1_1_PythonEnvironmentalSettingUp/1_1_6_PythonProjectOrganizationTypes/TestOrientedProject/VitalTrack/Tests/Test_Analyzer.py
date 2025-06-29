#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os, sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));

import pytest;
from Core.Analyzer import analyzeVitals;
from Core.Analyzer import analyzeRespiration;

#*********************************************************************************************************#
#********************************************* TESTING FUNCTIONS *****************************************#
#*********************************************************************************************************#
class DummyBody:
    def __init__(self, name, inHeightCm, inWeightKg):
        self.name = name;
        self.heightCm = inHeightCm;
        self.weightKg = inWeightKg;

@pytest.mark.parametrize("inWeight, inHeight, inExpectedBMI, inExpectedStatus", [
    (50, 170, 17.3, "Underweight"),
    (65, 170, 22.49, "Normal"),
    (80, 170, 27.68, "Overweight"),
    (95, 170, 32.87, "Obese"),
])
def test_analyzeVitals(inWeight, inHeight, inExpectedBMI, inExpectedStatus):
    person = DummyBody("TestSubject", inHeight, inWeight);
    result = analyzeVitals(person);
    assert result["name"] == "TestSubject";
    assert round(result["bmi"], 2) == inExpectedBMI;
    assert result["status"] == inExpectedStatus;