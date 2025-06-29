#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os, sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));

import pytest;
from Core.Respiration import calculateBPM, classifyBreathing;

#*********************************************************************************************************#
#********************************************* TESTING FUNCTIONS *****************************************#
#*********************************************************************************************************#
@pytest.mark.parametrize("inTimeStamps, inExpectedBPM", [
    ([0, 4, 8, 12], 45.0),     # 3 breaths in 12 seconds → 15 BPM
    ([0, 3, 6, 9, 12], 60.0),  # 4 breaths in 12 seconds → 20 BPM
    ([0, 2, 4, 6, 8], 60.0),   # 4 breaths in 8 seconds → 30 BPM
    ([0], 0),                  # Not enough data
])
def test_calculateBPM(inTimeStamps, inExpectedBPM):
    assert calculateBPM(inTimeStamps) == inExpectedBPM;

@pytest.mark.parametrize("inBPM, inExpectedStatus", [
    (8, "Shallow"),
    (15, "Normal"),
    (25, "Rapid"),
])
def test_classifyBreathing(inBPM, inExpectedStatus):
    assert classifyBreathing(inBPM) == inExpectedStatus;