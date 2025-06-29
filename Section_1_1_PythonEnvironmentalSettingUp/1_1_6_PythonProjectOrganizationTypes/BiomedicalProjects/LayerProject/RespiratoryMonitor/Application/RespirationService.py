#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from Domain.BreathingLogic import calculateBPM, classifyBreathing;
from Domain.Models import BreathingSignal;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def analyzeRespiration(timestamps):
    """
    Processes breathing timestamps to determine breathing rate and classification.

    This function takes a list of breath event timestamps, encapsulates them in 
    a BreathingSignal object, calculates the breathing rate (in breaths per minute),
    and classifies the breathing status based on that rate.

    Args:
        timestamps (list of float): A list of timestamps (in seconds) representing
                                    individual breath events (e.g., inhalations).

    Returns:
        dict: A dictionary containing:
            - "breaths_per_minute" (float): Calculated breathing frequency.
            - "status" (str): Classification of breathing pattern (e.g., "Normal", "Shallow", "Rapid").
    """

    signal = BreathingSignal(timestamps);
    bpm = calculateBPM(signal.timestamps);
    status = classifyBreathing(bpm);
    return {
        "breaths_per_minute": bpm,
        "status": status
    };