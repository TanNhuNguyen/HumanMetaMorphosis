#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def calculateBPM(inTimeStamps):
    """
    Calculates the average breaths per minute (BPM) from a sequence of timestamps.

    This function estimates the respiratory rate by:
    - Measuring the total time between the first and last breath timestamp.
    - Counting the number of breath intervals (which is one less than the number of timestamps).
    - Computing the breaths per minute by dividing breath count by duration (in seconds) and multiplying by 60.

    Args:
        inTimeStamps (list of float): List of timestamps (in seconds) for detected breath peaks,
                                      sorted in chronological order.

    Returns:
        float: Estimated breathing rate in breaths per minute (BPM), 
               rounded to two decimal places. Returns 0 if fewer than two timestamps are given.
    """

    if len(inTimeStamps) < 2:
        return 0;
    duration = inTimeStamps[-1] - inTimeStamps[0];
    breath_count = len(inTimeStamps) - 1;
    return round((breath_count / duration) * 60, 2);
def classifyBreathing(inBPM):
    """
    Classifies the breathing pattern based on breaths per minute (BPM).

    This function uses simple thresholds to categorize the breathing rate:
    - "Shallow": Less than 10 BPM
    - "Normal": Between 10 and 20 BPM (inclusive)
    - "Rapid": Greater than 20 BPM

    Args:
        inBPM (float): Breathing rate in breaths per minute.

    Returns:
        str: A string label representing the breathing classification.
    """

    if inBPM < 10:
        return "Shallow"
    elif inBPM <= 20:
        return "Normal"
    else:
        return "Rapid"
