#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class BreathingSignal:
    """
    Represents a breathing signal constructed from a sequence of timestamp data.

    Attributes:
        timestamps (list of float): List of timestamps (in seconds) indicating
                                    when breaths occurred during a session.
    """

    def __init__(self, timestamps):
        self.timestamps = timestamps;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def calculateBPM(inTimeStamps):
    """
    Calculates the average breaths per minute (BPM) from a list of breathing timestamps.

    This function determines the respiratory rate based on the time difference between 
    the first and last breath events, and the number of breaths that occurred. It assumes 
    timestamps are sorted and measured in seconds.

    Formula:
        BPM = ((number of breaths - 1) / duration in seconds) * 60

    Args:
        inTimeStamps (list of float): A list of timestamps (in seconds) indicating 
                                      when each breath occurred, in chronological order.

    Returns:
        float: The calculated breaths per minute (BPM), rounded to 2 decimal places.
               Returns 0 if fewer than two timestamps are provided.
    """

    if len(inTimeStamps) < 2:
        return 0;
    duration = inTimeStamps[-1] - inTimeStamps[0];
    return round(((len(inTimeStamps) - 1) / duration) * 60, 2);
def classifyBreathing(inBPM):
    """
    Classifies the breathing pattern based on breaths per minute (BPM).

    This function uses standard threshold values to determine whether a person's
    respiratory rate is within a healthy range:
    - "Shallow": Fewer than 10 breaths per minute
    - "Normal": 10 to 20 breaths per minute (inclusive)
    - "Rapid" : More than 20 breaths per minute

    Args:
        inBPM (float): Breathing rate in breaths per minute.

    Returns:
        str: A string representing the breathing status classification:
             "Shallow", "Normal", or "Rapid".
    """

    if inBPM < 10:
        return "Shallow";
    elif inBPM <= 20:
        return "Normal";
    return "Rapid";