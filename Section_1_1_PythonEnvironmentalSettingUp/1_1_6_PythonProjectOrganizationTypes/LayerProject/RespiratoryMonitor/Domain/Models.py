#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class BreathingSignal:
    """
    Represents a breathing signal constructed from a sequence of timestamp data.

    This class stores the timestamps of detected breaths (e.g., peak inhalations)
    and serves as a container for any future signal processing or feature 
    extraction functions related to respiratory analysis.

    Attributes:
        timestamps (list of float): List of timestamps (in seconds) indicating
                                    when breaths occurred during a session.
    """

    def __init__(self, timestamps):
        self.timestamps = timestamps;  # List of breath peak timestamps (in seconds)