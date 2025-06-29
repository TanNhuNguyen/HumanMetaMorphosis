#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def getSampleData():
    """
    Returns a dictionary containing simulated vital signs for testing or demonstration purposes.

    This function provides fixed sample data values representing common health metrics:
    - Heart rate in beats per minute (bpm)
    - Body temperature in degrees Celsius
    - Blood oxygen saturation level (SpO2) as a percentage

    Returns:
        dict: A dictionary with keys "HeartRate", "Temperature", and "SpO2",
              each mapped to a representative numeric value.
    """

    return {
        "HeartRate": 78,
        "Temperature": 36.7,
        "SpO2": 97
    };