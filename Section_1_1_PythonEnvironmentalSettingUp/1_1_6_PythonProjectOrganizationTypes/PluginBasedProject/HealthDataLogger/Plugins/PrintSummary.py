#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
def run(data):
    """
    Prints a formatted health summary from a given data dictionary.

    This function takes in a dictionary of key-value health metrics 
    (e.g., heart rate, temperature) and prints each item in a readable format.
    It's typically used within a plugin system to display a quick summary report.

    Args:
        data (dict): A dictionary containing health-related data points,
                     where keys are metric names (e.g., "HeartRate") and
                     values are their corresponding measurements.
    """

    print("PrintSummary::run:: Health Summary:")
    for key, value in data.items():
        print(f"PrintSummary::run:: {key}: {value}");