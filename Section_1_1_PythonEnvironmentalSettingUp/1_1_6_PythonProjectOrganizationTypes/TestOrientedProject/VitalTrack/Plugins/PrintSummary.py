#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def run(data):
    """
    Prints a structured summary of health-related data to the console.

    This function takes in a dictionary containing health metrics and outputs
    each key-value pair with a consistent prefix for easy identification during
    debugging or real-time monitoring.

    Args:
        data (dict): A dictionary of health statistics, where keys are metric
                     labels (e.g., "HeartRate", "BMI") and values are their
                     corresponding readings.

    Example:
        data = {"HeartRate": 72, "BMI": 22.4}
        Output:
            PrintSummary::run:: Health Summary:
            PrintSummary::run:: HeartRate: 72
            PrintSummary::run:: BMI: 22.4
    """

    print("PrintSummary::run:: Health Summary:");
    for key, value in data.items():
        print(f"PrintSummary::run:: {key}: {value}");