#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import csv, os;

#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
currentFolder = os.path.dirname(os.path.abspath(__file__));
healthLogFilePath = currentFolder + "/../HealthLog.csv";

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def run(data):
    """
    Writes health-related data to a CSV file for record-keeping or analysis.

    This function creates (or overwrites) a CSV file at the path specified by 
    'healthLogFilePath'. It writes a header row followed by each key-value pair 
    in the provided data dictionary as a new row in the file.

    Args:
        data (dict): A dictionary containing health metrics (e.g., heart rate, temperature).
                     Keys represent metric names, and values represent corresponding measurements.
    
    Notes:
        - Assumes 'healthLogFilePath' is a valid global variable that stores the output file path.
        - Each row in the CSV will have the format: Metric, Value
    """

    with open(healthLogFilePath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in data.items():
            writer.writerow([key, value])