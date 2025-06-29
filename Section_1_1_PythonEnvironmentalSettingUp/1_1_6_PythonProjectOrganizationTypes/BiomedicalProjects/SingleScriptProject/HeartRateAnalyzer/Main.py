# Supporting libraries
import statistics

# Supporting functions
def loadHeartRateData():
    """
    Simulate reading heart rate data (in BPM) from a wearable device or sensor.
    Returns a list of readings.
    """
    # Simulated 20-second sample (1 reading per second), the patient conducted the measurements of the heart rates in 20 seconds
    return [78, 80, 77, 82, 85, 120, 125, 88, 76, 73, 70, 90, 110, 89, 75, 72, 71, 100, 95, 93];
def analyzeHeartRate(inHeartRateData):
    """
    Analyze heart rate data to calculate average, variability, and provide health warnings.
    Args:
        inHeartRateData (list of int): Heart rate values in beats per minute (BPM).
    Returns:
        dict: Summary with average BPM, standard deviation, max/min, and condition warning.
    """
    # Compute the mean value from the heart rate data
    meanHeartRate = round(statistics.mean(inHeartRateData), 2);

    # Compute the std value of from the heart rate data
    stdHeartRate = round(statistics.stdev(inHeartRateData), 2);

    # Based on the heart rate data, we generate a summarization of the data
    summary = {
        "Average HR": meanHeartRate,
        "HR Variability (std dev)": stdHeartRate,
        "Max HR": max(inHeartRateData),
        "Min HR": min(inHeartRateData),
        "Warning": None
    }

    # Check the mean heart rates to make conclusion about the condition of the heart rates
    if meanHeartRate < 60:
        summary["Warning"] = "Possible bradycardia (low heart rate)"
    elif meanHeartRate > 100:
        summary["Warning"] = "Possible tachycardia (elevated heart rate)"

    # Return the summary of the heart rate
    return summary

# Processing function
def simulateHeartRateAnalyses():
    # Initialize
    print("main::simulateHeartRateAnalyses:: Initializing ...");

    # Loading the synthetic heart rate data
    print("main::simulateHeartRateAnalyses:: Loading the synthetic heart rate data ...");
    data = loadHeartRateData();

    # Call the function to analyze the heart rates
    print("main::simulateHeartRateAnalyses:: Calling the function to analyze the heart rates ...");
    result = analyzeHeartRate(data)

    # Print out the summary
    print("main::simulateHeartRateAnalyses:: Heart Rate Summary:")
    for key, value in result.items():
        print(f"main::simulateHeartRateAnalyses:: {key}: {value}");

# Main function
def main():
    simulateHeartRateAnalyses();

if __name__ == "__main__":
    main()