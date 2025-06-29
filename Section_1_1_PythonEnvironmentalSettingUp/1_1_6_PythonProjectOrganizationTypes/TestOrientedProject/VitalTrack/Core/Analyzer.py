#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from Core.Vitals import calculateBMI, interpretBMI;
from Core.Respiration import BreathingSignal, calculateBPM, classifyBreathing;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def analyzeVitals(body):
    """
    Analyzes basic vital statistics to compute BMI and categorize health status.

    This function calculates the Body Mass Index (BMI) using the weight and height 
    from the provided `body` object and classifies the BMI into a health status 
    (e.g., Underweight, Normal, Overweight, Obese).

    Args:
        body (object): An object representing a person's physical characteristics.
                       Expected attributes:
                           - name (str): Person's name
                           - weightKg (float): Weight in kilograms
                           - heightCm (float): Height in centimeters

    Returns:
        dict: A dictionary containing:
            - "name" (str): Person's name
            - "bmi" (float): Calculated BMI value
            - "status" (str): Health status based on the BMI category
    """

    bmi = calculateBMI(body.weightKg, body.heightCm);
    return {
        "name": body.name,
        "bmi": bmi,
        "status": interpretBMI(bmi)
    };
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

    signal = BreathingSignal(timestamps)
    bpm = calculateBPM(signal.timestamps)
    status = classifyBreathing(bpm)
    return {
        "breaths_per_minute": bpm,
        "status": status
    }
