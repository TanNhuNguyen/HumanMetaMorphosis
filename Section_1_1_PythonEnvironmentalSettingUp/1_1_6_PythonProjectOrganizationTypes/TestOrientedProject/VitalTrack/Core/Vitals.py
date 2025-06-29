#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def calculateBMI(inWeightKg, inHeightCm):
    """
    Calculates the Body Mass Index (BMI) based on weight and height.

    The Body Mass Index is a widely used metric for assessing whether a person’s
    body weight is within a healthy range relative to their height. This function
    converts height from centimeters to meters and applies the BMI formula:
        BMI = weight (kg) / (height (m))²

    Args:
        inWeightKg (float): Weight of the individual in kilograms.
        inHeightCm (float): Height of the individual in centimeters.

    Returns:
        float: The calculated BMI value, rounded to two decimal places.
    """

    height_m = inHeightCm / 100;
    return round(inWeightKg / (height_m ** 2), 2);
def interpretBMI(inBMI):
    """
    Interprets a numeric BMI value and classifies it into a health category.

    This function compares the input Body Mass Index (BMI) against standard 
    thresholds to categorize the individual's weight status according to 
    widely accepted health guidelines.

    Categories:
        - BMI < 18.5     → "Underweight"
        - 18.5 ≤ BMI < 25 → "Normal"
        - 25 ≤ BMI < 30   → "Overweight"
        - BMI ≥ 30        → "Obese"

    Args:
        inBMI (float): The Body Mass Index value to interpret.

    Returns:
        str: A classification label describing the BMI category.
    """

    if inBMI < 18.5:
        return "Underweight";
    elif inBMI < 25:
        return "Normal";
    elif inBMI < 30:
        return "Overweight";
    return "Obese";