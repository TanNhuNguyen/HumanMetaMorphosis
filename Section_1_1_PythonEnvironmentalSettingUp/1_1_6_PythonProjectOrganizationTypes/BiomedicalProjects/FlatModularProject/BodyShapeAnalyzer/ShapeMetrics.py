#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def calculateBmi(weightKg, heightCm):
    """
    Calculates the Body Mass Index (BMI) based on weight and height.
    BMI is a metric used to assess body fat based on height and weight.
    The formula used is: BMI = weight (kg) / height² (m²)
    Args:
        weightKg (float): Person's weight in kilograms.
        heightCm (float): Person's height in centimeters.
    Returns:
        float: The BMI value, rounded to two decimal places.
    Example:
        >>> calculateBmi(70, 175)
        22.86
    """
    # Getting the height in meters
    heightM = heightCm / 100;

    # Compute and return the BMI value
    return round(weightKg / (heightM ** 2), 2);
def determineBodyShape(waistCm, hipCm, shoulderCm):
    """
    Classifies human body shape based on circumference ratios.
    This function uses the waist-to-hip and waist-to-shoulder ratios to infer
    common body shape categories. It follows a basic heuristic model based on
    proportional relationships between the upper and lower body measurements.
    Args:
        waistCm (float): Waist circumference in centimeters.
        hipCm (float): Hip circumference in centimeters.
        shoulderCm (float): Shoulder width in centimeters.
    Returns:
        str: A string label representing the body shape. Options are:
             "Apple", "Pear", "Rectangle", or "Hourglass".
    Example:
        >>> determineBodyShape(80, 90, 95)
        'Hourglass'
    """

    # Compute the waist to hip ratio
    waistToHipRatio = waistCm / hipCm;

    # Compute the waist to shoulder ratio
    waistToShoulderRatio = waistCm / shoulderCm;

    # Based on the waist and hip ratio make description of the body shape
    if waistToHipRatio > 0.9 and waistToShoulderRatio > 0.9:
        return "Apple";
    elif hipCm > shoulderCm:
        return "Pear";
    elif abs(shoulderCm - hipCm) < 5:
        return "Rectangle";
    else:
        return "Hourglass";