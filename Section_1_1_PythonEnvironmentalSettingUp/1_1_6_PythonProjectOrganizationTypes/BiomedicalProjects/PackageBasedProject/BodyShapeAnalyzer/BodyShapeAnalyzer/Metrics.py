#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def calculateBMI(weight_kg, height_cm):
    """
    Calculates the Body Mass Index (BMI) based on weight and height.

    BMI is a standard health metric calculated by dividing a person's 
    weight in kilograms by the square of their height in meters. 
    This function returns the BMI rounded to two decimal places.

    Args:
        weight_kg (float): The person's weight in kilograms.
        height_cm (float): The person's height in centimeters.

    Returns:
        float: The calculated BMI value, rounded to two decimal places.
    """

    height_m = height_cm / 100;
    return round(weight_kg / (height_m ** 2), 2);
def determineShape(waist, hip, shoulder):
    """
    Determines the body shape based on waist, hip, and shoulder measurements.

    This function classifies body shape into one of four categories using 
    simple proportional rules:
    - "Apple": Waist is proportionally larger than both hip and shoulder widths.
    - "Pear": Hips are wider than shoulders, and waist is not dominant.
    - "Rectangle": Shoulders and hips are nearly equal in width (within 5 cm).
    - "Hourglass": Shoulders and hips differ by more than 5 cm but waist is smaller.

    Args:
        waist (float): Waist circumference in centimeters.
        hip (float): Hip circumference in centimeters.
        shoulder (float): Shoulder width in centimeters.

    Returns:
        str: A string label representing the estimated body shape.
    """

    if waist / hip > 0.9 and waist / shoulder > 0.9:
        return "Apple";
    elif hip > shoulder:
        return "Pear";
    elif abs(shoulder - hip) < 5:
        return "Rectangle";
    else:
        return "Hourglass";