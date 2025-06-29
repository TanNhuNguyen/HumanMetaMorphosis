#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from ShapeMetrics import calculateBmi, determineBodyShape;

#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def analyzeBody(inBody):
    """
    Analyzes a HumanBody instance to assess health status and classify body shape.
    This function calculates the Body Mass Index (BMI), determines the body shape
    based on circumference ratios, and interprets the BMI category such as
    "Normal", "Overweight", or "Obese".
    Args:
        inBody (HumanBody): An instance of the HumanBody class containing measurement attributes.
    Returns:
        dict: A dictionary summarizing the analysis result with:
            - 'name' (str): Person’s name
            - 'bmi' (float): Calculated BMI value
            - 'shape' (str): Body shape classification
            - 'status' (str): BMI interpretation label
    Example:
        >>> body = HumanBody("Tan-Nhu", 170, 65, 80, 90, 95)
        >>> analyzeBody(body)
        {'name': 'Tan-Nhu', 'bmi': 22.49, 'shape': 'Hourglass', 'status': 'Normal'}
    """

    # Compute the BMI value
    bmiScore = calculateBmi(inBody.weightKg, inBody.heightCm);

    # Getting the shape types as a string of description
    shapeType = determineBodyShape(inBody.waistCm, inBody.hipCm, inBody.shoulderCm);

    # Return the output as the field names and their values using dictionary
    return {
        "name": inBody.name,
        "bmi": bmiScore,
        "shape": shapeType,
        "status": interpretBmi(bmiScore)
    };
def interpretBmi(inBMIScore):
    """
    Categorizes the BMI score into a standard health classification.
    This function evaluates the numerical BMI score and maps it to one of four
    categories defined by common clinical guidelines.
    Args:
        inBMIScore (float): The Body Mass Index (BMI) value to interpret.
    Returns:
        str: A string representing the health classification.
             Options: "Underweight", "Normal", "Overweight", or "Obese".
    Example:
        >>> interpretBmi(22.5)
        'Normal'
    """
    
    # Based on the value of BMI we inteprete the BMI scores
    if inBMIScore < 18.5:
        return "Underweight";
    elif inBMIScore < 25:
        return "Normal";
    elif inBMIScore < 30:
        return "Overweight";
    else:
        return "Obese";