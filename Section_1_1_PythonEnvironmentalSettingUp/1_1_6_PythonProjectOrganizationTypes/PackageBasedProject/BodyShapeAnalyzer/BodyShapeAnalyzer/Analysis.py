#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from .Metrics import calculateBMI, determineShape;

#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def analyze(body):
    """
    Analyzes a person's physical metrics to determine BMI and body shape.
    This function uses the provided body measurements to calculate:
    - Body Mass Index (BMI)
    - Estimated body shape based on waist, hip, and shoulder dimensions
    - BMI status (e.g., underweight, normal, overweight, etc.)
    Args:
        body (object): An object with the following attributes:
            - name (str): The person's name
            - weight_kg (float): Weight in kilograms
            - height_cm (float): Height in centimeters
            - waist_cm (float): Waist circumference in centimeters
            - hip_cm (float): Hip circumference in centimeters
            - shoulder_cm (float): Shoulder width in centimeters
    Returns:
        dict: A dictionary containing the person's name, BMI value,
              estimated body shape, and interpreted BMI status.
    """

    bmi = calculateBMI(body.weight_kg, body.height_cm)
    shape = determineShape(body.waist_cm, body.hip_cm, body.shoulder_cm)
    return {
        "name": body.name,
        "bmi": bmi,
        "shape": shape,
        "status": interpret_bmi(bmi)
    }
def interpret_bmi(bmi):
    """
    Interprets a numeric Body Mass Index (BMI) value into a standard health category.
    This function classifies the BMI into one of the following categories 
    based on WHO guidelines:
    - Underweight: BMI < 18.5
    - Normal: 18.5 <= BMI < 25
    - Overweight: 25 <= BMI < 30
    - Obese: BMI >= 30
    Args:
        bmi (float): The calculated Body Mass Index value.
    Returns:
        str: A string representing the corresponding BMI category.
    """

    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"