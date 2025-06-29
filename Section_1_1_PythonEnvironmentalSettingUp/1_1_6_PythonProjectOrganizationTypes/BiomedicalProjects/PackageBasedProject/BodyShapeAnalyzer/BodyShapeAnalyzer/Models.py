#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class HumanBody:
    """
    Represents a human body with physical measurements used for analysis.

    This class is designed to hold individual-specific data such as name,
    height, weight, and body circumferences. It serves as a structured 
    input for functions that analyze BMI and body shape.

    Attributes:
        name (str): The person's name.
        height_cm (float): Height in centimeters.
        weight_kg (float): Weight in kilograms.
        waist_cm (float): Waist circumference in centimeters.
        hip_cm (float): Hip circumference in centimeters.
        shoulder_cm (float): Shoulder width in centimeters.
    """

    def __init__(self, name, height_cm, weight_kg, waist_cm, hip_cm, shoulder_cm):
        self.name = name;
        self.height_cm = height_cm;
        self.weight_kg = weight_kg;
        self.waist_cm = waist_cm;
        self.hip_cm = hip_cm;
        self.shoulder_cm = shoulder_cm;