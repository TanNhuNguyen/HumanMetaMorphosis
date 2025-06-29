#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class HumanBody:
    """
    Represents a human body with key physical measurements.

    This class is useful for modeling individuals in body shape analysis
    or biometric assessments. It stores height, weight, and key circumference
    values for shape classification or health calculations (e.g. BMI).
    """
    # Properties using camelCase
    def __init__(self, name, heightCm, weightKg, waistCm, hipCm, shoulderCm):
        """
        Initializes a HumanBody instance with basic measurements.

        Args:
            name (str): The person's name.
            heightCm (float): Height in centimeters.
            weightKg (float): Weight in kilograms.
            waistCm (float): Waist circumference in centimeters.
            hipCm (float): Hip circumference in centimeters.
            shoulderCm (float): Shoulder width in centimeters.
        """
        self.name = name; # Getting the name of the user
        self.heightCm = heightCm; # Getting the height of the user
        self.weightKg = weightKg; # Getting the weight of the user
        self.waistCm = waistCm; # Getting the waist length
        self.hipCm = hipCm; # Getting the hip length
        self.shoulderCm = shoulderCm; # Getting the shoulder length
