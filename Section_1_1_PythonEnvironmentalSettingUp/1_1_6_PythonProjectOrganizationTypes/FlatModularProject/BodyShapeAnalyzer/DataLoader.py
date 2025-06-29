#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from BodyModel import HumanBody;

#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def getSampleData():
    """
    Creates and returns a list of sample HumanBody instances for testing.
    This function generates two example profiles with predefined body measurements
    to simulate real user data for development, testing, or demonstration purposes.
    Returns:
        list: A list containing two HumanBody objects.
    """

    # Generate the first body shape with the information of the shape
    body1 = HumanBody("Tan-Nhu", 170, 65, 80, 90, 95);

    # Generate the second body shape with information of the shape
    body2 = HumanBody("Alex", 160, 85, 100, 95, 90);

    # Return the two body shape as the two elements inside an array
    return [body1, body2];