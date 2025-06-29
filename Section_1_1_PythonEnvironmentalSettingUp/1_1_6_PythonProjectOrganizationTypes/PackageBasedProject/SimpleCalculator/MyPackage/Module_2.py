#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#

#*********************************************************************************************************#
#********************************************* SUPPORTING BUFFERS ****************************************#
#*********************************************************************************************************#

#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class TwoNumberMultiplier:
    # Initializing functions
    def __init__(self, inNumber1, inNumber2):
        self.number1 = inNumber1;
        self.number2 = inNumber2;
        self.result = 0;
    
    # Processing functions
    def computeResult(self):
        self.result = self.number1 * self.number2;
    
    # Interfacing functions
    def setNumber1(self, inNumber1):
        self.number1 = inNumber1;
    def setNumber2(self, inNumber2):
        self.number2 = inNumber2;
    def setTwoNumbers(self, inNumber1, inNumber2):
        self.number1 = inNumber1;
        self.number2 = inNumber2;
    def getResult(self):
        return self.result;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def addTwoNumbers(inNumber1, inNumber2):
    result = inNumber1 + inNumber2;
    return result;
def subtractTwoNumbers(inNumber1, inNumber2):
    result = inNumber1 - inNumber2;
    return result;
def divideTwoNumbers(inNumber1, inNumber2):
    return inNumber1 / inNumber2;