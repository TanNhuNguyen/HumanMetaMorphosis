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
def displayInformation_1():
    print("main::displayInformation_1:: This is the information 1.");
def displayInformation_2():
    print("main::displayInformation_2:: This is the information 2.");
def addTwoNumbers(inNumber1, inNumber2):
    result = inNumber1 + inNumber2;
    return result;
def subtractTwoNumbers(inNumber1, inNumber2):
    result = inNumber1 - inNumber2;
    return result;

#*********************************************************************************************************#
#********************************************* MAIN PROCESSING FUNCTION **********************************#
#*********************************************************************************************************#
def processingStep_1():
    # Initializing
    print("main::processingStep_1:: Initializing ...");
    
    # Print the information 1
    print("main::processingStep_1:: Printing the information 1 ...");
    displayInformation_1();

    # Print the information 2
    print("main::processingStep_1:: Printing the information 2 ...");
    displayInformation_2();

    # Finished processing
    print("main::processingStep_1:: Finished processing.");
def processingStep_2():
    # Initializing
    print("main::processingStep_2:: Initializing ...");
    
    # Adding two numbers
    print("main::processingStep_2:: Adding two numbers ...");
    number1 = 10; number2 = 20;
    result = addTwoNumbers(number1, number2);
    print("\t Number 1: ", number1);
    print("\t Number 2: ", number2);
    print("\t Resut: ", result);

    # Subtract two numbers
    print("main::processingStep_2:: Subtracting two numbers ...");
    number1 = 10; number2 = 20;
    result = subtractTwoNumbers(number1, number2);
    print(f"\t Number 1: {number1}, Number 2: {number2}, Result: {result}");

    # Finished processing
    print("main::processingStep_2:: Finished processing.");
def processingStep_3():
    # Initializing
    print("main::processingStep_3:: Initializing ...");
    
    # Multiply two numbers
    print("main::processingStep_3:: Multiplying two numbers ...");
    ## Initialize the class for multiplication and compute the first case
    number1 = 100; number2 = 200;
    multiplier = TwoNumberMultiplier(number1, number2);
    multiplier.computeResult();
    result = multiplier.getResult();
    print(f"\t The first case: {number1} x {number2} = {result}");
    ## Update the numbers and compute the second case
    number1 = 300; number2 = 400;
    multiplier.setNumber1(number1); multiplier.setNumber2(number2);
    multiplier.computeResult();
    result = multiplier.getResult();
    print(f"\t The second case: {number1} x {number2} = {result}");
    ## Update the numbers and compute the third case
    number1 = 500; number2 = 600;
    multiplier.setTwoNumbers(number1, number2);
    multiplier.computeResult();
    result = multiplier.getResult();
    print(f"\t The second case: {number1} x {number2} = {result}");

    # Finished processing
    print("main::processingStep_3:: Finished processing.");
def processingStep_4():
    # Initializing
    print("main::processingStep_4:: Initializing ...");
    def divideTwoNumbers(inNumber1, inNumber2):
        return inNumber1 / inNumber2;

    # Divide two numbers
    print("main::processingStep_4:: Dividing two numbers ...");
    number1 = 600; number2 = 3;
    result = divideTwoNumbers(number1, number2);
    print(f"\t {number1} x {number2} = {result}");
    
    # Finished processing
    print("main::processingStep_4:: Finished processing.");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTION *********************************************#
#*********************************************************************************************************#
def main():
    processingStep_4();
if __name__ == "__main__":
    main();