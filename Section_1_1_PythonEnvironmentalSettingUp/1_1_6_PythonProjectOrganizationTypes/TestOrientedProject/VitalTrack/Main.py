#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os;
from Core.Analyzer import analyzeVitals

#*********************************************************************************************************#
#********************************************* SUPPORTING CLASSES ****************************************#
#*********************************************************************************************************#
class HumanBody:
    def __init__(self, name, inHeightCm, inWeightKg):
        self.name = name;
        self.heightCm = inHeightCm;
        self.weightKg = inWeightKg;

#*********************************************************************************************************#
#********************************************* MAIN PROCESSING FUNCTION **********************************#
#*********************************************************************************************************#
def analyzeHumanVitals():
    # Initialize
    print("Main::analyzeHumanVitals:: Initializing ...");

    # Generate human body structure
    print("Main::analyzeHumanVitals:: Generate human body structure ...");
    person = HumanBody("Tan-Nhu", 170, 65);

    # Analyze the human vital
    print("Main::analyzeHumanVitals:: Analyze the human vital ...");
    result = analyzeVitals(person);

    # Print summary
    print("Main::analyzeHumanVitals:: Print summary ...");
    from Plugins import PrintSummary;
    PrintSummary.run(result);

    # Finished processing.
    print("Main::analyzeHumanVitals:: Finished processing ...");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTION *********************************************#
#*********************************************************************************************************#
def main():
    os.system("cls");
    analyzeHumanVitals();
if __name__ == "__main__":
    main();