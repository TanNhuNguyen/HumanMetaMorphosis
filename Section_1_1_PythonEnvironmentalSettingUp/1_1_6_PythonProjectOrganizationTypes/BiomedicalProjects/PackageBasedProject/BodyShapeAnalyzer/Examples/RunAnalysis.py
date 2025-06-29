#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os; import sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))); # Add the project folder to the system path
from BodyShapeAnalyzer.Models import HumanBody;
from BodyShapeAnalyzer.Analysis import analyze;

#*********************************************************************************************************#
#********************************************* PROCESSING FUNCTIONS **************************************#
#*********************************************************************************************************#
def generateSyntheticBodyAndAnalyze():
    # Initialize
    print("Main::generateSyntheticBodyAndAnalyze:: Initializing ...");

    # Generate synthetic data
    print("Main::generateSyntheticBodyAndAnalyze:: Generating synthetic data ...");
    body = HumanBody("Tan-Nhu", 170, 65, 80, 90, 95);
    report = analyze(body);

    # Print out the report from the analyzed results
    print("Main::generateSyntheticBodyAndAnalyze:: Print out the report from the analyzed results ...");
    for key, value in report.items():
        print(f"Main::generateSyntheticBodyAndAnalyze:: {key}: {value}");

    # Finished processing
    print("Main::generateSyntheticBodyAndAnalyze:: Finished processing.");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTION *********************************************#
#*********************************************************************************************************#
def main():
    os.system("cls");
    generateSyntheticBodyAndAnalyze();
if __name__ == '__main__':
    main();