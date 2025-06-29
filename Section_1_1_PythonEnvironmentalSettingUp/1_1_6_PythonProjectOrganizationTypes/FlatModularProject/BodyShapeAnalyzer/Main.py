#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os;

from DataLoader import getSampleData;
from HealthAnalysis import analyzeBody;
from Visualizer import printReport;

#*********************************************************************************************************#
#********************************************* PROCESSING FUNCTIONS **************************************#
#*********************************************************************************************************#
def loadDataAndAnalyzeBodyShape():
    # Initializing
    print("Main::loadDataAndAnalyzeBodyShape:: Initializing ...");

    # Getting the list of bodies
    print("Main::loadDataAndAnalyzeBodyShape:: Getting the list of body shape information ...");
    bodyList = getSampleData();
    print("Main::loadDataAndAnalyzeBodyShape:: \t The body list: ", bodyList);

    # For each body in the body list try to analyze and print the information
    print("Main::loadDataAndAnalyzeBodyShape:: For each body in the list, analyze and print out the result ...");
    for body in bodyList:
        bodyReport = analyzeBody(body);
        printReport(bodyReport);

    # Finished processing
    print("Main::loadDataAndAnalyzeBodyShape:: Finished processing.");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTIONS ********************************************#
#*********************************************************************************************************#
def main():
    os.system("cls");
    loadDataAndAnalyzeBodyShape();
if __name__ == "__main__":
    main();