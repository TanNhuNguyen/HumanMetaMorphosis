#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os, sys;
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));

from Infrastructure.SensorSimulator import simulateBreathingSignal;
from Application.RespirationService import analyzeRespiration;

#*********************************************************************************************************#
#********************************************* MAIN PROCESSING FUNCTIONS *********************************#
#*********************************************************************************************************#
def simulateAndAnalyzeRespiration():
    # Initialize
    print("main::simulateAndAnalyzeRespiration:: Initializing.");

    # Generate breathing signal
    print("main::simulateAndAnalyzeRespiration:: Generating breathing signals ...");
    timestamps = simulateBreathingSignal()
    
    # Analyze the generated breathing signal
    print("main::simulateAndAnalyzeRespiration:: Ananlyzed the generated breathing signals ...");
    result = analyzeRespiration(timestamps)

    # Print out analyzed result
    print("main::simulateAndAnalyzeRespiration:: Printing out analyzed results ...");
    print(f"main::simulateAndAnalyzeRespiration:: \t Breaths per Minute: {result['breaths_per_minute']}")
    print(f"main::simulateAndAnalyzeRespiration:: \t Breathing Status: {result['status']}")

    # Finished processing
    print("main::simulateAndAnalyzeRespiration:: Finished processing.");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTION *********************************************#
#*********************************************************************************************************#
def main():
    os.system("cls");
    simulateAndAnalyzeRespiration();
if __name__ == "__main__":
    main()