#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import os;

from Core.Logger import getSampleData;
from Core.PluginManager import loadPlugins;

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def getSampleAndRunPlugins():
    # Initializing
    print("Main::getSampleAndRunPlugins:: Initializing ...");

    # Get sample data
    print("Main::getSampleAndRunPlugins:: Get sample data ...");
    data = getSampleData();

    # Load plugins
    print("Main::getSampleAndRunPlugins:: Load plugins ...");
    plugins = loadPlugins();

    # Run all plugins
    print("Main::getSampleAndRunPlugins:: Run all plugins ...");
    for plugin in plugins:
        plugin.run(data);

    # Finished processing
    print("Main::getSampleAndRunPlugins:: Finished processing.");

#*********************************************************************************************************#
#********************************************* MAIN FUNCTION *********************************************#
#*********************************************************************************************************#
def main():
    os.system("cls");
    getSampleAndRunPlugins();
if __name__ == "__main__":
    main()