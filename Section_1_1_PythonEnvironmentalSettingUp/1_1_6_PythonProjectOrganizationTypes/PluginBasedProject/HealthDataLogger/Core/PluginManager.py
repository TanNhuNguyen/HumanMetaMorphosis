#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
import importlib;
import os;

#*********************************************************************************************************#
#********************************************* SUPPORTING BUFFERS ****************************************#
#*********************************************************************************************************#
currentFolder = os.path.dirname(os.path.abspath(__file__));
PLUGIN_FOLDER = currentFolder + "/../Plugins";

#*********************************************************************************************************#
#********************************************* SUPPORTING FUNCTIONS **************************************#
#*********************************************************************************************************#
def loadPlugins():
    """
    Dynamically loads all plugin modules from the PLUGIN_FOLDER that define a 'run' function.

    This function scans the plugin directory for Python files (excluding '__init__.py'),
    imports each module, and checks for the presence of a callable named 'run'.
    Modules that meet this criterion are collected into a list and returned.

    Assumes that PLUGIN_FOLDER is the name of the package or directory (e.g., "Plugins").

    Returns:
        list: A list of plugin modules that contain a 'run' function.
    """

    plugins = [];
    for file in os.listdir(PLUGIN_FOLDER):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3];
            module = importlib.import_module(f"Plugins.{module_name}");
            if hasattr(module, "run"):
                plugins.append(module);
    return plugins;