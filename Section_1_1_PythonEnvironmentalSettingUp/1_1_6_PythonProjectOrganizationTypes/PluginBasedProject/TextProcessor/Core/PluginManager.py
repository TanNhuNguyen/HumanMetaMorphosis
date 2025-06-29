import os;
import importlib;

PLUGIN_FOLDER = "Plugins";

def loadPlugins():
    """
    Dynamically loads all valid plugin modules from the PLUGIN_FOLDER.

    This function scans the specified plugin directory for Python files 
    (excluding special files like '__init__.py'), imports each module, 
    and checks for the presence of a 'process' function. Any valid plugin 
    modules with this function are collected and returned in a list 
    as (moduleName, processFunction) tuples.

    Returns:
        list of tuples: Each tuple contains the module name and a reference
        to its 'process' function.
    """
    # Defining buffers for saving the plugins
    plugins = [];

    # Iterate for each files inside the plugin folder
    for filename in os.listdir(PLUGIN_FOLDER):
        # Ensure it is the plugin not __init__.py
        if filename.endswith(".py") and not filename.startswith("__"):
            # Getting the module name by not having the extension and dot "."
            moduleName = filename[:-3];

            # Import the module to the buffer
            module = importlib.import_module(f"{PLUGIN_FOLDER}.{moduleName}");

            # Check if each module have the process function
            if hasattr(module, "process"):
                plugin = (moduleName, module.process);
                plugins.append(plugin);

    # Return all plugins to the output
    return plugins;

