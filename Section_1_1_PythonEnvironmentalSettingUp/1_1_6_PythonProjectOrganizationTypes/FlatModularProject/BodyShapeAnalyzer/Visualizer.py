#*********************************************************************************************************#
#********************************************* INTERFACING FUNCTIONS *************************************#
#*********************************************************************************************************#
def printReport(inAnalyzedReport):
    """
    Displays the analysis report of a human body profile in a labeled format.
    This function iterates over the fields of an analysis summary and prints
    each attribute (e.g. name, BMI, shape, status) with consistent formatting.
    Args:
        inAnalyzedReport (dict): A dictionary containing body analysis results,
                                 typically returned from analyzeBody().
    Returns:
        None
    Example:
        >>> printReport({
                "name": "Tan-Nhu",
                "bmi": 22.5,
                "shape": "Hourglass",
                "status": "Normal"
            })
        Visualizer::PrintReport:: name: Tan-Nhu
        Visualizer::PrintReport:: bmi: 22.5
        Visualizer::PrintReport:: shape: Hourglass
        Visualizer::PrintReport:: status: Normal
    """

    # Print the report about the body shape by printing each field and its value
    for fieldName, fieldValue in inAnalyzedReport.items():
        print(f"Visualizer::PrintReport:: {fieldName}: {fieldValue}");