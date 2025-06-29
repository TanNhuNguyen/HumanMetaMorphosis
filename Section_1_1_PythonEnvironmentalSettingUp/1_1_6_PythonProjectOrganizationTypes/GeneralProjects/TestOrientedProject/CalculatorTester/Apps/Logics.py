# Interfacing functions
def add(inA, inB):
    return inA + inB;
def divide(inA, inB):
    if inB == 0:
        raise ZeroDivisionError("TestLogics::divide:: Cannot divide by zero.");