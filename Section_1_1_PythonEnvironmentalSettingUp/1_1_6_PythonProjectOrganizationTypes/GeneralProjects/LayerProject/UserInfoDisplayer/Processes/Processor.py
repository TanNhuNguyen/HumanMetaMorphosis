from Database.DatabaseHandler import saveUser, fetchUsers

def registerUser(inName, inEmail):
    """
    Validates user input and registers a new user to the system.
    This function checks for empty fields and basic email format validation,
    then saves the user's name and email to the database after trimming whitespace.
    Args:
        inName (str): The user's full name.
        inEmail (str): The user's email address.
    Returns:
        None
    Example:
        >>> registerUser("Tan-Nhu", "tan@example.com")
        User registered!
    """
    if not inName or not inEmail or "@" not in inEmail:
        print("processor::registerUser:: Invalid name or email.");
        return;
    saveUser({"name": inName.strip(), "email": inEmail.strip()});
    print("processor::registerUser:: User registered!");
def getAllUsers():
    """
    Retrieves the complete list of registered users from the database layer.
    Delegates the call to the data access function `fetchUsers` and returns its result.
    Returns:
        list: A list of user dictionaries, each containing user attributes such as 'name' and 'email'.
    Example:
        >>> getAllUsers()
        [{'name': 'Tan-Nhu', 'email': 'tan@example.com'}, ...]
    """
    return fetchUsers();