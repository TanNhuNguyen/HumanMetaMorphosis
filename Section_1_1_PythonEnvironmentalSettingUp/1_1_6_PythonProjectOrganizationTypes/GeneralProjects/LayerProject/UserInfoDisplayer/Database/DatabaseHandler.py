# General buffers
_userDatabase = [];

# Interfacing functions
def saveUser(inUserDictionary):
    """
    Saves user information to the internal user database.
    Args:
        inUserDictionary (dict): A dictionary containing user details such as
            'name', 'email', and other relevant profile data.
    Returns:
        None
    Example:
        >>> saveUser({'name': 'Tan-Nhu', 'email': 'tan@example.com'})
    """
    _userDatabase.append(inUserDictionary);
def fetchUsers():
    """
    Retrieves all stored user entries from the internal database.
    Returns:
        list: A list of dictionaries, each representing a registered user.
    Example:
        >>> fetchUsers()
        [{'name': 'Tan-Nhu', 'email': 'tan@example.com'}, ...]
    """
    return _userDatabase;
