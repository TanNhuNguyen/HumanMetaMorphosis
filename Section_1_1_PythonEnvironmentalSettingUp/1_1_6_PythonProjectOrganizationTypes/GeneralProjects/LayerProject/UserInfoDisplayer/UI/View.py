from Processes.Processor import registerUser, getAllUsers

def showMenu():
    print("view::showMenu:: User System Menu")
    print("view::showMenu:: 1. Register User")
    print("view::showMenu:: 2. Show All Users")
    print("view::showMenu:: 3. Exit")
    choice = input("view::showMenu:: Choose option: ")

    if choice == "1":
        name = input("view::showMenu:: Enter name: ")
        email = input("view::showMenu:: Enter email: ")
        registerUser(name, email)
    elif choice == "2":
        users = getAllUsers()
        if not users:
            print("view::showMenu:: No users found.")
        else:
            for user in users:
                print(f"view::showMenu:: {user['name']} - {user['email']}")
    elif choice == "3":
        print("view::showMenu:: Goodbye!")
        exit()
    else:
        print("view::showMenu:: Invalid choice.")
