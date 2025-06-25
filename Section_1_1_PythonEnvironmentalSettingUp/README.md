# Section 1.1 – Python Environment Setup

This module provides a foundational starting point for readers of the book  
"Human Anatomical Shape Metamorphosis: Statistical Shape Modelling, AI-Driven Prediction, and Applications"  
by Tan-Nhu Nguyen and Tien-Tuan Dao (ISTE Group, London, 2025).

It walks you through setting up a Python development environment suitable for biomedical image analysis, 3D shape processing, and AI-driven modelling tasks. The tools and packages installed here are essential for executing all subsequent code examples across the book’s chapters.

## Short Contents

- Installation of Python on Microsoft Windows  
- Installation of Visual Studio Code on Microsoft Windows  
- Installation of Windows Terminal on Microsoft Windows  
- Examples of how to install numpy or other packages for python using pip in Terminal

## 1. Installation of Python on Microsoft Windows

- **Step 1:** Download the Python installer for Windows  
  Visit the official Python website:  
  [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)  

  In this book, we support only Python versions **from 3.7 to 3.12**.  
  Choose the appropriate version for your system (most users should select the **64-bit Windows installer**).

- **Step 2:** Run the downloaded installer  
  Double-click the `.exe` file to launch the Python Setup Wizard.  
  On the first screen, make sure to **check the box**:
  - ✅ *Add Python 3.x to PATH*  
  Then click **Customize installation** (recommended) or **Install Now** for default settings.

- **Step 3 (Optional - Customize Installation):**  
  If you selected **Customize installation**, it's recommended to:
  - Keep all optional features checked (e.g., `pip`, `IDLE`, `Documentation`, `Python test suite`)
  - On the **Advanced Options** screen, enable:
    - ✅ *Install for all users*
    - ✅ *Add Python to environment variables*
    - ✅ *Precompile standard library*

  Click **Install** to proceed.

- **Step 4:** Verify the installation  
  After installation completes, open the **Command Prompt** (`Win + R`, type `cmd`, press Enter), then type:

  ```bash
  python --version

## 2. Installation of Visual Studio Code on Microsoft Windows

- **Step 1:** Download the Visual Studio Code (VS Code) installer for Windows from the official website:  
  [https://code.visualstudio.com/](https://code.visualstudio.com/)

- **Step 2:** Run the downloaded installer. During the setup process, it is recommended to:
  - Accept the license agreement  
  - Choose the default installation location  
  - **Enable the following options when prompted:**
    - *Add "Open with Code" action to Windows Explorer file context menu*
    - *Add to PATH (available after restart)*  
  These options make it easier to open folders/files in VS Code and access the `code` command from the Terminal.

- **Step 3:** Complete the installation by clicking **Install**, then **Finish**.

- **Step 4:** After installation, open **Visual Studio Code** from the Start Menu or Desktop shortcut. You can verify it's working by launching it and checking for the welcome screen.

- **Step 5 (Optional but Recommended):** Install the **Python extension**:
  - Go to the **Extensions** tab (left sidebar or press `Ctrl+Shift+X`)
  - Search for **Python** and install the extension developed by **Microsoft**  
  This extension provides syntax highlighting, linting, IntelliSense, and debugging support for Python.

## 3. Installation of Windows Terminal on Microsoft Windows

The Windows Terminal provides a modern, fast, and flexible terminal interface for Windows users. It is recommended for use with Python, Git, and other command-line tools in this book.

- **Step 1:** Open the Microsoft Store  
  Click on the **Start Menu**, search for **Microsoft Store**, and open it.

- **Step 2:** Search for "Windows Terminal"  
  In the Microsoft Store search bar, type:  
  `Windows Terminal`  
  Select the official app published by **Microsoft Corporation**.

- **Step 3:** Click **Get** or **Install**  
  Wait for the download and installation to complete.

- **Step 4:** Launch Windows Terminal  
  After installation, click **Open**, or find it later via the Start Menu by typing `Windows Terminal`.

- **Step 5 (Optional but Recommended):**  
  Set **Windows Terminal** as your default terminal:
  - Open **Windows Terminal**
  - Click the **dropdown arrow** (top bar) → **Settings**
  - Under **Startup**, set:
    - *Default terminal application*: `Windows Terminal`
    - *Default profile*: choose your preferred shell (e.g., **Command Prompt** or **PowerShell**)

  Click **Save** to apply changes.

Windows Terminal is now ready for use with Python, Git, and other development tools throughout this book.
