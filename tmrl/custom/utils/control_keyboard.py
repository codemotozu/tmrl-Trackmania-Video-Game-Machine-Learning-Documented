# http://www.flint.jp/misc/?q=dik&lang=en  key indicator  # URL for reference of key indicators.  # URL für die Referenz von Tastenindikatoren.

# standard library imports  # Importieren von Standardbibliotheken.  # Standardbibliotheken importieren
import platform  # Imports the platform module to detect the operating system.  # Importiert das Modul 'platform', um das Betriebssystem zu erkennen.

if platform.system() == "Windows":  # Checks if the operating system is Windows.  # Überprüft, ob das Betriebssystem Windows ist.
    # standard library imports  # Standardbibliotheken importieren.  # Importiert Standardbibliotheken
    import ctypes  # Imports the ctypes library for calling C functions from DLLs.  # Importiert die ctypes-Bibliothek, um C-Funktionen aus DLLs aufzurufen.

    SendInput = ctypes.windll.user32.SendInput  # Defines SendInput function for simulating keyboard events.  # Definiert die SendInput-Funktion zum Simulieren von Tastaturereignissen.

    # constants:  # Defining constant values for keycodes.  # Definieren von Konstanten für Tastencodes.
    W = 0x11  # Hexadecimal keycode for the "W" key.  # Hexadezimaler Tastencode für die "W"-Taste.
    A = 0x1E  # Hexadecimal keycode for the "A" key.  # Hexadezimaler Tastencode für die "A"-Taste.
    S = 0x1F  # Hexadecimal keycode for the "S" key.  # Hexadezimaler Tastencode für die "S"-Taste.
    D = 0x20  # Hexadecimal keycode for the "D" key.  # Hexadezimaler Tastencode für die "D"-Taste.
    DEL = 0xD3  # Hexadecimal keycode for the "DEL" key.  # Hexadezimaler Tastencode für die "DEL"-Taste.
    R = 0x13  # Hexadecimal keycode for the "R" key.  # Hexadezimaler Tastencode für die "R"-Taste.

    # C struct redefinitions  # Redefinition von C-Strukturen.  # C-Strukturen umdefinieren.

    PUL = ctypes.POINTER(ctypes.c_ulong)  # Defines a pointer type for a ULONG (unsigned long).  # Definiert einen Zeigertyp für ULONG (unsigned long).

    # KeyBdInput structure: Defines keyboard input events.  # Struktur 'KeyBdInput': Definiert Tastatur-Eingabeereignisse.
    class KeyBdInput(ctypes.Structure):  
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    # HardwareInput structure: Defines hardware input events.  # Struktur 'HardwareInput': Definiert Hardware-Eingabeereignisse.
    class HardwareInput(ctypes.Structure):  
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short), ("wParamH", ctypes.c_ushort)]

    # MouseInput structure: Defines mouse input events.  # Struktur 'MouseInput': Definiert Maus-Eingabeereignisse.
    class MouseInput(ctypes.Structure):  
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

    # Input_I union: Unites different input types.  # Eingabeunion 'Input_I': Vereint verschiedene Eingabetypen.
    class Input_I(ctypes.Union):  
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

    # Input structure: Defines the general input structure.  # Struktur 'Input': Definiert die allgemeine Eingabestruktur.
    class Input(ctypes.Structure):  
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

    # Key Functions  # Tastenfunktionen.  # Funktionen für Tasteneingaben.

    def PressKey(hexKeyCode):  # Function to press a key.  # Funktion, um eine Taste zu drücken.
        extra = ctypes.c_ulong(0)  # Creates an extra variable for additional input info.  # Erstellt eine zusätzliche Variable für weitere Eingabedaten.
        ii_ = Input_I()  # Creates an Input_I object.  # Erstellt ein Input_I-Objekt.
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))  # Sets up the key press event.  # Richten Sie das Tastendrücken-Ereignis ein.
        x = Input(ctypes.c_ulong(1), ii_)  # Creates an Input object with type 1 (keyboard).  # Erstellt ein Input-Objekt mit Typ 1 (Tastatur).
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # Sends the input event to the system.  # Sendet das Eingabeereignis an das System.

    def ReleaseKey(hexKeyCode):  # Function to release a key.  # Funktion, um eine Taste loszulassen.
        extra = ctypes.c_ulong(0)  # Creates an extra variable for additional input info.  # Erstellt eine zusätzliche Variable für weitere Eingabedaten.
        ii_ = Input_I()  # Creates an Input_I object.  # Erstellt ein Input_I-Objekt.
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))  # Sets up the key release event.  # Richten Sie das Tastendruckloslassen-Ereignis ein.
        x = Input(ctypes.c_ulong(1), ii_)  # Creates an Input object with type 1 (keyboard).  # Erstellt ein Input-Objekt mit Typ 1 (Tastatur).
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # Sends the input event to the system.  # Sendet das Eingabeereignis an das System.

    def apply_control(action):  # Function to apply movement control based on action.  # Funktion, um Steuerung je nach Aktion anzuwenden.
        if 'f' in action:  # If the action includes "f", press the "W" key.  # Wenn die Aktion "f" enthält, drücke die "W"-Taste.
            PressKey(W)  
        else:
            ReleaseKey(W)  # Otherwise, release the "W" key.  # Andernfalls lasse die "W"-Taste los.
        if 'b' in action:  # If the action includes "b", press the "S" key.  # Wenn die Aktion "b" enthält, drücke die "S"-Taste.
            PressKey(S)  
        else:
            ReleaseKey(S)  # Otherwise, release the "S" key.  # Andernfalls lasse die "S"-Taste los.
        if 'l' in action:  # If the action includes "l", press the "A" key.  # Wenn die Aktion "l" enthält, drücke die "A"-Taste.
            PressKey(A)  
        else:
            ReleaseKey(A)  # Otherwise, release the "A" key.  # Andernfalls lasse die "A"-Taste los.
        if 'r' in action:  # If the action includes "r", press the "D" key.  # Wenn die Aktion "r" enthält, drücke die "D"-Taste.
            PressKey(D)  
        else:
            ReleaseKey(D)  # Otherwise, release the "D" key.  # Andernfalls lasse die "D"-Taste los.

    def keyres():  # Function to reset keys by pressing and releasing DEL.  # Funktion, um Tasten zurückzusetzen, indem DEL gedrückt und losgelassen wird.
        PressKey(DEL)  # Press the "DEL" key.  # Drücke die "DEL"-Taste.
        ReleaseKey(DEL)  # Release the "DEL" key.  # Lasse die "DEL"-Taste los.

elif platform.system() == "Linux":  # Checks if the operating system is Linux.  # Überprüft, ob das Betriebssystem Linux ist.
    import subprocess  # Imports subprocess for running shell commands.  # Importiert subprocess, um Shell-Befehle auszuführen.
    import logging  # Imports logging for debugging.  # Importiert logging zum Debuggen.

    KEY_UP = "Up"  # Defines the key name for the "Up" key.  # Definiert den Tastenbezeichner für die "Up"-Taste.
    KEY_DOWN = "Down"  # Defines the key name for the "Down" key.  # Definiert den Tastenbezeichner für die "Down"-Taste.
    KEY_RIGHT = "Right"  # Defines the key name for the "Right" key.  # Definiert den Tastenbezeichner für die "Right"-Taste.
    KEY_LEFT = "Left"  # Defines the key name for the "Left" key.  # Definiert den Tastenbezeichner für die "Left"-Taste.
    KEY_BACKSPACE = "BackSpace"  # Defines the key name for the "Backspace" key.  # Definiert den Tastenbezeichner für die "Backspace"-Taste.

    process = None  # Initialize the process variable to None.  # Initialisiert die Prozessvariable mit None.

    def execute_command(c):  # Executes the given command.  # Führt den angegebenen Befehl aus.
        global process  # Makes the process variable global.  # Macht die Prozessvariable global.
        if process is None or process.poll() is not None:  # Checks if the process is not running or has finished.  # Überprüft, ob der Prozess nicht läuft oder beendet ist.
            logging.debug("(re-)create process")  # Logs that the process is being recreated.  # Protokolliert, dass der Prozess neu erstellt wird.
            process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE)  # Starts a new bash process.  # Startet einen neuen bash-Prozess.
        process.stdin.write(c.encode())  # Writes the command to the subprocess's input.  # Schreibt den Befehl in die Eingabe des Subprozesses.
        process.stdin.flush()  # Ensures that the input is processed.  # Stellt sicher, dass die Eingabe verarbeitet wird.

    def PressKey(key):  # Function to press a key.  # Funktion, um eine Taste zu drücken.
        c = f"xdotool keydown {str(key)}\n"  # Prepares the command to press the key.  # Bereitet den Befehl vor, um die Taste zu drücken.
        execute_command(c)  # Executes the command to press the key.  # Führt den Befehl aus, um die Taste zu drücken.

    def ReleaseKey(key):  # Function to release a key.  # Funktion, um eine Taste loszulassen.
        c = f"xdotool keyup {str(key)}\n"  # Prepares the command to release the key.  # Bereitet den Befehl vor, um die Taste loszulassen.
        execute_command(c)  # Executes the command to release the key.  # Führt den Befehl aus, um die Taste loszulassen.

    def apply_control(action, window_id=None):  # Function to apply movement control based on action.  # Funktion, um Steuerung basierend auf der Aktion anzuwenden.
        if window_id is not None:  # If a window ID is provided, focus the window.  # Wenn eine Fenster-ID angegeben ist, fokussiere das Fenster.
            c_focus = f"xdotool windowfocus {str(window_id)}"  # Command to focus on the window.  # Befehl, um das Fenster zu fokussieren.
            execute_command(c_focus)  # Executes the focus command.  # Führt den Fokus-Befehl aus.

        if 'f' in action:  # If the action includes "f", press the "Up" key.  # Wenn die Aktion "f" enthält, drücke die "Up"-Taste.
            PressKey(KEY_UP)  
        else:
            ReleaseKey(KEY_UP)  # Otherwise, release the "Up" key.  # Andernfalls lasse die "Up"-Taste los.
        if 'b' in action:  # If the action includes "b", press the "Down" key.  # Wenn die Aktion "b" enthält, drücke die "Down"-Taste.
            PressKey(KEY_DOWN)  
        else:
            ReleaseKey(KEY_DOWN)  # Otherwise, release the "Down" key.  # Andernfalls lasse die "Down"-Taste los.
        if 'l' in action:  # If the action includes "l", press the "Left" key.  # Wenn die Aktion "l" enthält, drücke die "Left"-Taste.
            PressKey(KEY_LEFT)  
        else:
            ReleaseKey(KEY_LEFT)  # Otherwise, release the "Left" key.  # Andernfalls lasse die "Left"-Taste los.
        if 'r' in action:  # If the action includes "r", press the "Right" key.  # Wenn die Aktion "r" enthält, drücke die "Right"-Taste.
            PressKey(KEY_RIGHT)  
        else:
            ReleaseKey(KEY_RIGHT)  # Otherwise, release the "Right" key.  # Andernfalls lasse die "Right"-Taste los.

    def keyres():  # Function to reset keys by pressing and releasing Backspace.  # Funktion, um Tasten zurückzusetzen, indem die Backspace-Taste gedrückt und losgelassen wird.
        PressKey(KEY_BACKSPACE)  # Press the "Backspace" key.  # Drücke die "Backspace"-Taste.
        ReleaseKey(KEY_BACKSPACE)  # Release the "Backspace" key.  # Lasse die "Backspace"-Taste los.

else:  # If the operating system is not Windows or Linux.  # Wenn das Betriebssystem weder Windows noch Linux ist.

    def apply_control(action):  # Define empty control function.  # Leere Steuerfunktion definieren.
        pass  # No control logic.  # Keine Steuerlogik.

    def keyres():  # Define empty key reset function.  # Leere Funktion zum Zurücksetzen von Tasten definieren.
        pass  # No key reset logic.  # Keine Logik zum Zurücksetzen von Tasten.
