import logging  # Importing the logging module for logging messages.  # Importiert das Logging-Modul für das Loggen von Nachrichten.

import platform  # Importing platform module to detect the operating system.  # Importiert das Platform-Modul, um das Betriebssystem zu erkennen.

import numpy as np  # Importing numpy for numerical operations, especially for image data.  # Importiert numpy für numerische Operationen, insbesondere für Bilddaten.

import tmrl.config.config_constants as cfg  # Importing configuration constants from the tmrl.config module.  # Importiert Konfigurationskonstanten aus dem Modul tmrl.config.

# Check if the platform is Windows.
if platform.system() == "Windows":  # Checking if the operating system is Windows.  # Überprüft, ob das Betriebssystem Windows ist.

    import win32gui  # Importing win32gui to interact with Windows GUI components.  # Importiert win32gui, um mit Windows-GUI-Komponenten zu interagieren.
    import win32ui  # Importing win32ui for user interface functionalities in Windows.  # Importiert win32ui für Benutzeroberflächenfunktionen in Windows.
    import win32con  # Importing win32con for constants related to Windows GUI operations.  # Importiert win32con für Konstanten, die mit Windows-GUI-Operationen zusammenhängen.

    # Define a class to interact with a specific window.
    class WindowInterface:  # Defining a class to handle window operations.  # Definiert eine Klasse für Fensteroperationen.
        def __init__(self, window_name):  # Initializer for the WindowInterface class.  # Initialisierer für die Klasse WindowInterface.
            self.window_name = window_name  # Storing the window name passed as argument.  # Speichert den Fensternamen, der als Argument übergeben wird.

            hwnd = win32gui.FindWindow(None, self.window_name)  # Finding the window handle (hwnd) using the window name.  # Findet das Fenster-Handle (hwnd) anhand des Fensternamens.
            assert hwnd != 0, f"Could not find a window named {self.window_name}."  # Ensures the window exists.  # Stellt sicher, dass das Fenster existiert.

            while True:  # Loop to handle if the window is minimized.  # Schleife, um zu behandeln, wenn das Fenster minimiert ist.
                wr = win32gui.GetWindowRect(hwnd)  # Getting the window's rectangle (position and size).  # Holt das Fenster-Rechteck (Position und Größe).
                cr = win32gui.GetClientRect(hwnd)  # Getting the client area of the window.  # Holt den Client-Bereich des Fensters.
                if cr[2] > 0 and cr[3] > 0:  # Check if client area width and height are greater than 0.  # Überprüft, ob die Breite und Höhe des Client-Bereichs größer als 0 sind.
                    break  # Exits loop once valid client area is found.  # Verlässt die Schleife, sobald ein gültiger Client-Bereich gefunden wurde.

            # Calculate differences for window borders.
            self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]  # Difference in width between the window and client area.  # Unterschied in der Breite zwischen Fenster und Client-Bereich.
            self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]  # Difference in height between the window and client area.  # Unterschied in der Höhe zwischen Fenster und Client-Bereich.

            self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)  # Calculating border sizes.  # Berechnung der Randgrößen.

            self.x_origin_offset = - self.w_diff // 2  # X-axis offset for positioning the window.  # X-Achsen-Verschiebung für die Fensterpositionierung.
            self.y_origin_offset = 0  # Y-axis offset for positioning the window.  # Y-Achsen-Verschiebung für die Fensterpositionierung.

        # Function to take a screenshot of the window.
        def screenshot(self):  # Method to capture a screenshot of the window.  # Methode, um einen Screenshot des Fensters zu erstellen.
            hwnd = win32gui.FindWindow(None, self.window_name)  # Finding the window handle again.  # Findet das Fenster-Handle erneut.
            assert hwnd != 0, f"Could not find a window named {self.window_name}."  # Verifying the window exists.  # Überprüft, ob das Fenster existiert.

            while True:  # Loop to avoid crashes if the window is minimized.  # Schleife, um Abstürze zu vermeiden, wenn das Fenster minimiert ist.
                x, y, x1, y1 = win32gui.GetWindowRect(hwnd)  # Get the window position and size.  # Holt die Fensterposition und -größe.
                w = x1 - x - self.w_diff  # Calculating width considering borders.  # Berechnet die Breite unter Berücksichtigung der Ränder.
                h = y1 - y - self.h_diff  # Calculating height considering borders.  # Berechnet die Höhe unter Berücksichtigung der Ränder.
                if w > 0 and h > 0:  # Checks if valid width and height are found.  # Überprüft, ob gültige Breite und Höhe gefunden wurden.
                    break  # Exit loop once valid dimensions are found.  # Verlässt die Schleife, wenn gültige Dimensionen gefunden wurden.
            hdc = win32gui.GetWindowDC(hwnd)  # Getting the device context for the window.  # Holt den Geräte-Kontext für das Fenster.
            dc = win32ui.CreateDCFromHandle(hdc)  # Creating DC object from window's handle.  # Erstellt ein DC-Objekt aus dem Fenster-Handle.
            memdc = dc.CreateCompatibleDC()  # Creating a compatible memory device context.  # Erstellt einen kompatiblen Speicher-Geräte-Kontext.
            bitmap = win32ui.CreateBitmap()  # Creating a bitmap object for the screenshot.  # Erstellt ein Bitmap-Objekt für den Screenshot.
            bitmap.CreateCompatibleBitmap(dc, w, h)  # Creating a compatible bitmap of the specified size.  # Erstellt ein kompatibles Bitmap der angegebenen Größe.
            oldbmp = memdc.SelectObject(bitmap)  # Selecting the bitmap into memory DC.  # Wählt das Bitmap in den Speicher-DC aus.
            memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)  # Copying window image to memory.  # Kopiert das Fensterbild in den Speicher.
            bits = bitmap.GetBitmapBits(True)  # Getting the raw bitmap data.  # Holt die Rohbitmap-Daten.
            img = (np.frombuffer(bits, dtype='uint8'))  # Converts the raw data to a numpy array.  # Konvertiert die Rohdaten in ein numpy-Array.
            img.shape = (h, w, 4)  # Reshapes the array to match the image dimensions.  # Ändert die Form des Arrays, um den Bilddimensionen zu entsprechen.
            memdc.SelectObject(oldbmp)  # Restores the previous object to memory DC.  # Stellt das vorherige Objekt im Speicher-DC wieder her.
            win32gui.DeleteObject(bitmap.GetHandle())  # Deletes the bitmap object to free memory.  # Löscht das Bitmap-Objekt, um den Speicher freizugeben.
            memdc.DeleteDC()  # Deletes the memory device context.  # Löscht den Speicher-Geräte-Kontext.
            win32gui.ReleaseDC(hwnd, hdc)  # Releases the device context.  # Gibt den Geräte-Kontext frei.
            return img  # Returns the screenshot image as numpy array.  # Gibt das Screenshot-Bild als numpy-Array zurück.

        # Function to move and resize the window.
        def move_and_resize(self, x=1, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):  # Method to move and resize the window.  # Methode, um das Fenster zu verschieben und die Größe anzupassen.
            x += self.x_origin_offset  # Adjusting the X position with the origin offset.  # Passt die X-Position mit der Ursprungsverschiebung an.
            y += self.y_origin_offset  # Adjusting the Y position with the origin offset.  # Passt die Y-Position mit der Ursprungsverschiebung an.
            w += self.w_diff  # Adjusting width to include window border.  # Passt die Breite an, um den Fensterrahmen einzubeziehen.
            h += self.h_diff  # Adjusting height to include window border.  # Passt die Höhe an, um den Fensterrahmen einzubeziehen.
            hwnd = win32gui.FindWindow(None, self.window_name)  # Finding the window handle again.  # Findet das Fenster-Handle erneut.
            assert hwnd != 0, f"Could not find a window named {self.window_name}."  # Ensures the window exists.  # Stellt sicher, dass das Fenster existiert.
            win32gui.MoveWindow(hwnd, x, y, w, h, True)  # Moves and resizes the window with new dimensions.  # Verschiebt und ändert die Größe des Fensters mit den neuen Dimensionen.

# Check if the platform is Linux.
elif platform.system() == "Linux":  # Checking if the operating system is Linux.  # Überprüft, ob das Betriebssystem Linux ist.

    import subprocess  # Importing subprocess for running shell commands.  # Importiert subprocess zum Ausführen von Shell-Befehlen.
    import time  # Importing time module for time-related operations.  # Importiert das Zeitmodul für zeitbezogene Operationen.
    import mss  # Importing mss for screen capture functionality.  # Importiert mss für Bildschirmaufnahme-Funktionalität.

    # Function to get window ID by name.
    def get_window_id(name):  # Method to find a window by name using xdotool.  # Methode, um ein Fenster anhand des Namens mit xdotool zu finden.
        try:  # Try block to handle errors during execution.  # Try-Block zur Fehlerbehandlung während der Ausführung.
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)  # Searching for the window using xdotool.  # Sucht das Fenster mit xdotool.
            window_ids = result.stdout.strip().split('\n')  # Extracting window IDs from the output.  # Extrahiert Fenster-IDs aus der Ausgabe.
            for window_id in window_ids:  # Looping through each window ID to find the correct window.  # Schleift durch jede Fenster-ID, um das richtige Fenster zu finden.
                result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                        capture_output=True, text=True, check=True)  # Fetching the window name by ID.  # Holt den Fensternamen anhand der ID.
                if result.stdout.strip() == name:  # Checking if the window name matches.  # Überprüft, ob der Fensternamen übereinstimmt.
                    logging.debug(f"detected window {name}, id={window_id}")  # Logs detected window name and ID.  # Protokolliert den erkannten Fensternamen und die ID.
                    return window_id  # Returns the found window ID.  # Gibt die gefundene Fenster-ID zurück.

            logging.error(f"failed to find window '{name}'")  # Logs an error if the window is not found.  # Protokolliert einen Fehler, wenn das Fenster nicht gefunden wurde.
            raise NoSuchWindowException(name)  # Raises an exception if the window is not found.  # Wirft eine Ausnahme, wenn das Fenster nicht gefunden wurde.

        except subprocess.CalledProcessError as e:  # Handling subprocess errors.  # Behandelt Fehler im Zusammenhang mit subprocess.
            logging.error(f"process error searching for window '{name}")  # Logs an error during the process execution.  # Protokolliert einen Fehler während der Ausführung des Prozesses.
            raise NoSuchWindowException(name)  # Raises an exception if the process fails.  # Wirft eine Ausnahme, wenn der Prozess fehlschlägt.


def get_window_geometry(name):  # Defines a function to get the geometry of a window by its name.  # Definiert eine Funktion, um die Geometrie eines Fensters anhand seines Namens zu erhalten.
    """  # A docstring explaining the purpose of the function.  # Ein Docstring, der den Zweck der Funktion erklärt.
    FIXME: xdotool doesn't agree with MSS, so we use hardcoded offsets instead for now  # Comment indicating a limitation and workaround.  # Kommentar, der eine Einschränkung und eine Umgehungslösung angibt.
    """
    try:  # Tries the following block of code.  # Versucht den folgenden Codeblock.
        result = subprocess.run(['xdotool', 'search', '--name', name, 'getwindowgeometry', '--shell'],  # Executes xdotool to get window geometry.  # Führt xdotool aus, um die Fenstergeometrie zu erhalten.
                                capture_output=True, text=True, check=True)  # Captures the output and ensures the command runs successfully.  # Erfasst die Ausgabe und stellt sicher, dass der Befehl erfolgreich ausgeführt wird.
        elements = result.stdout.strip().split('\n')  # Splits the output into lines.  # Teilt die Ausgabe in Zeilen.
        res_id = None  # Initializes variables for window ID, position, and size.  # Initialisiert Variablen für Fenster-ID, Position und Größe.
        res_x = None  #  #  # 
        res_y = None  #  #  # 
        res_w = None  #  #  # 
        res_h = None  #  #  # 
        for elt in elements:  # Loops through the elements of the output.  # Schleift durch die Elemente der Ausgabe.
            low_elt = elt.lower()  # Converts the element to lowercase for comparison.  # Wandelt das Element in Kleinbuchstaben um, um Vergleiche durchzuführen.
            if low_elt.startswith("window="):  # Checks if the element is the window ID.  # Überprüft, ob das Element die Fenster-ID ist.
                res_id = elt[7:]  # Extracts the window ID.  # Extrahiert die Fenster-ID.
            elif low_elt.startswith("x="):  # Checks if the element is the x-coordinate.  # Überprüft, ob das Element die x-Koordinate ist.
                res_x = int(elt[2:])  # Extracts and converts the x-coordinate.  # Extrahiert und konvertiert die x-Koordinate.
            elif low_elt.startswith("y="):  # Checks if the element is the y-coordinate.  # Überprüft, ob das Element die y-Koordinate ist.
                res_y = int(elt[2:])  # Extracts and converts the y-coordinate.  # Extrahiert und konvertiert die y-Koordinate.
            elif low_elt.startswith("width="):  # Checks if the element is the width.  # Überprüft, ob das Element die Breite ist.
                res_w = int(elt[6:])  # Extracts and converts the width.  # Extrahiert und konvertiert die Breite.
            elif low_elt.startswith("height="):  # Checks if the element is the height.  # Überprüft, ob das Element die Höhe ist.
                res_h = int(elt[7:])  # Extracts and converts the height.  # Extrahiert und konvertiert die Höhe.

        if None in (res_id, res_x, res_y, res_w, res_h):  # Checks if any required value is missing.  # Überprüft, ob ein erforderlicher Wert fehlt.
            raise GeometrySearchException(f"Found None in window '{name}' geometry: {(res_id, res_x, res_y, res_w, res_h)}")  # Raises an exception if any value is missing.  # Wirft eine Ausnahme, wenn ein Wert fehlt.

        return res_id, res_x, res_y, res_w, res_h  # Returns the window geometry data.  # Gibt die Fenstergeometriedaten zurück.

    except subprocess.CalledProcessError as e:  # Catches errors from the subprocess command.  # Fängt Fehler des Subprozesses ab.
        logging.error(f"process error searching for {name} window geometry")  # Logs the error.  # Protokolliert den Fehler.
        raise e  # Raises the exception.  # Wirft die Ausnahme erneut.

class NoSuchWindowException(Exception):  # Defines a custom exception for when a window cannot be found.  # Definiert eine benutzerdefinierte Ausnahme, wenn ein Fenster nicht gefunden werden kann.
    """thrown if a named window can't be found"""  # Docstring for the exception.  # Docstring für die Ausnahme.
    pass  # Passes without doing anything.  # Macht nichts.

class GeometrySearchException(Exception):  # Defines a custom exception for geometry search failure.  # Definiert eine benutzerdefinierte Ausnahme für das Scheitern der Geometriesuche.
    """thrown if geometry search fails"""  # Docstring for the exception.  # Docstring für die Ausnahme.
    pass  # Passes without doing anything.  # Macht nichts.

class WindowInterface:  # Defines a class to interact with windows.  # Definiert eine Klasse zur Interaktion mit Fenstern.
    def __init__(self, window_name):  # Initializes the window interface with a window name.  # Initialisiert das Fensterinterface mit einem Fensternamen.
        self.sct = mss.mss()  # Initializes a screen capture object (mss) for screenshot functionality.  # Initialisiert ein Bildschirmaufnahme-Objekt (mss) für Screenshot-Funktionalität.
        self.window_name = window_name  # Sets the window name.  # Setzt den Fensternamen.
        try:  # Tries the following block of code.  # Versucht den folgenden Codeblock.
            self.window_id = get_window_id(window_name)  # Retrieves the window ID for the given window name.  # Ruft die Fenster-ID für den angegebenen Fensternamen ab.
        except NoSuchWindowException as e:  # Catches the exception if the window is not found.  # Fängt die Ausnahme, wenn das Fenster nicht gefunden wird.
            logging.error(f"get_window_id failed, is xdotool correctly installed? {str(e)}")  # Logs the error if window ID retrieval fails.  # Protokolliert den Fehler, wenn das Abrufen der Fenster-ID fehlschlägt.
            self.window_id = None  # Sets the window ID to None.  # Setzt die Fenster-ID auf None.

        self.w = None  # Initializes the window width to None.  # Initialisiert die Fensterbreite auf None.
        self.h = None  # Initializes the window height to None.  # Initialisiert die Fensterhöhe auf None.
        self.x = None  # Initializes the window x-coordinate to None.  # Initialisiert die x-Koordinate des Fensters auf None.
        self.y = None  # Initializes the window y-coordinate to None.  # Initialisiert die y-Koordinate des Fensters auf None.
        self.x_offset = cfg.LINUX_X_OFFSET  # Sets the x offset from the configuration.  # Setzt den x-Versatz aus der Konfiguration.
        self.y_offset = cfg.LINUX_Y_OFFSET  # Sets the y offset from the configuration.  # Setzt den y-Versatz aus der Konfiguration.
        self.process = None  # Initializes the process variable to None.  # Initialisiert die Prozessvariable auf None.

    def __del__(self):  # Destructor method to clean up resources.  # Destruktormethode zum Bereinigen von Ressourcen.
        pass  # Does nothing in this case.  # Macht in diesem Fall nichts.
        self.sct.close()  # Closes the screen capture object.  # Schließt das Bildschirmaufnahme-Objekt.

    def execute_command(self, c):  # Executes a command in the terminal.  # Führt einen Befehl im Terminal aus.
        if self.process is None or self.process.poll() is not None:  # Checks if there is no process or if the process has finished.  # Überprüft, ob kein Prozess vorhanden ist oder der Prozess beendet wurde.
            self.process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE,  # Starts a new subprocess.  # Startet einen neuen Unterprozess.
                                            stderr=subprocess.PIPE)
        self.process.stdin.write(c.encode())  # Sends the command to the process's input.  # Sendet den Befehl an die Eingabe des Prozesses.
        self.process.stdin.flush()  # Flushes the input buffer.  # Leert den Eingabepuffer.

    def screenshot(self):  # Captures a screenshot of the window.  # Macht einen Screenshot des Fensters.
        try:  # Tries the following block of code.  # Versucht den folgenden Codeblock.
            monitor = {"top": self.x + self.x_offset, "left": self.y + self.y_offset, "width": self.w, "height": self.h}  # Defines the capture area based on window geometry.  # Definiert den Aufnahmebereich basierend auf der Fenstergeometrie.
            img = np.array(self.sct.grab(monitor))  # Captures the screenshot as a NumPy array.  # Macht den Screenshot als NumPy-Array.
            return img  # Returns the captured image.  # Gibt das aufgenommene Bild zurück.

        except subprocess.CalledProcessError as e:  # Catches errors from the subprocess.  # Fängt Fehler des Subprozesses ab.
            logging.error(f"failed to capture screenshot")  # Logs the error.  # Protokolliert den Fehler.
            raise e  # Raises the exception.  # Wirft die Ausnahme erneut.

    def move_and_resize(self, x=0, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):  # Moves and resizes the window.  # Bewegt und ändert die Größe des Fensters.
        logging.debug(f"prepare {self.window_name} to {w}x{h} @ {x}, {y}")  # Logs the intended move and resize operation.  # Protokolliert die beabsichtigte Verschiebung und Größenänderung.

        try:  # Tries the following block of code.  # Versucht den folgenden Codeblock.
            c_focus = f"xdotool windowfocus {self.window_id}\n"  # Prepares the command to focus the window.  # Bereitet den Befehl vor, um das Fenster in den Fokus zu setzen.
            self.execute_command(c_focus)  # Executes the focus command.  # Führt den Fokusbefehl aus.

            c_move = f"xdotool windowmove {str(self.window_id)} {str(x)} {str(y)}\n"  # Prepares the move command.  # Bereitet den Verschiebebefehl vor.
            self.execute_command(c_move)  # Executes the move command.  # Führt den Verschiebebefehl aus.

            c_resize = f"xdotool windowsize {str(self.window_id)} {str(w)} {str(h)}\n"  # Prepares the resize command.  # Bereitet den Größenänderungsbefehl vor.
            self.execute_command(c_resize)  # Executes the resize command.  # Führt den Größenänderungsbefehl aus.

            self.w = w  # Sets the window width.  # Setzt die Fensterbreite.
            self.h = h  # Sets the window height.  # Setzt die Fensterhöhe.
            self.x = x  # Sets the window x-coordinate.  # Setzt die x-Koordinate des Fensters.
            self.y = y  # Sets the window y-coordinate.  # Setzt die y-Koordinate des Fensters.

            time.sleep(1)  # Pauses for 1 second to ensure the window resize is complete.  # Pausiert für 1 Sekunde, um sicherzustellen, dass die Fenstergrößenänderung abgeschlossen ist.

        except subprocess.CalledProcessError as e:  # Catches subprocess errors.  # Fängt Unterprozessfehler ab.
            logging.error(f"failed to resize window_id '{self.window_id}'")  # Logs the error.  # Protokolliert den Fehler.

        except NoSuchWindowException as e:  # Catches the exception if the window cannot be found.  # Fängt die Ausnahme ab, wenn das Fenster nicht gefunden werden kann.
            logging.error(f"failed to find window: {str(e)}")  # Logs the error.  # Protokolliert den Fehler.

def profile_screenshot():  # Defines a function to profile screenshot performance.  # Definiert eine Funktion zur Profilerstellung der Screenshot-Leistung.
    from pyinstrument import Profiler  # Imports the Profiler class from pyinstrument.  # Importiert die Profiler-Klasse aus pyinstrument.
    pro = Profiler()  # Creates a new profiler instance.  # Erstellt eine neue Profiler-Instanz.
    window_interface = WindowInterface("Trackmania")  # Creates a window interface for the "Trackmania" window.  # Erstellt ein Fensterinterface für das Fenster "Trackmania".
    pro.start()  # Starts the profiler.  # Startet den Profiler.
    for _ in range(5000):  # Loops 5000 times.  # Schleift 5000 Mal.
        snap = window_interface.screenshot()  # Takes a screenshot during each iteration.  # Macht während jeder Iteration einen Screenshot.
    pro.stop()  # Stops the profiler.  # Stoppt den Profiler.
    pro.print(show_all=True)  # Prints the profiling results.  # Gibt die Profiling-Ergebnisse aus.

if __name__ == "__main__":  # Checks if the script is being run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    profile_screenshot()  # Calls the profile_screenshot function.  # Ruft die Funktion profile_screenshot auf.
