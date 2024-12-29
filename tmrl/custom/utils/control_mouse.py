# standard library imports
import platform  # Import the platform module to check the operating system.  # Importiert das Plattform-Modul, um das Betriebssystem zu überprüfen.

if platform.system() == "Windows":  # Check if the operating system is Windows.  # Überprüft, ob das Betriebssystem Windows ist.

    # third-party imports
    from pyautogui import click, mouseUp  # Import functions from the pyautogui library for mouse automation.  # Importiert Funktionen aus der pyautogui-Bibliothek zur Mausautomatisierung.

    def mouse_close_finish_pop_up_tm20(small_window=False):  # Defines a function to close a pop-up window, accepting an optional parameter to handle different window sizes.  # Definiert eine Funktion, um ein Pop-up-Fenster zu schließen, wobei ein optionaler Parameter die Handhabung von Fenstern unterschiedlicher Größe ermöglicht.
        if small_window:  # If the window is small, click at a specific position.  # Wenn das Fenster klein ist, wird an einer bestimmten Position geklickt.
            click(138, 100)  # Click at the coordinates (138, 100) on the screen.  # Klickt an den Koordinaten (138, 100) auf dem Bildschirm.
        else:  # If the window is not small, click at another position.  # Wenn das Fenster nicht klein ist, wird an einer anderen Position geklickt.
            click(550, 300)  # Click at the coordinates (550, 300) where the "improve" button is located.  # Klickt an den Koordinaten (550, 300), wo sich der "verbessern"-Button befindet.
        mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.

    def mouse_change_name_replay_tm20(small_window=False):  # Defines a function to change the name in the replay window.  # Definiert eine Funktion, um den Namen im Wiederholungsfenster zu ändern.
        if small_window:  # If the window is small, click twice at a specific position.  # Wenn das Fenster klein ist, wird zweimal an einer bestimmten Position geklickt.
            click(138, 124)  # Click at the coordinates (138, 124).  # Klickt an den Koordinaten (138, 124).
            click(138, 124)  # Click again at the same coordinates.  # Klickt erneut an denselben Koordinaten.
        else:  # If the window is not small, click twice at a different position.  # Wenn das Fenster nicht klein ist, wird zweimal an einer anderen Position geklickt.
            click(500, 390)  # Click at the coordinates (500, 390).  # Klickt an den Koordinaten (500, 390).
            click(500, 390)  # Click again at the same coordinates.  # Klickt erneut an denselben Koordinaten.

    def mouse_close_replay_window_tm20(small_window=False):  # Defines a function to close the replay window.  # Definiert eine Funktion, um das Wiederholungsfenster zu schließen.
        if small_window:  # If the window is small, click at a specific position.  # Wenn das Fenster klein ist, wird an einer bestimmten Position geklickt.
            click(130, 95)  # Click at the coordinates (130, 95).  # Klickt an den Koordinaten (130, 95).
        else:  # If the window is not small, click at another position.  # Wenn das Fenster nicht klein ist, wird an einer anderen Position geklickt.
            click(500, 280)  # Click at the coordinates (500, 280).  # Klickt an den Koordinaten (500, 280).
        mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.

    def mouse_save_replay_tm20(small_window=False):  # Defines a function to save the replay.  # Definiert eine Funktion, um die Wiederholung zu speichern.
        time.sleep(5.0)  # Wait for 5 seconds before proceeding.  # Wartet 5 Sekunden, bevor fortgefahren wird.
        if small_window:  # If the window is small, click at a specific position.  # Wenn das Fenster klein ist, wird an einer bestimmten Position geklickt.
            click(130, 110)  # Click at the coordinates (130, 110).  # Klickt an den Koordinaten (130, 110).
            mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.
            time.sleep(0.2)  # Wait for 0.2 seconds before proceeding.  # Wartet 0,2 Sekunden, bevor fortgefahren wird.
            click(130, 104)  # Click at the coordinates (130, 104).  # Klickt an den Koordinaten (130, 104).
            mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.
        else:  # If the window is not small, click at another position.  # Wenn das Fenster nicht klein ist, wird an einer anderen Position geklickt.
            click(500, 335)  # Click at the coordinates (500, 335).  # Klickt an den Koordinaten (500, 335).
            mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.
            time.sleep(0.2)  # Wait for 0.2 seconds before proceeding.  # Wartet 0,2 Sekunden, bevor fortgefahren wird.
            click(500, 310)  # Click at the coordinates (500, 310).  # Klickt an den Koordinaten (500, 310).
            mouseUp()  # Release the mouse button after clicking.  # Lässt die Maustaste nach dem Klicken los.

else:  # If the operating system is not Windows, define empty functions.  # Wenn das Betriebssystem nicht Windows ist, werden leere Funktionen definiert.

    def mouse_close_finish_pop_up_tm20(small_window=False):  # Empty function for non-Windows systems.  # Leere Funktion für Nicht-Windows-Systeme.
        pass  # No action for non-Windows systems.  # Keine Aktion für Nicht-Windows-Systeme.

    def mouse_change_name_replay_tm20(small_window=False):  # Empty function for non-Windows systems.  # Leere Funktion für Nicht-Windows-Systeme.
        pass  # No action for non-Windows systems.  # Keine Aktion für Nicht-Windows-Systeme.

    def mouse_save_replay_tm20(small_window=False):  # Empty function for non-Windows systems.  # Leere Funktion für Nicht-Windows-Systeme.
        pass  # No action for non-Windows systems.  # Keine Aktion für Nicht-Windows-Systeme.

    def mouse_close_replay_window_tm20(small_window=False):  # Empty function for non-Windows systems.  # Leere Funktion für Nicht-Windows-Systeme.
        pass  # No action for non-Windows systems.  # Keine Aktion für Nicht-Windows-Systeme.

if __name__ == "__main__":  # Check if the script is being run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    # standard library imports
    import time  # Import the time module to control timing in the script.  # Importiert das Zeit-Modul, um das Timing im Skript zu steuern.

    mouse_save_replay_tm20()  # Call the function to save the replay.  # Ruft die Funktion auf, um die Wiederholung zu speichern.
