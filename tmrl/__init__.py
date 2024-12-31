# logger (basicConfig must be called before importing anything)  # Initializes logging configuration.  # Logging-Konfiguration initialisieren.
import logging  # Imports the logging module for logging messages.  # Importiert das Logging-Modul für Nachrichtenprotokollierung.
import sys  # Imports the sys module to interact with the system.  # Importiert das sys-Modul zur Systeminteraktion.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Sets up basic logging to output messages to the console.  # Richtet grundlegendes Logging ein, um Nachrichten auf der Konsole auszugeben.

# fixes for Windows:  # Fixes specific to Windows systems.  # Korrekturen für Windows-Systeme.
import platform  # Imports the platform module to check the operating system.  # Importiert das platform-Modul, um das Betriebssystem zu prüfen.
if platform.system() == "Windows":  # Checks if the operating system is Windows.  # Überprüft, ob das Betriebssystem Windows ist.
    # fix pywin32 in case it fails to import:  # Handles pywin32 import errors.  # Behandelt Importfehler von pywin32.
    try:  
        import win32gui  # Tries to import the win32gui module for GUI interaction.  # Versucht, das win32gui-Modul für GUI-Interaktionen zu importieren.
        import win32ui  # Tries to import the win32ui module for UI components.  # Versucht, das win32ui-Modul für UI-Komponenten zu importieren.
        import win32con  # Tries to import the win32con module for constants.  # Versucht, das win32con-Modul für Konstanten zu importieren.
    except ImportError as e1:  # Catches ImportError if pywin32 fails.  # Fängt ImportError ab, falls pywin32 fehlschlägt.
        logging.info("pywin32 failed to import. Attempting to fix pywin32 installation...")  # Logs a message about pywin32 failure.  # Protokolliert eine Nachricht über den Fehler von pywin32.
        from tmrl.tools.init_package.init_pywin32 import fix_pywin32  # Imports a utility to fix pywin32.  # Importiert ein Werkzeug zur Reparatur von pywin32.
        try:
            fix_pywin32()  # Attempts to fix the pywin32 installation.  # Versucht, die pywin32-Installation zu reparieren.
            import win32gui  # Tries importing win32gui again after the fix.  # Versucht erneut, win32gui nach der Reparatur zu importieren.
            import win32ui  # Tries importing win32ui again after the fix.  # Versucht erneut, win32ui nach der Reparatur zu importieren.
            import win32con  # Tries importing win32con again after the fix.  # Versucht erneut, win32con nach der Reparatur zu importieren.
        except ImportError as e2:  # Handles any remaining ImportErrors.  # Behandelt verbleibende ImportErrors.
            logging.error(f"tmrl could not fix pywin32 on your system. The following exceptions were raised:\
            \n=== Exception 1 ===\n{str(e1)}\n=== Exception 2 ===\n{str(e2)}\
            \nPlease install pywin32 manually.")  # Logs an error with details about the failed fixes.  # Protokolliert einen Fehler mit Details zu den fehlgeschlagenen Reparaturen.
            raise RuntimeError("Please install pywin32 manually: https://github.com/mhammond/pywin32")  # Raises a RuntimeError if pywin32 cannot be fixed.  # Löst einen RuntimeError aus, falls pywin32 nicht repariert werden kann.

# TMRL folder initialization:  # Initializes the TMRL folder.  # Initialisiert den TMRL-Ordner.
from tmrl.tools.init_package.init_tmrl import TMRL_FOLDER  # Imports the TMRL folder setup module.  # Importiert das Modul zur Einrichtung des TMRL-Ordners.

# do not remove this  # Indicates this line should not be removed.  # Zeigt an, dass diese Zeile nicht entfernt werden soll.
from dataclasses import dataclass  # Imports dataclass for defining structured data.  # Importiert dataclass, um strukturierte Daten zu definieren.

from tmrl.envs import GenericGymEnv  # Imports the generic environment for Gymnasium.  # Importiert die generische Umgebung für Gymnasium.
from tmrl.config.config_objects import CONFIG_DICT  # Imports configuration data for the environment.  # Importiert Konfigurationsdaten für die Umgebung.

def get_environment():  # Defines a function to return the default environment.  # Definiert eine Funktion, um die Standardumgebung zurückzugeben.
    """  # Docstring describing the function.  # Docstring, der die Funktion beschreibt.
    Default TMRL Gymnasium environment for TrackMania 2020.  # English description of the environment.  # Englische Beschreibung der Umgebung.

    Returns:  # Indicates the return value of the function.  # Zeigt den Rückgabewert der Funktion an.
        gymnasium.Env: An instance of the default TMRL Gymnasium environment  # Returns an environment object.  # Gibt ein Umgebungsobjekt zurück.
    """
    import tmrl.config.config_constants as cfg  # Imports configuration constants.  # Importiert Konfigurationskonstanten.
    return GenericGymEnv(id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})  # Creates and returns the environment.  # Erstellt und gibt die Umgebung zurück.
