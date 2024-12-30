import os  # Import the os module for interacting with the operating system.  # Importiere das os-Modul zur Interaktion mit dem Betriebssystem.
import sys  # Import the sys module for system-specific parameters and functions.  # Importiere das sys-Modul für systemspezifische Parameter und Funktionen.
import glob  # Import the glob module for file pattern matching.  # Importiere das glob-Modul für das Dateimuster-Matching.
import shutil  # Import the shutil module for high-level file operations.  # Importiere das shutil-Modul für hochgradige Dateioperationen.
import sysconfig  # Import the sysconfig module for accessing Python configuration.  # Importiere das sysconfig-Modul zum Zugreifen auf die Python-Konfiguration.

try:  # Attempt to import winreg module to interact with Windows registry.  # Versuche, das winreg-Modul zu importieren, um mit der Windows-Registrierung zu arbeiten.
    import winreg as winreg  # Import winreg for working with the Windows registry.  # Importiere winreg, um mit der Windows-Registrierung zu arbeiten.
except:  # If importing winreg fails, handle the exception.  # Wenn der Import von winreg fehlschlägt, behandle die Ausnahme.
    import winreg  # Import winreg as a fallback.  # Importiere winreg als Ersatz.

import tempfile  # Import tempfile module for temporary file handling.  # Importiere das tempfile-Modul zur Handhabung von temporären Dateien.

tee_f = open(os.path.join(tempfile.gettempdir(), "pywin32_postinstall.log"), "w")  # Open a log file in the system's temp directory for writing.  # Öffne eine Logdatei im temporären Verzeichnis des Systems zum Schreiben.

class Tee:  # Define a class to redirect output to multiple locations.  # Definiere eine Klasse, um die Ausgabe an mehrere Orte umzuleiten.
    def __init__(self, file):  # Initialize the Tee class with a file to write to.  # Initialisiere die Tee-Klasse mit einer Datei zum Schreiben.
        self.f = file  # Store the file object.  # Speichere das Dateiobjekt.

    def write(self, what):  # Define the write method to redirect text to the file.  # Definiere die Methode write, um Text an die Datei umzuleiten.
        if self.f is not None:  # If a file object exists, write to it.  # Wenn ein Dateiobjekt vorhanden ist, schreibe in diese Datei.
            try:  # Try to write the content to the file.  # Versuche, den Inhalt in die Datei zu schreiben.
                self.f.write(what.replace("\n", "\r\n"))  # Replace newline characters with the appropriate line endings.  # Ersetze Zeilenumbruchzeichen mit den entsprechenden Zeilenenden.
            except IOError:  # Handle potential I/O errors.  # Behandle mögliche I/O-Fehler.
                pass  # Do nothing if there's an error.  # Tue nichts bei einem Fehler.

        tee_f.write(what)  # Write the same content to the log file.  # Schreibe denselben Inhalt in die Logdatei.

    def flush(self):  # Define the flush method to ensure all data is written.  # Definiere die Methode flush, um sicherzustellen, dass alle Daten geschrieben werden.
        if self.f is not None:  # If a file object exists, flush the data.  # Wenn ein Dateiobjekt vorhanden ist, spüle die Daten.
            try:  # Try to flush the file.  # Versuche, die Datei zu spülen.
                self.f.flush()  # Flush the file object.  # Spüle das Dateiobjekt.
            except IOError:  # Handle potential I/O errors.  # Behandle mögliche I/O-Fehler.
                pass  # Do nothing if there's an error.  # Tue nichts bei einem Fehler.
        tee_f.flush()  # Flush the log file.  # Spüle die Logdatei.

# For some unknown reason, when running under bdist_wininst we will start up
# with sys.stdout as None but stderr is hooked up. This work-around allows
# bdist_wininst to see the output we write and display it at the end of
# the install.
if sys.stdout is None:  # If sys.stdout is None, redirect output to stderr.  # Wenn sys.stdout None ist, leite die Ausgabe an stderr um.
    sys.stdout = sys.stderr  # Set sys.stdout to sys.stderr for output redirection.  # Setze sys.stdout auf sys.stderr für die Ausgabeumleitung.

sys.stderr = Tee(sys.stderr)  # Redirect stderr output to the Tee class.  # Leite die stderr-Ausgabe an die Tee-Klasse um.
sys.stdout = Tee(sys.stdout)  # Redirect stdout output to the Tee class.  # Leite die stdout-Ausgabe an die Tee-Klasse um.

com_modules = [  # Define a list of COM modules to be used.  # Definiere eine Liste von COM-Modulen, die verwendet werden sollen.
    ("win32com.servers.interp", "Interpreter"),  # Add module and class names.  # Füge Modul- und Klassennamen hinzu.
    ("win32com.servers.dictionary", "DictionaryPolicy"),  # Add module and class names.  # Füge Modul- und Klassennamen hinzu.
    ("win32com.axscript.client.pyscript", "PyScript"),  # Add module and class names.  # Füge Modul- und Klassennamen hinzu.
]

silent = 0  # Define a flag for silent installation (0 for no, 1 for yes).  # Definiere ein Flag für stille Installation (0 für nein, 1 für ja).
verbose = 1  # Define the verbosity level for output messages (0 for quiet, 1 for verbose).  # Definiere das Detailniveau der Ausgabemeldungen (0 für leise, 1 für detailliert).

root_key_name = "Software\\Python\\PythonCore\\" + sys.winver  # Define the registry key for the current Python version.  # Definiere den Registrierungs-Schlüssel für die aktuelle Python-Version.

try:  # Attempt to check if running inside the bdist_wininst installer.  # Versuche zu überprüfen, ob das Skript im bdist_wininst-Installer läuft.
    file_created  # Check for the file_created function.  # Überprüfe die Funktion file_created.
    is_bdist_wininst = True  # If successful, set the flag for bdist_wininst installer.  # Wenn erfolgreich, setze das Flag für den bdist_wininst-Installer.
except NameError:  # If file_created is not defined, handle the exception.  # Wenn file_created nicht definiert ist, behandle die Ausnahme.
    is_bdist_wininst = False  # Set the flag to False for non-bdist_wininst installation.  # Setze das Flag auf False für Nicht-bdist_wininst-Installationen.

    def file_created(file):  # Define the file_created function.  # Definiere die Funktion file_created.
        pass  # No action for non-bdist_wininst installer.  # Keine Aktion für Nicht-bdist_wininst-Installationen.

    def directory_created(directory):  # Define the directory_created function.  # Definiere die Funktion directory_created.
        pass  # No action for non-bdist_wininst installer.  # Keine Aktion für Nicht-bdist_wininst-Installationen.

    def get_root_hkey():  # Define a function to get the root registry key.  # Definiere eine Funktion, um den Haupt-Registrierungsschlüssel zu erhalten.
        try:  # Try to open the registry key.  # Versuche, den Registrierungsschlüssel zu öffnen.
            winreg.OpenKey(  # Open the registry key.  # Öffne den Registrierungsschlüssel.
                winreg.HKEY_LOCAL_MACHINE, root_key_name, 0, winreg.KEY_CREATE_SUB_KEY  # Specify parameters for key access.  # Gib Parameter für den Schlüsselausschnitt an.
            )
            return winreg.HKEY_LOCAL_MACHINE  # Return the HKEY_LOCAL_MACHINE if successful.  # Gib HKEY_LOCAL_MACHINE zurück, wenn erfolgreich.
        except OSError:  # If an error occurs, handle it.  # Wenn ein Fehler auftritt, behandle diesen.
            return winreg.HKEY_CURRENT_USER  # If failed, return HKEY_CURRENT_USER.  # Wenn fehlgeschlagen, gib HKEY_CURRENT_USER zurück.


try:  # Attempt to check if create_shortcut function exists.  # Versuche zu überprüfen, ob die Funktion create_shortcut existiert.
    create_shortcut  # Check if the create_shortcut function is available.  # Überprüfe, ob die Funktion create_shortcut verfügbar ist.
except NameError:  # If create_shortcut is not defined, handle the exception.  # Wenn create_shortcut nicht definiert ist, behandle die Ausnahme.
    def create_shortcut(  # Define a function to create a shortcut.  # Definiere eine Funktion zum Erstellen einer Verknüpfung.
        path, description, filename, arguments="", workdir="", iconpath="", iconindex=0  # Specify parameters for creating a shortcut.  # Gib Parameter zum Erstellen einer Verknüpfung an.
    ):
        import pythoncom  # Import pythoncom module for COM support.  # Importiere das pythoncom-Modul für COM-Unterstützung.
        from win32com.shell import shell  # Import shell module for working with Windows shell.  # Importiere das shell-Modul zur Arbeit mit der Windows-Shell.

        ilink = pythoncom.CoCreateInstance(  # Create a shell link object.  # Erstelle ein Shell-Link-Objekt.
            shell.CLSID_ShellLink,  # Specify the ShellLink CLSID.  # Gib die ShellLink CLSID an.
            None,  # No initialization required.  # Keine Initialisierung erforderlich.
            pythoncom.CLSCTX_INPROC_SERVER,  # Set context for server execution.  # Setze den Kontext für die Serverausführung.
            shell.IID_IShellLink,  # Specify the IShellLink interface.  # Gib das IShellLink-Interface an.
        )
        ilink.SetPath(path)  # Set the path for the shortcut.  # Setze den Pfad für die Verknüpfung.
        ilink.SetDescription(description)  # Set the description for the shortcut.  # Setze die Beschreibung für die Verknüpfung.
        if arguments:  # If arguments are provided, set them for the shortcut.  # Wenn Argumente angegeben sind, setze sie für die Verknüpfung.
            ilink.SetArguments(arguments)  # Set the arguments for the shortcut.  # Setze die Argumente für die Verknüpfung.
        if workdir:  # If a working directory is provided, set it.  # Wenn ein Arbeitsverzeichnis angegeben ist, setze es.
            ilink.SetWorkingDirectory(workdir)  # Set the working directory.  # Setze das Arbeitsverzeichnis.
        if iconpath or iconindex:  # If icon path or index is provided, set them.  # Wenn ein Icon-Pfad oder Index angegeben ist, setze diese.
            ilink.SetIconLocation(iconpath, iconindex)  # Set the icon location for the shortcut.  # Setze den Speicherort des Symbols für die Verknüpfung.
        ipf = ilink.QueryInterface(pythoncom.IID_IPersistFile)  # Query the IPersistFile interface.  # Frage das IPersistFile-Interface ab.
        ipf.Save(filename, 0)  # Save the shortcut to the specified filename.  # Speichere die Verknüpfung unter dem angegebenen Dateinamen.

    def get_special_folder_path(path_name):  # Define a function to get special folder paths.  # Definiere eine Funktion, um spezielle Ordnerspeicherorte zu erhalten.
        from win32com.shell import shell, shellcon  # Import shell and shellcon for accessing shell paths.  # Importiere shell und shellcon zum Zugriff auf Shell-Speicherorte.

        for maybe in """  # Loop through known special folder paths.  # Schleife durch bekannte spezielle Ordnerspeicherorte.
            CSIDL_COMMON_STARTMENU CSIDL_STARTMENU CSIDL_COMMON_APPDATA
            CSIDL_LOCAL_APPDATA CSIDL_APPDATA CSIDL_COMMON_DESKTOPDIRECTORY
            CSIDL_DESKTOPDIRECTORY CSIDL_COMMON_STARTUP CSIDL_STARTUP
            CSIDL_COMMON_PROGRAMS CSIDL_PROGRAMS CSIDL_PROGRAM_FILES_COMMON
            CSIDL_PROGRAM_FILES CSIDL_FONTS""".split():  # Iterate through the list of known special folder IDs.  # Iteriere durch die Liste bekannter spezieller Ordner-IDs.
            if maybe == path_name:  # If the path matches a known folder name, get its path.  # Wenn der Pfad mit einem bekannten Ordnernamen übereinstimmt, hole seinen Pfad.
                csidl = getattr(shellcon, maybe)  # Get the CSIDL constant for the folder.  # Hole die CSIDL-Konstante für den Ordner.
                return shell.SHGetSpecialFolderPath(0, csidl, False)  # Return the folder path.  # Gib den Ordnerspeicherort zurück.
        raise ValueError("%s is an unknown path ID" % (path_name,))  # Raise an error if the path name is unknown.  # Wirf einen Fehler, wenn der Pfadname unbekannt ist.

def CopyTo(desc, src, dest):  # Defines a function CopyTo with parameters desc (description), src (source), and dest (destination).  # Definiert eine Funktion CopyTo mit den Parametern desc (Beschreibung), src (Quelle) und dest (Ziel).
    import win32api, win32con  # Imports win32api and win32con modules to interact with Windows API.  # Importiert die Module win32api und win32con, um mit der Windows-API zu interagieren.

    while 1:  # Starts an infinite loop.  # Startet eine Endlosschleife.
        try:  # Tries to execute the following block of code.  # Versucht, den folgenden Codeblock auszuführen.
            win32api.CopyFile(src, dest, 0)  # Copies the file from the source to the destination using win32api.  # Kopiert die Datei von der Quelle zum Ziel mit win32api.
            return  # If the copy is successful, exits the function.  # Wenn das Kopieren erfolgreich ist, verlässt die Funktion.
        except win32api.error as details:  # Catches any errors raised by win32api.  # Fängt alle Fehler ab, die von win32api ausgelöst werden.
            if details.winerror == 5:  # access denied - user not admin.  # Überprüft, ob der Fehler "Zugriff verweigert" (Fehler 5) ist.
                raise  # Raises the error if access is denied.  # Löst den Fehler aus, wenn der Zugriff verweigert wird.
            if silent:  # Checks if silent mode is enabled.  # Überprüft, ob der stille Modus aktiviert ist.
                raise  # Re-raises the error in silent mode.  # Löst den Fehler im stillen Modus erneut aus.
            full_desc = (  # Constructs an error message with the description and error details.  # Erstellt eine Fehlermeldung mit der Beschreibung und den Fehlermeldungen.
                "Error %s\n\n"
                "If you have any Python applications running, "
                "please close them now\nand select 'Retry'\n\n%s"
                % (desc, details.strerror)  # Inserts description and error message into the formatted string.  # Fügt die Beschreibung und die Fehlermeldung in den formatierten String ein.
            )
            rc = win32api.MessageBox(  # Displays a message box with options.  # Zeigt ein Meldungsfenster mit Optionen an.
                0, full_desc, "Installation Error", win32con.MB_ABORTRETRYIGNORE  # The message box displays the error and options (Abort, Retry, Ignore).  # Das Meldungsfenster zeigt den Fehler und Optionen (Abbrechen, Wiederholen, Ignorieren) an.
            )
            if rc == win32con.IDABORT:  # If the user clicks "Abort", raises an error.  # Wenn der Benutzer auf "Abbrechen" klickt, wird ein Fehler ausgelöst.
                raise  # Raises the error if "Abort" is selected.  # Löst den Fehler aus, wenn "Abbrechen" ausgewählt wird.
            elif rc == win32con.IDIGNORE:  # If the user clicks "Ignore", return from the function.  # Wenn der Benutzer auf "Ignorieren" klickt, wird die Funktion verlassen.
                return  # Exits the function if "Ignore" is selected.  # Verlässt die Funktion, wenn "Ignorieren" ausgewählt wird.
            # else retry - around we go again.  # If "Retry" is selected, the loop continues.  # Wenn "Wiederholen" ausgewählt wird, geht die Schleife weiter.

def LoadSystemModule(lib_dir, modname):  # Defines a function to load a system module.  # Definiert eine Funktion, um ein Systemmodul zu laden.
    import importlib.util, importlib.machinery  # Imports modules for dynamic importing of system libraries.  # Importiert Module für das dynamische Laden von Systembibliotheken.

    suffix = "_d" if "_d.pyd" in importlib.machinery.EXTENSION_SUFFIXES else ""  # Checks if the current build is a debug build by looking for "_d.pyd".  # Überprüft, ob es sich bei der aktuellen Version um eine Debug-Version handelt, indem nach "_d.pyd" gesucht wird.
    filename = "%s%d%d%s.dll" % (  # Constructs the filename for the DLL.  # Erstellt den Dateinamen für die DLL.
        modname,  # Module name.  # Modulname.
        sys.version_info[0],  # Python major version.  # Python-Hauptversion.
        sys.version_info[1],  # Python minor version.  # Python-Neben-Version.
        suffix,  # Debug suffix if applicable.  # Debug-Suffix, falls zutreffend.
    )
    filename = os.path.join(lib_dir, "pywin32_system32", filename)  # Constructs the full path for the DLL.  # Erstellt den vollständigen Pfad für die DLL.
    loader = importlib.machinery.ExtensionFileLoader(modname, filename)  # Creates a loader to load the extension file.  # Erstellt einen Loader, um die Erweiterungsdatei zu laden.
    spec = importlib.machinery.ModuleSpec(name=modname, loader=loader, origin=filename)  # Creates a module specification for the loader.  # Erstellt eine Modulspezifikation für den Loader.
    mod = importlib.util.module_from_spec(spec)  # Loads the module using the specification.  # Lädt das Modul mit der Spezifikation.
    spec.loader.exec_module(mod)  # Executes the module loading.  # Führt das Laden des Moduls aus.

def SetPyKeyVal(key_name, value_name, value):  # Defines a function to set a registry key value.  # Definiert eine Funktion, um einen Registrierungswert festzulegen.
    root_hkey = get_root_hkey()  # Gets the root registry key.  # Holt den Stamm-Registrierungsschlüssel.
    root_key = winreg.OpenKey(root_hkey, root_key_name)  # Opens the registry key using the root key.  # Öffnet den Registrierungskey mit dem Stamm-Schlüssel.
    try:  # Tries to execute the following block of code.  # Versucht, den folgenden Codeblock auszuführen.
        my_key = winreg.CreateKey(root_key, key_name)  # Creates a new registry key.  # Erstellt einen neuen Registrierungskey.
        try:  # Tries to execute the following block of code.  # Versucht, den folgenden Codeblock auszuführen.
            winreg.SetValueEx(my_key, value_name, 0, winreg.REG_SZ, value)  # Sets the value of the registry key.  # Setzt den Wert des Registrierungswerts.
            if verbose:  # If verbose mode is enabled.  # Wenn der ausführliche Modus aktiviert ist.
                print("-> %s\\%s[%s]=%r" % (root_key_name, key_name, value_name, value))  # Prints the registry key and value.  # Gibt den Registrierungskey und Wert aus.
        finally:  # Ensures the key is closed after use.  # Stellt sicher, dass der Schlüssel nach der Verwendung geschlossen wird.
            my_key.Close()  # Closes the created registry key.  # Schließt den erstellten Registrierungskey.
    finally:  # Ensures the root key is closed after use.  # Stellt sicher, dass der Stamm-Schlüssel nach der Verwendung geschlossen wird.
        root_key.Close()  # Closes the root registry key.  # Schließt den Stamm-Registrierungskey.

def UnsetPyKeyVal(key_name, value_name, delete_key=False):  # Defines a function to unset (delete) a registry key value.  # Definiert eine Funktion zum Löschen eines Registrierungswerts.
    root_hkey = get_root_hkey()  # Gets the root registry key.  # Holt den Stamm-Registrierungsschlüssel.
    root_key = winreg.OpenKey(root_hkey, root_key_name)  # Opens the root registry key.  # Öffnet den Stamm-Registrierungsschlüssel.
    try:  # Tries to execute the following block of code.  # Versucht, den folgenden Codeblock auszuführen.
        my_key = winreg.OpenKey(root_key, key_name, 0, winreg.KEY_SET_VALUE)  # Opens the specified registry key.  # Öffnet den angegebenen Registrierungskey.
        try:  # Tries to execute the following block of code.  # Versucht, den folgenden Codeblock auszuführen.
            winreg.DeleteValue(my_key, value_name)  # Deletes the specified registry value.  # Löscht den angegebenen Registrierungswert.
            if verbose:  # If verbose mode is enabled.  # Wenn der ausführliche Modus aktiviert ist.
                print("-> DELETE %s\\%s[%s]" % (root_key_name, key_name, value_name))  # Prints the deletion of the registry value.  # Gibt die Löschung des Registrierungswerts aus.
        finally:  # Ensures the key is closed after use.  # Stellt sicher, dass der Schlüssel nach der Verwendung geschlossen wird.
            my_key.Close()  # Closes the registry key.  # Schließt den Registrierungskey.
        if delete_key:  # If the delete_key flag is set to True.  # Wenn das Flag delete_key auf True gesetzt ist.
            winreg.DeleteKey(root_key, key_name)  # Deletes the specified registry key.  # Löscht den angegebenen Registrierungskey.
            if verbose:  # If verbose mode is enabled.  # Wenn der ausführliche Modus aktiviert ist.
                print("-> DELETE %s\\%s" % (root_key_name, key_name))  # Prints the deletion of the registry key.  # Gibt die Löschung des Registrierungskeys aus.
    except OSError as why:  # Catches any OSError.  # Fängt alle OSError ab.
        winerror = getattr(why, "winerror", why.errno)  # Retrieves the error code from the exception.  # Holt den Fehlercode aus der Ausnahme.
        if winerror != 2:  # file not found.  # Überprüft, ob der Fehlercode nicht 2 (Datei nicht gefunden) ist.
            raise  # Raises the error if the error code is not 2.  # Löst den Fehler aus, wenn der Fehlercode nicht 2 ist.
    finally:  # Ensures the root key is closed after use.  # Stellt sicher, dass der Stamm-Schlüssel nach der Verwendung geschlossen wird.
        root_key.Close()  # Closes the root registry key.  # Schließt den Stamm-Registrierungskey.

def RegisterCOMObjects(register=True):  # Define the function RegisterCOMObjects that takes an argument register with a default value of True.  # Definiert die Funktion RegisterCOMObjects, die ein Argument register mit einem Standardwert von True hat.
    import win32com.server.register  # Import the necessary module for COM object registration.  # Importiert das notwendige Modul für die Registrierung von COM-Objekten.

    if register:  # Check if the register argument is True.  # Überprüft, ob das Argument register True ist.
        func = win32com.server.register.RegisterClasses  # Set the function to RegisterClasses if registering.  # Setzt die Funktion auf RegisterClasses, wenn registriert wird.
    else:  # If not registering, do the opposite.  # Wenn nicht registriert wird, wird das Gegenteil gemacht.
        func = win32com.server.register.UnregisterClasses  # Set the function to UnregisterClasses if unregistering.  # Setzt die Funktion auf UnregisterClasses, wenn de-registriert wird.
    flags = {}  # Initialize an empty dictionary for flags.  # Initialisiert ein leeres Wörterbuch für Flags.
    if not verbose:  # Check if the verbose flag is not set.  # Überprüft, ob das verbose-Flag nicht gesetzt ist.
        flags["quiet"] = 1  # Set the quiet flag to suppress unnecessary output.  # Setzt das quiet-Flag, um unnötige Ausgaben zu unterdrücken.
    for module, klass_name in com_modules:  # Iterate over each module and class name in the com_modules list.  # Iteriert über jedes Modul und Klassennamen in der Liste com_modules.
        __import__(module)  # Dynamically import the module.  # Importiert das Modul dynamisch.
        mod = sys.modules[module]  # Access the imported module from sys.modules.  # Greift auf das importierte Modul aus sys.modules zu.
        flags["finalize_register"] = getattr(mod, "DllRegisterServer", None)  # Get the DllRegisterServer function if it exists.  # Ruft die Funktion DllRegisterServer ab, wenn sie existiert.
        flags["finalize_unregister"] = getattr(mod, "DllUnregisterServer", None)  # Get the DllUnregisterServer function if it exists.  # Ruft die Funktion DllUnregisterServer ab, wenn sie existiert.
        klass = getattr(mod, klass_name)  # Get the class from the module using the class name.  # Ruft die Klasse aus dem Modul anhand des Klassennamens ab.
        func(klass, **flags)  # Call the register or unregister function with the class and flags.  # Ruft die Registrierungs- oder Deregistrierungsfunktion mit der Klasse und den Flags auf.

def RegisterHelpFile(register=True, lib_dir=None):  # Define the function RegisterHelpFile to register or unregister a help file.  # Definiert die Funktion RegisterHelpFile, um eine Hilfsdatei zu registrieren oder zu deregistrieren.
    if lib_dir is None:  # Check if the lib_dir argument is not provided.  # Überprüft, ob das Argument lib_dir nicht angegeben wurde.
        lib_dir = sysconfig.get_paths()["platlib"]  # Set lib_dir to the platform's library directory if not provided.  # Setzt lib_dir auf das Bibliotheksverzeichnis der Plattform, wenn es nicht angegeben wurde.
    if register:  # Check if registering the help file.  # Überprüft, ob die Hilfsdatei registriert werden soll.
        chm_file = os.path.join(lib_dir, "PyWin32.chm")  # Define the path to the help file.  # Definiert den Pfad zur Hilfsdatei.
        if os.path.isfile(chm_file):  # Check if the help file exists.  # Überprüft, ob die Hilfsdatei existiert.
            SetPyKeyVal("Help", None, None)  # Set the help key to None (initialization).  # Setzt den Help-Schlüssel auf None (Initialisierung).
            SetPyKeyVal("Help\\Pythonwin Reference", None, chm_file)  # Register the help file in the registry.  # Registriert die Hilfsdatei in der Registrierung.
            return chm_file  # Return the path to the help file.  # Gibt den Pfad zur Hilfsdatei zurück.
        else:  # If the help file is not found, print a note.  # Wenn die Hilfsdatei nicht gefunden wird, gibt es eine Nachricht aus.
            print("NOTE: PyWin32.chm can not be located, so has not been registered")  # Print a warning message.  # Gibt eine Warnmeldung aus.
    else:  # If not registering, unregister the help file.  # Wenn nicht registriert, deregistriert die Hilfsdatei.
        UnsetPyKeyVal("Help\\Pythonwin Reference", None, delete_key=True)  # Remove the help file registry entry.  # Entfernt den Registrierungseintrag der Hilfsdatei.
    return None  # Return None if the help file is not registered.  # Gibt None zurück, wenn die Hilfsdatei nicht registriert ist.

def RegisterPythonwin(register=True, lib_dir=None):  # Define the function RegisterPythonwin to add or remove Pythonwin from the context menu.  # Definiert die Funktion RegisterPythonwin, um Pythonwin zum Kontextmenü hinzuzufügen oder zu entfernen.
    import os  # Import the os module for file and path operations.  # Importiert das Modul os für Datei- und Pfadoperationen.

    if lib_dir is None:  # Check if the lib_dir argument is not provided.  # Überprüft, ob das Argument lib_dir nicht angegeben wurde.
        lib_dir = sysconfig.get_paths()["platlib"]  # Set lib_dir to the platform's library directory if not provided.  # Setzt lib_dir auf das Bibliotheksverzeichnis der Plattform, wenn es nicht angegeben wurde.
    classes_root = get_root_hkey()  # Get the root registry key.  # Ruft den Stamm-Registrierungsschlüssel ab.
    pythonwin_exe = os.path.join(lib_dir, "Pythonwin", "Pythonwin.exe")  # Define the path to Pythonwin executable.  # Definiert den Pfad zur Pythonwin-Executable.
    pythonwin_edit_command = pythonwin_exe + ' -edit "%1"'  # Define the command to edit Python files with Pythonwin.  # Definiert den Befehl, um Python-Dateien mit Pythonwin zu bearbeiten.

keys_vals = [  # Define a list of registry key-value pairs to add to the context menu.  # Definiert eine Liste von Registrierungsschlüssel-Wert-Paaren, die dem Kontextmenü hinzugefügt werden sollen.
    (
        "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Pythonwin.exe",  # Path to Pythonwin executable in the registry.  # Pfad zur Pythonwin-Executable in der Registrierung.
        "",
        pythonwin_exe,  # Value of the key, representing the location of the Pythonwin executable.  # Wert des Schlüssels, der den Speicherort der Pythonwin-Executable darstellt.
    ),
    (
        "Software\\Classes\\Python.File\\shell\\Edit with Pythonwin",  # Registry key for Python file context menu.  # Registrierungsschlüssel für das Kontextmenü der Python-Datei.
        "command",  # Specifies that this registry entry will define a command to run.  # Gibt an, dass dieser Registrierungseintrag einen auszuführenden Befehl definiert.
        pythonwin_edit_command,  # The command to execute when selecting "Edit with Pythonwin" in the context menu.  # Der Befehl, der ausgeführt wird, wenn "Mit Pythonwin bearbeiten" im Kontextmenü ausgewählt wird.
    ),
    (
        "Software\\Classes\\Python.NoConFile\\shell\\Edit with Pythonwin",  # Registry key for Python files without extensions.  # Registrierungsschlüssel für Python-Dateien ohne Erweiterung.
        "command",  # Specifies that this registry entry will define a command to run.  # Gibt an, dass dieser Registrierungseintrag einen auszuführenden Befehl definiert.
        pythonwin_edit_command,  # The command to execute when selecting "Edit with Pythonwin" for files without extensions.  # Der Befehl, der ausgeführt wird, wenn "Mit Pythonwin bearbeiten" für Dateien ohne Erweiterung ausgewählt wird.
    ),
]


    try:
        if register:  # If registering, create the registry keys.  # Wenn registriert wird, erstellt die Registrierungsschlüssel.
            for key, sub_key, val in keys_vals:  # Iterate through the registry key-value pairs.  # Iteriert durch die Registrierungsschlüssel-Wert-Paare.
                hkey = winreg.CreateKey(classes_root, key)  # Create or open the registry key.  # Erstellt oder öffnet den Registrierungsschlüssel.
                if sub_key:  # If there is a sub-key, create it.  # Wenn ein Unter-Schlüssel vorhanden ist, wird dieser erstellt.
                    hkey = winreg.CreateKey(hkey, sub_key)  # Create the sub-key.  # Erstellt den Unter-Schlüssel.
                winreg.SetValueEx(hkey, None, 0, winreg.REG_SZ, val)  # Set the value in the registry.  # Setzt den Wert in der Registrierung.
                hkey.Close()  # Close the registry key.  # Schließt den Registrierungsschlüssel.
        else:  # If unregistering, delete the registry keys.  # Wenn de-registriert wird, löscht die Registrierungsschlüssel.
            for key, sub_key, val in keys_vals:  # Iterate through the registry key-value pairs.  # Iteriert durch die Registrierungsschlüssel-Wert-Paare.
                try:
                    if sub_key:  # If there is a sub-key, delete it.  # Wenn ein Unter-Schlüssel vorhanden ist, wird dieser gelöscht.
                        hkey = winreg.OpenKey(classes_root, key)  # Open the registry key.  # Öffnet den Registrierungsschlüssel.
                        winreg.DeleteKey(hkey, sub_key)  # Delete the sub-key.  # Löscht den Unter-Schlüssel.
                        hkey.Close()  # Close the registry key.  # Schließt den Registrierungsschlüssel.
                    winreg.DeleteKey(classes_root, key)  # Delete the registry key.  # Löscht den Registrierungsschlüssel.
                except OSError as why:  # If an error occurs, handle it.  # Wenn ein Fehler auftritt, wird dieser behandelt.
                    winerror = getattr(why, "winerror", why.errno)  # Get the error code.  # Holt den Fehlercode.
                    if winerror != 2:  # If the error is not "file not found", raise the error.  # Wenn der Fehler nicht "Datei nicht gefunden" ist, wird der Fehler ausgelöst.
                        raise
    finally:
        # Notify Windows about the registry changes.  # Benachrichtigt Windows über die Änderungen in der Registrierung.
        from win32com.shell import shell, shellcon  # Import the necessary modules for Windows shell notifications.  # Importiert die notwendigen Module für Windows-Shell-Benachrichtigungen.

        shell.SHChangeNotify(  # Notify the system about the changes.  # Benachrichtigt das System über die Änderungen.
            shellcon.SHCNE_ASSOCCHANGED, shellcon.SHCNF_IDLIST, None, None
        )

def get_shortcuts_folder():  # Function to determine the folder for shortcuts.  # Funktion zur Bestimmung des Ordners für Verknüpfungen.
    if get_root_hkey() == winreg.HKEY_LOCAL_MACHINE:  # Check if root key is HKEY_LOCAL_MACHINE (admin access).  # Überprüfen, ob der Root-Schlüssel HKEY_LOCAL_MACHINE (Admin-Rechte) ist.
        try:
            fldr = get_special_folder_path("CSIDL_COMMON_PROGRAMS")  # Get path for common programs folder.  # Hole den Pfad für den Ordner "Gemeinsame Programme".
        except OSError:  # Handle exception if the folder is not found.  # Behandeln der Ausnahme, falls der Ordner nicht gefunden wird.
            fldr = get_special_folder_path("CSIDL_PROGRAMS")  # Fallback to the standard programs folder.  # Fallback zum Standard-Programme-Ordner.
    else:  # If not running with admin rights.  # Wenn keine Administratorrechte vorhanden sind.
        fldr = get_special_folder_path("CSIDL_PROGRAMS")  # Always get the standard programs folder for non-admin installs.  # Immer den Standard-Programme-Ordner für Nicht-Admin-Installationen.

    try:
        install_group = winreg.QueryValue(  # Query the registry for the install group.  # Abfragen des Registrierungseintrags für die Installationsgruppe.
            get_root_hkey(), root_key_name + "\\InstallPath\\InstallGroup"  # Registry key for the install group path.  # Registrierungsschlüssel für den Installationsgruppenpfad.
        )
    except OSError:  # Handle error if registry query fails.  # Fehlerbehandlung, wenn die Registrierungabfrage fehlschlägt.
        vi = sys.version_info  # Get Python version information.  # Hole Informationen zur Python-Version.
        install_group = "Python %d.%d" % (vi[0], vi[1])  # Set install group name based on Python version.  # Setze den Namen der Installationsgruppe basierend auf der Python-Version.
    return os.path.join(fldr, install_group)  # Return the full path to the shortcuts folder.  # Gib den vollständigen Pfad zum Verknüpfungsordner zurück.

# Get the system directory, which may be the Wow64 directory if we are a 32bit
# python on a 64bit OS.  # Hole das Systemverzeichnis, das möglicherweise das Wow64-Verzeichnis ist, wenn wir eine 32-Bit-Python-Version auf einem 64-Bit-Betriebssystem haben.
def get_system_dir():  # Function to get the system directory path.  # Funktion zum Abrufen des Systemverzeichnispfads.
    import win32api  # Importing win32api module.  # Importiere das win32api-Modul.

    try:
        import pythoncom  # Import pythoncom for COM support.  # Importiere pythoncom für COM-Unterstützung.
        import win32process  # Import win32process for process handling.  # Importiere win32process für die Prozessverarbeitung.
        from win32com.shell import shell, shellcon  # Import shell modules for Windows shell access.  # Importiere die Shell-Module für den Zugriff auf die Windows-Shell.

        try:
            if win32process.IsWow64Process():  # Check if the current process is Wow64.  # Überprüfen, ob der aktuelle Prozess ein Wow64-Prozess ist.
                return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEMX86)  # Return the system directory for 32-bit on 64-bit OS.  # Gib das Systemverzeichnis für 32-Bit auf einem 64-Bit-Betriebssystem zurück.
            return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEM)  # Return the system directory for 64-bit OS.  # Gib das Systemverzeichnis für ein 64-Bit-Betriebssystem zurück.
        except (pythoncom.com_error, win32process.error):  # Handle exceptions during process check.  # Behandeln von Ausnahmen bei der Prozessprüfung.
            return win32api.GetSystemDirectory()  # Fallback to win32api for system directory.  # Fallback auf win32api für das Systemverzeichnis.
    except ImportError:  # Handle the case if imports fail.  # Behandeln des Falls, dass Imports fehlschlagen.
        return win32api.GetSystemDirectory()  # Fallback to win32api for system directory.  # Fallback auf win32api für das Systemverzeichnis.

def fixup_dbi():  # Function to fix issues with old dbi.pyd files.  # Funktion zur Behebung von Problemen mit alten dbi.pyd-Dateien.
    import win32api, win32con  # Import necessary win32 modules.  # Importiere notwendige win32-Module.

    pyd_name = os.path.join(os.path.dirname(win32api.__file__), "dbi.pyd")  # Path for dbi.pyd file.  # Pfad zur dbi.pyd-Datei.
    pyd_d_name = os.path.join(os.path.dirname(win32api.__file__), "dbi_d.pyd")  # Path for dbi_d.pyd file.  # Pfad zur dbi_d.pyd-Datei.
    py_name = os.path.join(os.path.dirname(win32con.__file__), "dbi.py")  # Path for dbi.py file.  # Pfad zur dbi.py-Datei.
    for this_pyd in (pyd_name, pyd_d_name):  # Loop through both pyd files.  # Schleife durch beide pyd-Dateien.
        this_dest = this_pyd + ".old"  # Set destination path for old pyd file.  # Setze Zielpfad für die alte pyd-Datei.
        if os.path.isfile(this_pyd) and os.path.isfile(py_name):  # Check if the pyd file and the py file exist.  # Überprüfen, ob die pyd- und die py-Datei existieren.
            try:
                if os.path.isfile(this_dest):  # If the old file already exists, delete the new one.  # Wenn die alte Datei bereits existiert, lösche die neue.
                    print(
                        "Old dbi '%s' already exists - deleting '%s'"
                        % (this_dest, this_pyd)
                    )  # Print message that the old file exists.  # Zeige Nachricht an, dass die alte Datei existiert.
                    os.remove(this_pyd)  # Remove the new pyd file.  # Lösche die neue pyd-Datei.
                else:
                    os.rename(this_pyd, this_dest)  # Rename the pyd file to mark it as old.  # Benenne die pyd-Datei um, um sie als alt zu kennzeichnen.
                    print("renamed '%s'->'%s.old'" % (this_pyd, this_pyd))  # Print message about renaming.  # Zeige Nachricht über das Umbenennen an.
                    file_created(this_pyd + ".old")  # Log the file creation.  # Protokolliere die Dateierstellung.
            except os.error as exc:  # Handle any file system errors.  # Behandle Fehler im Dateisystem.
                print("FAILED to rename '%s': %s" % (this_pyd, exc))  # Print error message if renaming fails.  # Zeige Fehlermeldung an, wenn das Umbenennen fehlschlägt.


def install(lib_dir):  # Defines the install function, taking 'lib_dir' as a parameter where library files will be installed.  # Definiert die Installationsfunktion, die 'lib_dir' als Parameter verwendet, in dem die Bibliotheksdateien installiert werden.

    import traceback  # Imports the traceback module to capture detailed error information.  # Importiert das traceback-Modul, um detaillierte Fehlerinformationen zu erfassen.

    # The .pth file is now installed as a regular file.  
    # Create the .pth file in the site-packages dir, and use only relative paths
    # We used to write a .pth directly to sys.prefix - clobber it.
    if os.path.isfile(os.path.join(sys.prefix, "pywin32.pth")):  # Checks if a file "pywin32.pth" exists in the system's prefix directory.  # Überprüft, ob eine Datei "pywin32.pth" im Präfixverzeichnis des Systems existiert.
        os.unlink(os.path.join(sys.prefix, "pywin32.pth"))  # Deletes the "pywin32.pth" file if it exists.  # Löscht die Datei "pywin32.pth", wenn sie existiert.

    # The .pth may be new and therefore not loaded in this session.
    # Setup the paths just in case.
    for name in "win32 win32\\lib Pythonwin".split():  # Iterates through the directories that need to be added to the system path.  # Iteriert durch die Verzeichnisse, die dem Systempfad hinzugefügt werden müssen.
        sys.path.append(os.path.join(lib_dir, name))  # Appends each directory to sys.path.  # Fügt jedes Verzeichnis zu sys.path hinzu.

    # It is possible people with old versions installed with still have
    # pywintypes and pythoncom registered.  We no longer need this, and stale
    # entries hurt us.
    for name in "pythoncom pywintypes".split():  # Iterates over the old registered modules that are no longer needed.  # Iteriert über die alten registrierten Module, die nicht mehr benötigt werden.
        keyname = "Software\\Python\\PythonCore\\" + sys.winver + "\\Modules\\" + name  # Constructs the registry key name for the module.  # Konstruiert den Registrierungs-Schlüsselname für das Modul.
        for root in winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER:  # Iterates through both machine-level and user-level registry roots.  # Iteriert durch die Maschinen- und Benutzer-Registry-Wurzeln.
            try:
                winreg.DeleteKey(root, keyname + "\\Debug")  # Attempts to delete the debug registry key.  # Versucht, den Debug-Registrierungsschlüssel zu löschen.
            except WindowsError:  # Catches any WindowsError exceptions during the deletion.  # Fängt WindowsError-Ausnahmen während des Löschens ab.
                pass  # Continues if the key does not exist.  # Setzt fort, wenn der Schlüssel nicht existiert.
            try:
                winreg.DeleteKey(root, keyname)  # Attempts to delete the main registry key for the module.  # Versucht, den Hauptregistrierungsschlüssel für das Modul zu löschen.
            except WindowsError:  # Catches any WindowsError exceptions during the deletion.  # Fängt WindowsError-Ausnahmen während des Löschens ab.
                pass  # Continues if the key does not exist.  # Setzt fort, wenn der Schlüssel nicht existiert.

    LoadSystemModule(lib_dir, "pywintypes")  # Loads the pywintypes system module.  # Lädt das Systemmodul pywintypes.
    LoadSystemModule(lib_dir, "pythoncom")  # Loads the pythoncom system module.  # Lädt das Systemmodul pythoncom.
    import win32api  # Imports the win32api module.  # Importiert das Modul win32api.

    # and now we can get the system directory:
    files = glob.glob(os.path.join(lib_dir, "pywin32_system32\\*.*"))  # Retrieves all files in the "pywin32_system32" directory.  # Ruft alle Dateien im Verzeichnis "pywin32_system32" ab.
    if not files:  # Checks if no files are found.  # Überprüft, ob keine Dateien gefunden wurden.
        raise RuntimeError("No system files to copy!!")  # Raises an error if no files are found.  # Löst einen Fehler aus, wenn keine Dateien gefunden werden.

    # Try the system32 directory first - if that fails due to "access denied",
    # it implies a non-admin user, and we use sys.prefix
    for dest_dir in [get_system_dir(), sys.prefix]:  # Iterates through directories to copy the files to.  # Iteriert durch Verzeichnisse, in die die Dateien kopiert werden sollen.
        # and copy some files over there
        worked = 0  # Initializes a variable to track if the copy operation was successful.  # Initialisiert eine Variable, um zu verfolgen, ob die Kopieroperation erfolgreich war.
        try:
            for fname in files:  # Iterates over the list of files to copy them.  # Iteriert über die Liste der Dateien, um sie zu kopieren.
                base = os.path.basename(fname)  # Extracts the file's base name.  # Extrahiert den Basisnamen der Datei.
                dst = os.path.join(dest_dir, base)  # Constructs the destination file path.  # Konstruiert den Ziel-Dateipfad.
                CopyTo("installing %s" % base, fname, dst)  # Copies the file to the destination.  # Kopiert die Datei zum Ziel.
                if verbose:  # If verbosity is enabled, prints the copy status.  # Wenn die Detailansicht aktiviert ist, wird der Kopierstatus ausgegeben.
                    print("Copied %s to %s" % (base, dst))  # Prints the source and destination paths.  # Gibt den Quell- und Zielpfad aus.
                file_created(dst)  # Registers the file as created.  # Registriert die Datei als erstellt.
                worked = 1  # Marks the copy operation as successful.  # Markiert die Kopieroperation als erfolgreich.
                # Nuke any other versions that may exist - having
                # duplicates causes major headaches.
                bad_dest_dirs = [
                    os.path.join(sys.prefix, "Library\\bin"),
                    os.path.join(sys.prefix, "Lib\\site-packages\\win32"),
                ]  # Specifies directories where duplicates of the file should be deleted.  # Gibt Verzeichnisse an, in denen Duplikate der Datei gelöscht werden sollen.
                if dest_dir != sys.prefix:  # Checks if the destination directory is not the system prefix.  # Überprüft, ob das Zielverzeichnis nicht das Systempräfix ist.
                    bad_dest_dirs.append(sys.prefix)  # Adds the system prefix to the list of bad destination directories.  # Fügt das Systempräfix zur Liste der ungültigen Zielverzeichnisse hinzu.
                for bad_dest_dir in bad_dest_dirs:  # Iterates through the bad destination directories to delete duplicates.  # Iteriert durch die ungültigen Zielverzeichnisse, um Duplikate zu löschen.
                    bad_fname = os.path.join(bad_dest_dir, base)  # Constructs the path to the duplicate file.  # Konstruiert den Pfad zur Duplikatdatei.
                    if os.path.exists(bad_fname):  # Checks if a duplicate file exists.  # Überprüft, ob eine Duplikatdatei existiert.
                        # let exceptions go here - delete must succeed
                        os.unlink(bad_fname)  # Deletes the duplicate file.  # Löscht die Duplikatdatei.
            if worked:  # If the copy operation was successful, breaks the loop.  # Wenn die Kopieroperation erfolgreich war, wird die Schleife beendet.
                break
        except win32api.error as details:  # Catches win32api-specific errors.  # Fängt win32api-spezifische Fehler ab.
            if details.winerror == 5:  # Checks if the error is "access denied".  # Überprüft, ob der Fehler "Zugriff verweigert" ist.
                # access denied - user not admin - try sys.prefix dir,
                # but first check that a version doesn't already exist
                # in that place - otherwise that one will still get used!
                if os.path.exists(dst):  # Checks if the destination file already exists.  # Überprüft, ob die Zieldatei bereits existiert.
                    msg = (
                        "The file '%s' exists, but can not be replaced "
                        "due to insufficient permissions.  You must "
                        "reinstall this software as an Administrator" % dst
                    )  # Constructs an error message if the file cannot be replaced.  # Konstruiert eine Fehlermeldung, wenn die Datei nicht ersetzt werden kann.
                    print(msg)  # Prints the error message.  # Gibt die Fehlermeldung aus.
                    raise RuntimeError(msg)  # Raises a runtime error with the message.  # Löst einen Laufzeitfehler mit der Nachricht aus.
                continue  # Continues to the next destination directory.  # Setzt mit dem nächsten Zielverzeichnis fort.
            raise  # Raises any other errors.  # Löst alle anderen Fehler aus.

    else:  # If the loop completes without success, raises an error.  # Wenn die Schleife ohne Erfolg abgeschlossen wird, wird ein Fehler ausgelöst.
        raise RuntimeError(
            "You don't have enough permissions to install the system files"
        )  # Raises a runtime error if the user doesn't have sufficient permissions.  # Löst einen Laufzeitfehler aus, wenn der Benutzer nicht über ausreichende Berechtigungen verfügt.

    # Pythonwin 'compiles' config files - record them for uninstall.
    pywin_dir = os.path.join(lib_dir, "Pythonwin", "pywin")  # Specifies the directory where config files are located.  # Gibt das Verzeichnis an, in dem sich Konfigurationsdateien befinden.
    for fname in glob.glob(os.path.join(pywin_dir, "*.cfg")):  # Iterates through the config files in the directory.  # Iteriert durch die Konfigurationsdateien im Verzeichnis.
        file_created(fname[:-1] + "c")  # Registers the .cfg file as a .cfc file for uninstallation.  # Registriert die .cfg-Datei als .cfc-Datei für die Deinstallation.

    # Register our demo COM objects.
    try:  # Tries to register demo COM objects.  # Versucht, Demo COM-Objekte zu registrieren.
        try:
            RegisterCOMObjects()  # Registers the COM objects.  # Registriert die COM-Objekte.
        except win32api.error as details:  # Handles errors related to COM object registration.  # Behandelt Fehler im Zusammenhang mit der Registrierung von COM-Objekten.
            if details.winerror != 5:  # ERROR_ACCESS_DENIED  # If the error is not "access denied", re-raises the error.  # Wenn der Fehler nicht "Zugriff verweigert" ist, wird der Fehler erneut ausgelöst.
                raise
            print("You do not have the permissions to install COM objects.")  # Prints a message if the user doesn't have permissions.  # Gibt eine Nachricht aus, wenn der Benutzer keine Berechtigungen hat.
            print("The sample COM objects were not registered.")  # Prints a message if COM objects are not registered.  # Gibt eine Nachricht aus, wenn COM-Objekte nicht registriert wurden.
    except Exception:  # Catches any other exceptions.  # Fängt alle anderen Ausnahmen ab.
        print("FAILED to register the Python COM objects")  # Prints a failure message.  # Gibt eine Fehlermeldung aus.
        traceback.print_exc()  # Prints the full traceback of the exception.  # Gibt die vollständige Fehlermeldung aus.

    # There may be no main Python key in HKCU if, eg, an admin installed
    # python itself.
    winreg.CreateKey(get_root_hkey(), root_key_name)  # Creates a registry key if it does not exist.  # Erstellt einen Registrierungs-Schlüssel, wenn er nicht existiert.

    chm_file = None  # Initializes the chm_file variable.  # Initialisiert die chm_file-Variable.
    try:  # Tries to register the help file.  # Versucht, die Hilfedatei zu registrieren.
        chm_file = RegisterHelpFile(True, lib_dir)  # Registers the help file and stores the result in chm_file.  # Registriert die Hilfedatei und speichert das Ergebnis in chm_file.
    except Exception:  # If an error occurs, prints an error message.  # Wenn ein Fehler auftritt, wird eine Fehlermeldung ausgegeben.
        print("Failed to register help file")  # Prints a message indicating failure to register the help file.  # Gibt eine Nachricht aus, die auf das Fehlschlagen der Registrierung der Hilfedatei hinweist.
        traceback.print_exc()  # Prints the full traceback of the exception.  # Gibt die vollständige Fehlermeldung aus.
    else:  # If no error occurs, prints a success message.  # Wenn kein Fehler auftritt, wird eine Erfolgsnachricht ausgegeben.
        if verbose:  # If verbosity is enabled, prints the registration success.  # Wenn die Detailansicht aktiviert ist, wird der Registrierungserfolg ausgegeben.
            print("Registered help file")  # Prints a success message for registering the help file.  # Gibt eine Erfolgsnachricht für die Registrierung der Hilfedatei aus.

    # misc other fixups.
    fixup_dbi()  # Executes any other necessary fixes or setup steps.  # Führt alle anderen erforderlichen Korrekturen oder Einrichtungsschritte aus.

    # Register Pythonwin in context menu
    try:  # Tries to register Pythonwin in the context menu.  # Versucht, Pythonwin im Kontextmenü zu registrieren.
        RegisterPythonwin(True, lib_dir)  # Registers Pythonwin in the context menu.  # Registriert Pythonwin im Kontextmenü.
    except Exception:  # If registration fails, prints a failure message.  # Wenn die Registrierung fehlschlägt, wird eine Fehlermeldung ausgegeben.
        print("Failed to register pythonwin as editor")  # Prints a message indicating failure.  # Gibt eine Nachricht aus, die auf das Fehlschlagen der Registrierung hinweist.
        traceback.print_exc()  # Prints the full traceback of the exception.  # Gibt die vollständige Fehlermeldung aus.
    else:  # If registration is successful, prints a success message.  # Wenn die Registrierung erfolgreich ist, wird eine Erfolgsnachricht ausgegeben.
        if verbose:  # If verbosity is enabled, prints the success message.  # Wenn die Detailansicht aktiviert ist, wird die Erfolgsnachricht ausgegeben.
            print("Pythonwin has been registered in context menu")  # Prints a message indicating Pythonwin is registered.  # Gibt eine Nachricht aus, die besagt, dass Pythonwin im Kontextmenü registriert wurde.

    # Create the win32com\gen_py directory.
    make_dir = os.path.join(lib_dir, "win32com", "gen_py")  # Specifies the path where the "gen_py" directory will be created.  # Gibt den Pfad an, an dem das Verzeichnis "gen_py" erstellt wird.
    if not os.path.isdir(make_dir):  # Checks if the directory doesn't exist.  # Überprüft, ob das Verzeichnis nicht existiert.
        if verbose:  # If verbosity is enabled, prints the directory creation process.  # Wenn die Detailansicht aktiviert ist, wird der Verzeichniserstellungsprozess ausgegeben.
            print("Creating directory %s" % (make_dir,))  # Prints the directory being created.  # Gibt das zu erstellende Verzeichnis aus.
        directory_created(make_dir)  # Creates the directory.  # Erstellt das Verzeichnis.
        os.mkdir(make_dir)  # Creates the directory physically on the file system.  # Erstellt das Verzeichnis physisch im Dateisystem.

    try:  # Tries to create shortcuts.  # Versucht, Verknüpfungen zu erstellen.
        # CSIDL_COMMON_PROGRAMS only available works on NT/2000/XP, and
        # will fail there if the user has no admin rights.
        fldr = get_shortcuts_folder()  # Gets the folder where shortcuts can be placed.  # Ruft das Verzeichnis ab, in dem Verknüpfungen abgelegt werden können.
        # If the group doesn't exist, then we don't make shortcuts - its
        # possible that this isn't a "normal" install.
        if os.path.isdir(fldr):  # Checks if the shortcuts folder exists.  # Überprüft, ob das Verzeichnis für Verknüpfungen existiert.
            dst = os.path.join(fldr, "PythonWin.lnk")  # Specifies the path for the Pythonwin shortcut.  # Gibt den Pfad für die Pythonwin-Verknüpfung an.
            create_shortcut(
                os.path.join(lib_dir, "Pythonwin\\Pythonwin.exe"),
                "The Pythonwin IDE",
                dst,
                "",
                sys.prefix,
            )  # Creates the Pythonwin shortcut.  # Erstellt die Pythonwin-Verknüpfung.
            file_created(dst)  # Registers the shortcut as created.  # Registriert die Verknüpfung als erstellt.
            if verbose:  # If verbosity is enabled, prints the shortcut creation status.  # Wenn die Detailansicht aktiviert ist, wird der Verknüpfungserstellungsstatus ausgegeben.
                print("Shortcut for Pythonwin created")  # Prints success message for Pythonwin shortcut.  # Gibt eine Erfolgsnachricht für die Pythonwin-Verknüpfung aus.
            # And the docs.
            if chm_file:  # If the help file exists, creates a shortcut for it.  # Wenn die Hilfedatei existiert, wird eine Verknüpfung dafür erstellt.
                dst = os.path.join(fldr, "Python for Windows Documentation.lnk")  # Specifies the path for the documentation shortcut.  # Gibt den Pfad für die Dokumentationsverknüpfung an.
                doc = "Documentation for the PyWin32 extensions"  # Describes the documentation.  # Beschreibt die Dokumentation.
                create_shortcut(chm_file, doc, dst)  # Creates the shortcut for the documentation.  # Erstellt die Verknüpfung für die Dokumentation.
                file_created(dst)  # Registers the documentation shortcut as created.  # Registriert die Dokumentationsverknüpfung als erstellt.
                if verbose:  # If verbosity is enabled, prints the shortcut creation status.  # Wenn die Detailansicht aktiviert ist, wird der Verknüpfungserstellungsstatus ausgegeben.
                    print("Shortcut to documentation created")  # Prints success message for documentation shortcut.  # Gibt eine Erfolgsnachricht für die Dokumentationsverknüpfung aus.
        else:  # If the folder for shortcuts does not exist, prints a failure message.  # Wenn das Verzeichnis für Verknüpfungen nicht existiert, wird eine Fehlermeldung ausgegeben.
            if verbose:  # If verbosity is enabled, prints failure message.  # Wenn die Detailansicht aktiviert ist, wird die Fehlermeldung ausgegeben.
                print("Can't install shortcuts - %r is not a folder" % (fldr,))  # Prints error message about shortcut installation failure.  # Gibt eine Fehlermeldung zur Verknüpfungsinstallation aus.
    except Exception as details:  # Catches any exceptions related to creating shortcuts.  # Fängt alle Ausnahmen im Zusammenhang mit der Erstellung von Verknüpfungen ab.
        print(details)  # Prints the exception details.  # Gibt die Ausnahme-Details aus.

    # importing win32com.client ensures the gen_py dir created - not strictly
    # necessary to do now, but this makes the installation "complete"
    try:  # Attempts to import win32com.client.  # Versucht, win32com.client zu importieren.
        import win32com.client  # Imports the win32com.client module.  # Importiert das win32com.client-Modul.
    except ImportError:  # If the import fails, it doesn't raise an error.  # Wenn der Import fehlschlägt, wird kein Fehler ausgelöst.
        pass  # Passes if there's an ImportError.  # Überspringt bei ImportError.
    print("The pywin32 extensions were successfully installed.")  # Prints a success message indicating installation completion.  # Gibt eine Erfolgsnachricht aus, die die erfolgreiche Installation anzeigt.

def uninstall(lib_dir):  # Function to uninstall or clean up components in the provided directory.  # Funktion, um Komponenten im angegebenen Verzeichnis zu deinstallieren oder zu bereinigen.
    LoadSystemModule(lib_dir, "pywintypes")  # Load the pywintypes system module from the library directory.  # Lade das pywintypes-Systemmodul aus dem Bibliotheksverzeichnis.
    LoadSystemModule(lib_dir, "pythoncom")  # Load the pythoncom system module from the library directory.  # Lade das pythoncom-Systemmodul aus dem Bibliotheksverzeichnis.

    try:  # Try block to handle potential exceptions.  # Versuchsblock, um mögliche Ausnahmen zu behandeln.
        RegisterCOMObjects(False)  # Attempt to unregister COM objects.  # Versuche, COM-Objekte zu deregistrieren.
    except Exception as why:  # Catch exception if unregistering COM objects fails.  # Fange Ausnahmen, falls das Deregistrieren der COM-Objekte fehlschlägt.
        print("Failed to unregister COM objects: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.

    try:  # Another try block for handling exceptions.  # Ein weiterer Versuchsblock zur Behandlung von Ausnahmen.
        RegisterHelpFile(False, lib_dir)  # Attempt to unregister the help file from the library directory.  # Versuche, die Hilfsdatei aus dem Bibliotheksverzeichnis zu deregistrieren.
    except Exception as why:  # Catch exception if unregistering the help file fails.  # Fange Ausnahmen, falls das Deregistrieren der Hilfsdatei fehlschlägt.
        print("Failed to unregister help file: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.
    else:  # If the try block succeeds.  # Wenn der Versuchsblock erfolgreich ist.
        if verbose:  # Check if verbose mode is enabled.  # Überprüfe, ob der ausführliche Modus aktiviert ist.
            print("Unregistered help file")  # Print success message.  # Drucke Erfolgsnachricht.

    try:  # Another try block.  # Ein weiterer Versuchsblock.
        RegisterPythonwin(False, lib_dir)  # Attempt to unregister Pythonwin from the library directory.  # Versuche, Pythonwin aus dem Bibliotheksverzeichnis zu deregistrieren.
    except Exception as why:  # Catch exception if unregistering Pythonwin fails.  # Fange Ausnahmen, falls das Deregistrieren von Pythonwin fehlschlägt.
        print("Failed to unregister Pythonwin: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.
    else:  # If the try block succeeds.  # Wenn der Versuchsblock erfolgreich ist.
        if verbose:  # Check if verbose mode is enabled.  # Überprüfe, ob der ausführliche Modus aktiviert ist.
            print("Unregistered Pythonwin")  # Print success message.  # Drucke Erfolgsnachricht.

    try:  # Another try block for cleaning up files.  # Ein weiterer Versuchsblock zum Bereinigen von Dateien.
        gen_dir = os.path.join(lib_dir, "win32com", "gen_py")  # Path to gen_py directory.  # Pfad zum gen_py-Verzeichnis.
        if os.path.isdir(gen_dir):  # Check if gen_py is a directory.  # Überprüfe, ob gen_py ein Verzeichnis ist.
            shutil.rmtree(gen_dir)  # Remove gen_py directory.  # Entferne das gen_py-Verzeichnis.
            if verbose:  # Check if verbose mode is enabled.  # Überprüfe, ob der ausführliche Modus aktiviert ist.
                print("Removed directory %s" % (gen_dir,))  # Print success message.  # Drucke Erfolgsnachricht.

        pywin_dir = os.path.join(lib_dir, "Pythonwin", "pywin")  # Path to pywin directory.  # Pfad zum pywin-Verzeichnis.
        for fname in glob.glob(os.path.join(pywin_dir, "*.cfc")):  # Iterate over .cfc files in pywin directory.  # Iteriere über .cfc-Dateien im pywin-Verzeichnis.
            os.remove(fname)  # Remove each .cfc file.  # Entferne jede .cfc-Datei.

        try:  # Try block for removing old dbi.pyd files.  # Versuchsblock zum Entfernen alter dbi.pyd-Dateien.
            os.remove(os.path.join(lib_dir, "win32", "dbi.pyd.old"))  # Remove dbi.pyd.old file.  # Entferne die dbi.pyd.old-Datei.
        except os.error:  # If there is an error removing dbi.pyd.old.  # Wenn ein Fehler beim Entfernen der dbi.pyd.old-Datei auftritt.
            pass  # Ignore error.  # Ignoriere den Fehler.
        try:  # Try block for removing dbi_d.pyd.old file.  # Versuchsblock zum Entfernen der dbi_d.pyd.old-Datei.
            os.remove(os.path.join(lib_dir, "win32", "dbi_d.pyd.old"))  # Remove dbi_d.pyd.old file.  # Entferne die dbi_d.pyd.old-Datei.
        except os.error:  # If there is an error removing dbi_d.pyd.old.  # Wenn ein Fehler beim Entfernen der dbi_d.pyd.old-Datei auftritt.
            pass  # Ignore error.  # Ignoriere den Fehler.

    except Exception as why:  # Catch exceptions if removing miscellaneous files fails.  # Fange Ausnahmen, falls das Entfernen von verschiedenen Dateien fehlschlägt.
        print("Failed to remove misc files: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.

    try:  # Try block to remove shortcuts.  # Versuchsblock zum Entfernen von Verknüpfungen.
        fldr = get_shortcuts_folder()  # Get the folder containing the shortcuts.  # Hole den Ordner, der die Verknüpfungen enthält.
        for link in ("PythonWin.lnk", "Python for Windows Documentation.lnk"):  # Iterate over the list of shortcuts.  # Iteriere über die Liste der Verknüpfungen.
            fqlink = os.path.join(fldr, link)  # Get the full path to each shortcut.  # Hole den vollständigen Pfad zu jeder Verknüpfung.
            if os.path.isfile(fqlink):  # Check if the shortcut exists.  # Überprüfe, ob die Verknüpfung existiert.
                os.remove(fqlink)  # Remove the shortcut.  # Entferne die Verknüpfung.
                if verbose:  # Check if verbose mode is enabled.  # Überprüfe, ob der ausführliche Modus aktiviert ist.
                    print("Removed %s" % (link,))  # Print success message.  # Drucke Erfolgsnachricht.
    except Exception as why:  # Catch exceptions if removing shortcuts fails.  # Fange Ausnahmen, falls das Entfernen von Verknüpfungen fehlschlägt.
        print("Failed to remove shortcuts: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.

    # Now remove the system32 files.  # Jetzt die system32-Dateien entfernen.
    files = glob.glob(os.path.join(lib_dir, "pywin32_system32\\*.*"))  # Get all files in pywin32_system32 directory.  # Hole alle Dateien im pywin32_system32-Verzeichnis.
    
    try:  # Try block to remove system files.  # Versuchsblock zum Entfernen von Systemdateien.
        for dest_dir in [get_system_dir(), sys.prefix]:  # Iterate over system directories.  # Iteriere über Systemverzeichnisse.
            worked = 0  # Initialize worked flag.  # Initialisiere das Flag 'worked'.
            for fname in files:  # Iterate over files to remove them.  # Iteriere über Dateien, um sie zu entfernen.
                base = os.path.basename(fname)  # Get the base name of the file.  # Hole den Basisnamen der Datei.
                dst = os.path.join(dest_dir, base)  # Create destination path for the file.  # Erstelle den Zielpfad für die Datei.
                if os.path.isfile(dst):  # Check if the file exists at the destination.  # Überprüfe, ob die Datei im Zielverzeichnis existiert.
                    try:
                        os.remove(dst)  # Try to remove the file.  # Versuche, die Datei zu entfernen.
                        worked = 1  # Set worked flag if removal was successful.  # Setze das Flag 'worked', wenn das Entfernen erfolgreich war.
                        if verbose:  # Check if verbose mode is enabled.  # Überprüfe, ob der ausführliche Modus aktiviert ist.
                            print("Removed file %s" % (dst))  # Print success message.  # Drucke Erfolgsnachricht.
                    except Exception:  # Catch exception if file removal fails.  # Fange Ausnahmen, wenn das Entfernen der Datei fehlschlägt.
                        print("FAILED to remove %s" % (dst,))  # Print failure message.  # Drucke Fehlermeldung.
            if worked:  # If any files were removed, break the loop.  # Wenn Dateien entfernt wurden, breche die Schleife ab.
                break
    except Exception as why:  # Catch exceptions if removing system files fails.  # Fange Ausnahmen, falls das Entfernen von Systemdateien fehlschlägt.
        print("FAILED to remove system files: %s" % (why,))  # Print failure message.  # Drucke Fehlermeldung.

# NOTE: If this script is run from inside the bdist_wininst created
# binary installer or uninstaller, the command line args are either
# '-install' or '-remove'.
# Hinweis: Wenn dieses Skript aus dem innerhalb des bdist_wininst erstellten
# Installationsprogramms oder Deinstallationsprogramms ausgeführt wird,
# sind die Befehlszeilenargumente entweder '-install' oder '-remove'.

# Important: From inside the binary installer this script MUST NOT
# call sys.exit() or raise SystemExit, otherwise not only this script
# but also the installer will terminate! (Is there a way to prevent
# this from the bdist_wininst C code?)
# Wichtig: Innerhalb des Binär-Installationsprogramms darf dieses Skript
# nicht sys.exit() oder SystemExit auslösen, da ansonsten nicht nur
# dieses Skript, sondern auch der Installer beendet wird! (Gibt es eine Möglichkeit,
# dies aus dem bdist_wininst C-Code zu verhindern?)

def verify_destination(location):  # Define function to verify the destination path.  # Definiert eine Funktion, um den Zielpfad zu überprüfen.
    if not os.path.isdir(location):  # Checks if the specified path is a directory.  # Überprüft, ob der angegebene Pfad ein Verzeichnis ist.
        raise argparse.ArgumentTypeError('Path "{}" does not exist!'.format(location))  # If not, raise an error.  # Wenn nicht, wird ein Fehler ausgelöst.
    return location  # Return the valid directory path.  # Gibt den gültigen Verzeichnispfad zurück.

def fix_pywin32():  # Function to fix pywin32 if needed.  # Funktion, um pywin32 bei Bedarf zu reparieren.
    """
    Use API this to fix pywin32 programatically.  # API verwenden, um pywin32 programmatisch zu reparieren.

    Note: this only needs to be done if pywin32 members fail to import.  # Hinweis: Dies muss nur durchgeführt werden, wenn pywin32-Module nicht importiert werden können.
    """
    destination = sysconfig.get_paths()["platlib"]  # Get the platform-specific library path.  # Holen Sie sich den plattform-spezifischen Bibliothekspfad.
    install(destination)  # Run the install function to fix the pywin32 installation.  # Führen Sie die Installationsfunktion aus, um die pywin32-Installation zu reparieren.

def main():  # Main function to handle argument parsing and installation/removal actions.  # Hauptfunktion zum Parsen der Argumente und Ausführen von Installations-/Entfernungsaktionen.
    import argparse  # Import the argparse module for command-line argument parsing.  # Importiert das argparse-Modul für das Parsen von Befehlszeilenargumenten.

    parser = argparse.ArgumentParser(  # Create argument parser object.  # Erstellt das Argumentparser-Objekt.
        formatter_class=argparse.RawDescriptionHelpFormatter,  # Set the formatter for help descriptions.  # Legt den Formatierer für Hilfsbeschreibungen fest.
        description="""A post-install script for the pywin32 extensions.  # A post-install script for pywin32 extensions.
    * Typical usage:  # Typische Verwendung:
    > python pywin32_postinstall.py -install  # Beispielaufruf zur Installation.
    If you installed pywin32 via a .exe installer, this should be run  # Wenn Sie pywin32 über einen .exe-Installer installiert haben, wird dies nach der Installation automatisch ausgeführt,
    automatically after installation, but if it fails you can run it again.  # aber wenn es fehlschlägt, können Sie es erneut ausführen.
    If you installed pywin32 via PIP, you almost certainly need to run this to  # Wenn Sie pywin32 über PIP installiert haben, müssen Sie dies wahrscheinlich ausführen, um
    setup the environment correctly.  # die Umgebung korrekt einzurichten.
    Execute with script with a '-install' parameter, to ensure the environment  # Führen Sie das Skript mit dem Parameter '-install' aus, um sicherzustellen, dass die Umgebung korrekt eingerichtet ist.
    is setup correctly.  # und richtig eingerichtet wird.
    """,  # End of description.  # Ende der Beschreibung.
    )
    parser.add_argument(  # Add '-install' argument to the parser.  # Fügt das Argument '-install' zum Parser hinzu.
        "-install",
        default=False,
        action="store_true",  # The flag is true if the argument is passed.  # Das Flag ist wahr, wenn das Argument übergeben wird.
        help="Configure the Python environment correctly for pywin32.",  # Description for '-install' argument.  # Beschreibung des Arguments '-install'.
    )
    parser.add_argument(  # Add '-remove' argument to the parser.  # Fügt das Argument '-remove' zum Parser hinzu.
        "-remove",
        default=False,
        action="store_true",  # The flag is true if the argument is passed.  # Das Flag ist wahr, wenn das Argument übergeben wird.
        help="Try and remove everything that was installed or copied.",  # Description for '-remove' argument.  # Beschreibung des Arguments '-remove'.
    )
    parser.add_argument(  # Add '-wait' argument to the parser to wait for a process.  # Fügt das Argument '-wait' zum Parser hinzu, um auf einen Prozess zu warten.
        "-wait",
        type=int,  # Specifies the type of the argument (integer).  # Gibt den Typ des Arguments an (Ganzzahl).
        help="Wait for the specified process to terminate before starting.",  # Description for '-wait' argument.  # Beschreibung des Arguments '-wait'.
    )
    parser.add_argument(  # Add '-silent' argument to the parser.  # Fügt das Argument '-silent' zum Parser hinzu.
        "-silent",
        default=False,
        action="store_true",  # The flag is true if the argument is passed.  # Das Flag ist wahr, wenn das Argument übergeben wird.
        help='Don\'t display the "Abort/Retry/Ignore" dialog for files in use.',  # Description for '-silent' argument.  # Beschreibung des Arguments '-silent'.
    )
    parser.add_argument(  # Add '-quiet' argument to the parser.  # Fügt das Argument '-quiet' zum Parser hinzu.
        "-quiet",
        default=False,
        action="store_true",  # The flag is true if the argument is passed.  # Das Flag ist wahr, wenn das Argument übergeben wird.
        help="Don't display progress messages.",  # Description for '-quiet' argument.  # Beschreibung des Arguments '-quiet'.
    )
    parser.add_argument(  # Add '-destination' argument to the parser for pywin32 installation path.  # Fügt das Argument '-destination' zum Parser hinzu, um den Installationspfad für pywin32 anzugeben.
        "-destination",
        default=sysconfig.get_paths()["platlib"],  # Default value is the platform-specific library path.  # Der Standardwert ist der plattform-spezifische Bibliothekspfad.
        type=verify_destination,  # Verifies the destination directory using the verify_destination function.  # Überprüft das Zielverzeichnis mithilfe der Funktion verify_destination.
        help="Location of the PyWin32 installation",  # Description for '-destination' argument.  # Beschreibung des Arguments '-destination'.
    )

    args = parser.parse_args()  # Parse the command-line arguments.  # Parsen der Befehlszeilenargumente.

    if not args.quiet:  # If not quiet mode, print parsed arguments.  # Wenn kein Ruhemodus, werden die geparsten Argumente ausgegeben.
        print("Parsed arguments are: {}".format(args))  # Ausgabe der geparsten Argumente.  # Ausgabe der geparsten Argumente.

    if not args.install ^ args.remove:  # Ensure only one of -install or -remove is chosen.  # Sicherstellen, dass entweder -install oder -remove gewählt wird.
        parser.error("You need to either choose to -install or -remove!")  # Error message if neither or both are selected.  # Fehlermeldung, wenn weder noch beide ausgewählt sind.

    if args.wait is not None:  # If wait argument is provided, wait for the specified process.  # Wenn das Argument -wait angegeben wurde, warten Sie auf den angegebenen Prozess.
        try:
            os.waitpid(args.wait, 0)  # Wait for the process with the given PID.  # Warten auf den Prozess mit der angegebenen PID.
        except os.error:  # If process is already dead, ignore the error.  # Wenn der Prozess bereits beendet ist, ignorieren Sie den Fehler.
            pass

    silent = args.silent  # Store the silent mode status.  # Speichert den Status des Stummmodus.
    verbose = not args.quiet  # Set verbose mode based on quiet argument.  # Legt den ausführlichen Modus basierend auf dem Argument quiet fest.

    if args.install:  # If the install argument is passed, call the install function.  # Wenn das Argument -install angegeben ist, rufen Sie die Installationsfunktion auf.
        install(args.destination)  # Install pywin32 to the specified destination.  # Installiert pywin32 am angegebenen Zielort.

    if args.remove:  # If the remove argument is passed, call the uninstall function.  # Wenn das Argument -remove angegeben ist, rufen Sie die Deinstallationsfunktion auf.
        if not is_bdist_wininst:  # Check if not running from a bdist_wininst package.  # Überprüfen, ob nicht aus einem bdist_wininst-Paket ausgeführt wird.
            uninstall(args.destination)  # Uninstall pywin32 from the specified destination.  # Deinstalliert pywin32 vom angegebenen Zielort.

if __name__ == "__main__":  # If this script is run directly, execute the main function.  # Wenn dieses Skript direkt ausgeführt wird, führen Sie die Hauptfunktion aus.
    main()  # Call the main function to start the process.  # Ruft die Hauptfunktion auf, um den Prozess zu starten.
