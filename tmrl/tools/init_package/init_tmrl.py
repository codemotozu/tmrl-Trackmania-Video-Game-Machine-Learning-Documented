import logging  # Importing the logging module to record log messages.  # Importieren des Logging-Moduls zum Aufzeichnen von Log-Nachrichten
import platform  # Importing the platform module to check the operating system.  # Importieren des Platform-Moduls, um das Betriebssystem zu überprüfen
from pathlib import Path  # Importing the Path class from pathlib for easier file path manipulation.  # Importieren der Path-Klasse von pathlib zur einfacheren Handhabung von Dateipfaden

def rmdir(directory):  # Function to recursively delete a directory and its contents.  # Funktion zum rekursiven Löschen eines Verzeichnisses und seines Inhalts
    directory = Path(directory)  # Converts the directory input into a Path object.  # Wandelt die Verzeichniseingabe in ein Path-Objekt um
    for item in directory.iterdir():  # Iterating over all items in the directory.  # Iterieren über alle Elemente im Verzeichnis
        if item.is_dir():  # If the item is a directory, recursively call rmdir.  # Wenn das Element ein Verzeichnis ist, rufe rmdir rekursiv auf
            rmdir(item)  # Recursive call to remove subdirectories.  # Rekursiver Aufruf, um Unterverzeichnisse zu entfernen
        else:  # If the item is a file, delete it.  # Wenn das Element eine Datei ist, lösche sie
            item.unlink()  # Deletes the file.  # Löscht die Datei
    directory.rmdir()  # Deletes the now empty directory.  # Löscht das nun leere Verzeichnis

def init_tmrl_data():  # Function to initialize TMRL data by wiping and re-generating required folders and files.  # Funktion zur Initialisierung der TMRL-Daten durch Löschen und erneutes Erstellen der benötigten Ordner und Dateien
    """
    Wipes and re-generates the TmrlData folder.  # Löscht und regeneriert den TmrlData-Ordner.
    """
    from shutil import copy2  # Importing copy2 from shutil to copy files with metadata.  # Importieren von copy2 aus shutil, um Dateien mit Metadaten zu kopieren
    from zipfile import ZipFile  # Importing ZipFile to handle zip file operations.  # Importieren von ZipFile zur Handhabung von ZIP-Dateioperationen
    import urllib.request  # Importing urllib.request for downloading resources from a URL.  # Importieren von urllib.request zum Herunterladen von Ressourcen von einer URL
    import urllib.error  # Importing urllib.error to handle URL-related errors.  # Importieren von urllib.error zur Behandlung von URL-bezogenen Fehlern
    import socket  # Importing socket to handle network errors.  # Importieren von socket zur Handhabung von Netzwerkfehlern

    resources_url = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"  # URL of the resources zip file to download.  # URL der zu ladenden Ressourcen-ZIP-Datei

    def url_retrieve(url: str, outfile: Path, overwrite: bool = False):  # Function to download a file from the URL.  # Funktion zum Herunterladen einer Datei von der URL
        """
        Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/  # Adapted method for URL download with timeout handling.  # Adaptierte Methode für den URL-Download mit Timeout-Behandlung
        """
        outfile = Path(outfile).expanduser().resolve()  # Resolving the output path.  # Auflösen des Ausgabepfads
        if outfile.is_dir():  # Check if the specified path is a directory.  # Überprüft, ob der angegebene Pfad ein Verzeichnis ist
            raise ValueError("Please specify full filepath, including filename")  # Raise an error if it's a directory.  # Wirft einen Fehler, wenn es ein Verzeichnis ist
        if overwrite or not outfile.is_file():  # If overwrite is True or file does not exist, proceed to download.  # Wenn Überschreiben wahr ist oder die Datei nicht existiert, fahre mit dem Download fort
            outfile.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they do not exist.  # Erstelle die übergeordneten Verzeichnisse, falls sie nicht existieren
            try:
                urllib.request.urlretrieve(url, str(outfile))  # Download the file from the URL.  # Lade die Datei von der URL herunter
            except (socket.gaierror, urllib.error.URLError) as err:  # Handle network-related errors.  # Behandle netzwerkbezogene Fehler
                raise ConnectionError(f"could not download {url} due to {err}")  # Raise an error if download fails.  # Wirft einen Fehler, wenn der Download fehlschlägt

    home_folder = Path.home()  # Get the user's home directory.  # Hole das Home-Verzeichnis des Benutzers
    tmrl_folder = home_folder / "TmrlData"  # Define the target directory for TMRL data.  # Definiere das Zielverzeichnis für TMRL-Daten

    # Wipe the tmrl folder:  # Lösche den TMRL-Ordner:
    if tmrl_folder.exists():  # If the folder exists, delete it.  # Wenn der Ordner existiert, lösche ihn
        rmdir(tmrl_folder)  # Call rmdir to recursively delete all contents.  # Rufe rmdir auf, um alle Inhalte rekursiv zu löschen

    # download relevant items IF THE tmrl FOLDER DOESN'T EXIST:  # Lade relevante Elemente herunter, WENN DER TMRL-ORDNER NICHT EXISTIERT:
    assert not tmrl_folder.exists(), f"Failed to delete {tmrl_folder}"  # Ensure the folder is deleted before proceeding.  # Stelle sicher, dass der Ordner vor dem Fortfahren gelöscht wurde

    checkpoints_folder = tmrl_folder / "checkpoints"  # Create a subdirectory for checkpoints.  # Erstelle ein Unterverzeichnis für Checkpoints
    dataset_folder = tmrl_folder / "dataset"  # Create a subdirectory for datasets.  # Erstelle ein Unterverzeichnis für Datensätze
    reward_folder = tmrl_folder / "reward"  # Create a subdirectory for reward data.  # Erstelle ein Unterverzeichnis für Belohnungsdaten
    weights_folder = tmrl_folder / "weights"  # Create a subdirectory for weights data.  # Erstelle ein Unterverzeichnis für Gewichtsdaten
    config_folder = tmrl_folder / "config"  # Create a subdirectory for configuration files.  # Erstelle ein Unterverzeichnis für Konfigurationsdateien
    checkpoints_folder.mkdir(parents=True, exist_ok=True)  # Create the checkpoints folder, ensuring parent directories exist.  # Erstelle den Checkpoints-Ordner, stelle sicher, dass übergeordnete Verzeichnisse existieren
    dataset_folder.mkdir(parents=True, exist_ok=True)  # Create the dataset folder.  # Erstelle den Datensatz-Ordner
    reward_folder.mkdir(parents=True, exist_ok=True)  # Create the reward folder.  # Erstelle den Belohnungs-Ordner
    weights_folder.mkdir(parents=True, exist_ok=True)  # Create the weights folder.  # Erstelle den Gewichte-Ordner
    config_folder.mkdir(parents=True, exist_ok=True)  # Create the config folder.  # Erstelle den Konfigurations-Ordner

    resources_target = tmrl_folder / "resources.zip"  # Define the target path for the downloaded resources zip file.  # Definiere den Zielpfad für die heruntergeladene Ressourcen-ZIP-Datei
    url_retrieve(resources_url, resources_target)  # Download the resources zip file.  # Lade die Ressourcen-ZIP-Datei herunter

    # unzip downloaded resources:  # Entpacke die heruntergeladenen Ressourcen:
    with ZipFile(resources_target, 'r') as zip_ref:  # Open the zip file for extraction.  # Öffne die ZIP-Datei zur Extraktion
        zip_ref.extractall(tmrl_folder)  # Extract all files to the tmrl folder.  # Extrahiere alle Dateien in den TMRL-Ordner

    # delete zip file:  # Lösche die ZIP-Datei:
    resources_target.unlink()  # Remove the zip file after extraction.  # Entferne die ZIP-Datei nach der Extraktion

    # copy relevant files:  # Kopiere relevante Dateien:
    resources_folder = tmrl_folder / "resources"  # Define the folder where resources were extracted.  # Definiere den Ordner, in den die Ressourcen extrahiert wurden
    copy2(resources_folder / "config.json", config_folder)  # Copy the config.json file to the config folder.  # Kopiere die config.json-Datei in den Konfigurations-Ordner
    copy2(resources_folder / "reward.pkl", reward_folder)  # Copy the reward.pkl file to the reward folder.  # Kopiere die reward.pkl-Datei in den Belohnungs-Ordner
    copy2(resources_folder / "SAC_4_LIDAR_pretrained.tmod", weights_folder)  # Copy the weights files to the weights folder.  # Kopiere die Gewichtdateien in den Gewichte-Ordner
    copy2(resources_folder / "SAC_4_imgs_pretrained.tmod", weights_folder)  # Copy another weights file to the weights folder.  # Kopiere eine weitere Gewichtdatei in den Gewichte-Ordner

    # on Windows, look for OpenPlanet:  # Auf Windows nach OpenPlanet suchen:
    if platform.system() == "Windows":  # Check if the operating system is Windows.  # Überprüfe, ob das Betriebssystem Windows ist
        openplanet_folder = home_folder / "OpenplanetNext"  # Define the path to the OpenPlanet folder.  # Definiere den Pfad zum OpenPlanet-Ordner

        if openplanet_folder.exists():  # If the OpenPlanet folder exists, proceed with copying the plugin.  # Wenn der OpenPlanet-Ordner existiert, fahre mit dem Kopieren des Plugins fort
            try:
                # remove old script if found  # Entferne das alte Skript, falls es gefunden wird
                op_scripts_folder = openplanet_folder / 'Scripts'  # Path to the OpenPlanet scripts folder.  # Pfad zum OpenPlanet-Skripte-Ordner
                if op_scripts_folder.exists():  # If the scripts folder exists, remove old script files.  # Wenn der Skripte-Ordner existiert, entferne alte Skriptdateien
                    to_remove = [op_scripts_folder / 'Plugin_GrabData_0_1.as',
                                 op_scripts_folder / 'Plugin_GrabData_0_1.as.sig',
                                 op_scripts_folder / 'Plugin_GrabData_0_2.as',
                                 op_scripts_folder / 'Plugin_GrabData_0_2.as.sig']
                    for old_file in to_remove:  # Iterate over files to remove.  # Iteriere über die zu entfernenden Dateien
                        if old_file.exists():  # If the file exists, delete it.  # Wenn die Datei existiert, lösche sie
                            old_file.unlink()  # Delete the file.  # Lösche die Datei
                # copy new plugin  # Kopiere das neue Plugin
                op_plugins_folder = openplanet_folder / 'Plugins'  # Path to the OpenPlanet plugins folder.  # Pfad zum OpenPlanet-Plugins-Ordner
                op_plugins_folder.mkdir(parents=True, exist_ok=True)  # Create the plugins folder if it does not exist.  # Erstelle den Plugins-Ordner, falls er nicht existiert
                tm20_plugin_1 = resources_folder / 'Plugins' / 'TMRL_GrabData.op'  # Path to the first plugin file.  # Pfad zur ersten Plugin-Datei
                tm20_plugin_2 = resources_folder / 'Plugins' / 'TMRL_SaveGhost.op'  # Path to the second plugin file.  # Pfad zur zweiten Plugin-Datei
                copy2(tm20_plugin_1, op_plugins_folder)  # Copy the first plugin to the OpenPlanet plugins folder.  # Kopiere das erste Plugin in den OpenPlanet-Plugins-Ordner
                copy2(tm20_plugin_2, op_plugins_folder)  # Copy the second plugin to the OpenPlanet plugins folder.  # Kopiere das zweite Plugin in den OpenPlanet-Plugins-Ordner
            except Exception as e:  # If an exception occurs, print an error message.  # Wenn eine Ausnahme auftritt, drucke eine Fehlermeldung
                print(
                    f"An exception was caught when trying to copy the OpenPlanet plugin automatically. \
                    Please copy the plugin manually for TrackMania 2020 support. The caught exception was: {str(e)}.")  # Print an error message for manual action.  # Drucke eine Fehlermeldung für manuelles Handeln
        else:  # If OpenPlanet folder is not found, notify the user.  # Wenn der OpenPlanet-Ordner nicht gefunden wurde, benachrichtige den Benutzer
            print(f"The OpenPlanet folder was not found at {openplanet_folder}. \
            Please copy the OpenPlanet script and signature manually for TrackMania 2020 support.")  # Notify the user that OpenPlanet is missing.  # Benachrichtige den Benutzer, dass OpenPlanet fehlt

TMRL_FOLDER = Path.home() / "TmrlData"  # Define the TMRL folder path.  # Definiere den TMRL-Ordner-Pfad

if not TMRL_FOLDER.exists():  # If the TMRL folder does not exist, attempt to create it.  # Wenn der TMRL-Ordner nicht existiert, versuche, ihn zu erstellen
    logging.warning(f"The TMRL folder was not found on your machine. Attempting download...")  # Log a warning message.  # Protokolliere eine Warnmeldung
    init_tmrl_data()  # Call the function to initialize the TMRL data.  # Rufe die Funktion zur Initialisierung der TMRL-Daten auf
    logging.info(f"TMRL folder successfully downloaded, please wait for initialization to complete...")  # Log an info message indicating success.  # Protokolliere eine Informationsnachricht zur erfolgreichen Durchführung
