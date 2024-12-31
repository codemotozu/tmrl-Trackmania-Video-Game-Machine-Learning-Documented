import os  # Imports the os module for interacting with the operating system.  # Importiert das os-Modul, um mit dem Betriebssystem zu interagieren.
import platform  # Imports the platform module to get information about the system.  # Importiert das platform-Modul, um Informationen über das System zu erhalten.
import sys  # Imports the sys module for system-specific parameters and functions.  # Importiert das sys-Modul für systemspezifische Parameter und Funktionen.
from setuptools import find_packages, setup  # Imports tools for packaging and distributing Python projects.  # Importiert Tools zum Verpacken und Verteilen von Python-Projekten.
from pathlib import Path  # Imports the Path class to handle filesystem paths.  # Importiert die Path-Klasse, um mit Dateisystempfaden umzugehen.
from shutil import copy2  # Imports copy2 for file copying with preservation of metadata.  # Importiert copy2 zum Kopieren von Dateien unter Beibehaltung der Metadaten.
from zipfile import ZipFile  # Imports the ZipFile class for working with ZIP files.  # Importiert die ZipFile-Klasse zum Arbeiten mit ZIP-Dateien.
import urllib.request  # Imports the urllib.request module for opening and reading URLs.  # Importiert das urllib.request-Modul zum Öffnen und Lesen von URLs.
import urllib.error  # Imports urllib.error for handling URL-related errors.  # Importiert urllib.error zum Behandeln von URL-bezogenen Fehlern.
import socket  # Imports the socket module for networking tasks.  # Importiert das socket-Modul für Netzwerkaufgaben.

if sys.version_info < (3, 7):  # Checks if Python version is less than 3.7.  # Überprüft, ob die Python-Version kleiner als 3.7 ist.
    sys.exit('Sorry, Python < 3.7 is not supported.')  # Exits the program if Python version is unsupported.  # Beendet das Programm, wenn die Python-Version nicht unterstützt wird.

# NB: the following code is duplicated under tmrl.tools.init_package.init_tmrl, 
# don't forget to update both whenever changing RESOURCES_URL.  # Hinweis: Der folgende Code ist auch in tmrl.tools.init_package.init_tmrl dupliziert. Vergessen Sie nicht, beide zu aktualisieren, wenn RESOURCES_URL geändert wird.

RESOURCES_URL = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"  # URL to download resources.  # URL zum Herunterladen der Ressourcen.

def url_retrieve(url: str, outfile: Path, overwrite: bool = False):  # Function to download a file from a URL.  # Funktion zum Herunterladen einer Datei von einer URL.
    """
    Adapted from https://www.scivision.dev/python-switch-urlretrieve-requests-timeout/
    """  # Explanation of where the code is adapted from.  # Erklärung, von wo der Code adaptiert wurde.
    outfile = Path(outfile).expanduser().resolve()  # Resolves the output path, expanding any user directories.  # Löst den Ausgabe-Pfad auf und erweitert Benutzerdirectorys.
    if outfile.is_dir():  # Checks if the given path is a directory.  # Überprüft, ob der angegebene Pfad ein Verzeichnis ist.
        raise ValueError("Please specify full filepath, including filename")  # Raises error if path is a directory instead of a file.  # Wirft einen Fehler, wenn der Pfad ein Verzeichnis statt einer Datei ist.
    if overwrite or not outfile.is_file():  # Checks if overwrite is allowed or if the file does not exist.  # Überprüft, ob Überschreiben erlaubt ist oder die Datei nicht existiert.
        outfile.parent.mkdir(parents=True, exist_ok=True)  # Creates parent directories if needed.  # Erstellt die übergeordneten Verzeichnisse, falls nötig.
        try:  # Tries to download the file.  # Versucht, die Datei herunterzuladen.
            urllib.request.urlretrieve(url, str(outfile))  # Downloads the file from the URL to the specified output file.  # Lädt die Datei von der URL auf die angegebene Ausgabedatei herunter.
        except (socket.gaierror, urllib.error.URLError) as err:  # Handles network-related errors.  # Behandelt netzwerkbezogene Fehler.
            raise ConnectionError(f"could not download {url} due to {err}")  # Raises a connection error if download fails.  # Wirft einen Verbindungsfehler, wenn der Download fehlschlägt.

# destination folder:  # Defining folder paths.  # Definiert Ordnerpfade.
HOME_FOLDER = Path.home()  # Gets the home directory of the current user.  # Ruft das Home-Verzeichnis des aktuellen Benutzers ab.
TMRL_FOLDER = HOME_FOLDER / "TmrlData"  # Creates a path for the 'TmrlData' folder inside the home directory.  # Erstellt einen Pfad für den Ordner 'TmrlData' im Home-Verzeichnis.

# download relevant items IF THE tmrl FOLDER DOESN'T EXIST:  # Downloads resources if the 'TmrlData' folder does not exist.  # Lädt Ressourcen herunter, wenn der Ordner 'TmrlData' nicht existiert.
if not TMRL_FOLDER.exists():  # Checks if the 'TmrlData' folder already exists.  # Überprüft, ob der Ordner 'TmrlData' bereits existiert.
    CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"  # Creates a path for checkpoints folder.  # Erstellt einen Pfad für den Checkpoints-Ordner.
    DATASET_FOLDER = TMRL_FOLDER / "dataset"  # Creates a path for dataset folder.  # Erstellt einen Pfad für den Dataset-Ordner.
    REWARD_FOLDER = TMRL_FOLDER / "reward"  # Creates a path for reward folder.  # Erstellt einen Pfad für den Reward-Ordner.
    WEIGHTS_FOLDER = TMRL_FOLDER / "weights"  # Creates a path for weights folder.  # Erstellt einen Pfad für den Weights-Ordner.
    CONFIG_FOLDER = TMRL_FOLDER / "config"  # Creates a path for config folder.  # Erstellt einen Pfad für den Config-Ordner.
    CHECKPOINTS_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the checkpoints directory if it doesn't exist.  # Erstellt das Checkpoints-Verzeichnis, wenn es nicht existiert.
    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the dataset directory if it doesn't exist.  # Erstellt das Dataset-Verzeichnis, wenn es nicht existiert.
    REWARD_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the reward directory if it doesn't exist.  # Erstellt das Reward-Verzeichnis, wenn es nicht existiert.
    WEIGHTS_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the weights directory if it doesn't exist.  # Erstellt das Weights-Verzeichnis, wenn es nicht existiert.
    CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the config directory if it doesn't exist.  # Erstellt das Config-Verzeichnis, wenn es nicht existiert.

    # download resources:  # Starts the resource download process.  # Beginnt mit dem Herunterladen der Ressourcen.
    RESOURCES_TARGET = TMRL_FOLDER / "resources.zip"  # Defines the target file for downloading resources.  # Definiert die Zieldatei zum Herunterladen der Ressourcen.
    url_retrieve(RESOURCES_URL, RESOURCES_TARGET)  # Calls the function to download the resources.  # Ruft die Funktion auf, um die Ressourcen herunterzuladen.

    # unzip downloaded resources:  # Extracts the downloaded ZIP file.  # Entpackt die heruntergeladene ZIP-Datei.
    with ZipFile(RESOURCES_TARGET, 'r') as zip_ref:  # Opens the downloaded ZIP file.  # Öffnet die heruntergeladene ZIP-Datei.
        zip_ref.extractall(TMRL_FOLDER)  # Extracts the contents of the ZIP file to the TMRL folder.  # Entpackt den Inhalt der ZIP-Datei in den TMRL-Ordner.

    # delete zip file:  # Deletes the ZIP file after extraction.  # Löscht die ZIP-Datei nach der Extraktion.
    RESOURCES_TARGET.unlink()  # Deletes the downloaded ZIP file.  # Löscht die heruntergeladene ZIP-Datei.

    # copy relevant files:  # Copies the necessary files to their respective folders.  # Kopiert die erforderlichen Dateien in ihre jeweiligen Ordner.
    RESOURCES_FOLDER = TMRL_FOLDER / "resources"  # Defines the resources folder after extraction.  # Definiert den Ressourcen-Ordner nach der Extraktion.
    copy2(RESOURCES_FOLDER / "config.json", CONFIG_FOLDER)  # Copies the config.json file.  # Kopiert die config.json-Datei.
    copy2(RESOURCES_FOLDER / "reward.pkl", REWARD_FOLDER)  # Copies the reward.pkl file.  # Kopiert die reward.pkl-Datei.
    copy2(RESOURCES_FOLDER / "SAC_4_LIDAR_pretrained.tmod", WEIGHTS_FOLDER)  # Copies the SAC model for weights.  # Kopiert das SAC-Modell für Gewichte.
    copy2(RESOURCES_FOLDER / "SAC_4_imgs_pretrained.tmod", WEIGHTS_FOLDER)  # Copies the SAC model for weights.  # Kopiert das SAC-Modell für Gewichte.

    # on Windows, look for OpenPlanet:  # Checks if OpenPlanet exists on Windows.  # Überprüft, ob OpenPlanet auf Windows vorhanden ist.
    if platform.system() == "Windows":  # Checks if the system is Windows.  # Überprüft, ob das System Windows ist.
        OPENPLANET_FOLDER = HOME_FOLDER / "OpenplanetNext"  # Defines the path for OpenPlanet folder.  # Definiert den Pfad für den OpenPlanet-Ordner.

        if OPENPLANET_FOLDER.exists():  # Checks if the OpenPlanet folder exists.  # Überprüft, ob der OpenPlanet-Ordner existiert.
            # copy the OpenPlanet script:  # Copies the OpenPlanet script.  # Kopiert das OpenPlanet-Skript.
            try:  # Tries to copy the script.  # Versucht, das Skript zu kopieren.
                # remove old script if found  # Removes old OpenPlanet scripts.  # Entfernt alte OpenPlanet-Skripte, falls vorhanden.
                OP_SCRIPTS_FOLDER = OPENPLANET_FOLDER / 'Scripts'  # Defines the scripts folder in OpenPlanet.  # Definiert den Skript-Ordner in OpenPlanet.
                if OP_SCRIPTS_FOLDER.exists():  # Checks if the scripts folder exists.  # Überprüft, ob der Skript-Ordner existiert.
                    to_remove = [OP_SCRIPTS_FOLDER / 'Plugin_GrabData_0_1.as',
                                 OP_SCRIPTS_FOLDER / 'Plugin_GrabData_0_1.as.sig',
                                 OP_SCRIPTS_FOLDER / 'Plugin_GrabData_0_2.as',
                                 OP_SCRIPTS_FOLDER / 'Plugin_GrabData_0_2.as.sig']  # Lists files to remove.  # Listet die zu entfernenden Dateien auf.
                    for old_file in to_remove:  # Loops through the files to remove.  # Schleift durch die zu entfernenden Dateien.
                        if old_file.exists():  # Checks if each file exists.  # Überprüft, ob jede Datei existiert.
                            old_file.unlink()  # Deletes the old file.  # Löscht die alte Datei.
                # copy new plugin  # Copies the new plugin to OpenPlanet.  # Kopiert das neue Plugin zu OpenPlanet.
                OP_PLUGINS_FOLDER = OPENPLANET_FOLDER / 'Plugins'  # Defines the plugins folder in OpenPlanet.  # Definiert den Plugins-Ordner in OpenPlanet.
                OP_PLUGINS_FOLDER.mkdir(parents=True, exist_ok=True)  # Creates the plugins folder if it doesn't exist.  # Erstellt den Plugins-Ordner, wenn er nicht existiert.
                TM20_PLUGIN_1 = RESOURCES_FOLDER / 'Plugins' / 'TMRL_GrabData.op'  # Defines the first plugin path.  # Definiert den Pfad zum ersten Plugin.
                TM20_PLUGIN_2 = RESOURCES_FOLDER / 'Plugins' / 'TMRL_SaveGhost.op'  # Defines the second plugin path.  # Definiert den Pfad zum zweiten Plugin.
                copy2(TM20_PLUGIN_1, OP_PLUGINS_FOLDER)  # Copies the first plugin.  # Kopiert das erste Plugin.
                copy2(TM20_PLUGIN_2, OP_PLUGINS_FOLDER)  # Copies the second plugin.  # Kopiert das zweite Plugin.
            except Exception as e:  # Handles any exceptions during the copying process.  # Behandelt Ausnahmen während des Kopierprozesses.
                print(
                    f"An exception was caught when trying to copy the OpenPlanet plugin automatically. \
                    Please copy the plugin manually for TrackMania 2020 support. The caught exception was: {str(e)}.")  # Prints the error message.  # Gibt die Fehlermeldung aus.
        else:  # If OpenPlanet is not found, print a warning message.  # Wenn OpenPlanet nicht gefunden wird, wird eine Warnmeldung ausgegeben.
            print(f"The OpenPlanet folder was not found at {OPENPLANET_FOLDER}. \
            Please copy the OpenPlanet script and signature manually for TrackMania 2020 support.")  # Warns the user.  # Warnt den Benutzer.

install_req = [  # Defines the list of required packages.  # Definiert die Liste der erforderlichen Pakete.
    'numpy',  # Numerical computing library.  # Numerische Rechenbibliothek.
    'torch>=2.0',  # PyTorch deep learning library.  # PyTorch Deep Learning Bibliothek.
    'pandas',  # Data manipulation and analysis library.  # Datenmanipulations- und Analysebibliothek.
    'gymnasium',  # Reinforcement learning environment library.  # Bibliothek für Verstärkungslern-Umgebungen.
    'rtgym>=0.13',  # RTGym package for reinforcement learning.  # RTGym-Paket für Verstärkungslernen.
    'pyyaml',  # YAML parsing library.  # YAML-Parsingsbibliothek.
    'wandb',  # Weights and Biases for experiment tracking.  # Weights and Biases für Experimentverfolgung.
    'requests',  # HTTP library for making requests.  # HTTP-Bibliothek zum Senden von Anfragen.
    'opencv-python',  # OpenCV for computer vision tasks.  # OpenCV für Aufgaben der Computer Vision.
    'pyautogui',  # Automation library for GUI control.  # Automatisierungsbibliothek für GUI-Steuerung.
    'pyinstrument',  # Profiling library for Python.  # Profiling-Bibliothek für Python.
    'tlspyo>=0.2.5',  # TLS library.  # TLS-Bibliothek.
    'chardet',  # Library for character encoding detection.  # Bibliothek zur Erkennung der Zeichencodierung.
    'packaging'  # Packaging utilities for Python.  # Verpackungswerkzeuge für Python.
]

# Dependencies for the TrackMania pipeline  # Abhängigkeiten für die TrackMania-Pipeline.
if platform.system() == "Windows":  # Checks if the platform is Windows.  # Überprüft, ob die Plattform Windows ist.
    install_req.append('pywin32>=303')  # Adds the Windows-specific package.  # Fügt das Windows-spezifische Paket hinzu.
    install_req.append('vgamepad')  # Adds the vgamepad package for virtual gamepad.  # Fügt das vgamepad-Paket für virtuelle Gamepads hinzu.
elif platform.system() == "Linux":  # Checks if the platform is Linux.  # Überprüft, ob die Plattform Linux ist.
    install_req.append('mss')  # Adds the mss package for screenshot capturing.  # Fügt das mss-Paket zum Aufnehmen von Screenshots hinzu.
    install_req.append('vgamepad>=0.1.0')  # Adds the vgamepad package for virtual gamepad.  # Fügt das vgamepad-Paket für virtuelle Gamepads hinzu.

# Short readme for PyPI  # Readme für PyPI (Python Package Index).
HERE = os.path.abspath(os.path.dirname(__file__))  # Gets the current directory of the script.  # Holt sich das aktuelle Verzeichnis des Skripts.
README_FOLDER = os.path.join(HERE, "readme")  # Defines the folder containing the README.  # Definiert den Ordner, der die README enthält.
with open(os.path.join(README_FOLDER, "pypi.md")) as fid:  # Opens the README file.  # Öffnet die README-Datei.
    README = fid.read()  # Reads the contents of the README.  # Liest den Inhalt der README.

setup(  # Sets up the Python package for distribution.  # Setzt das Python-Paket für die Verteilung auf.
    name='tmrl',  # The name of the package.  # Der Name des Pakets.
    version='0.6.6',  # The version of the package.  # Die Version des Pakets.
    description='Network-based framework for real-time robot learning',  # Short description of the package.  # Kurze Beschreibung des Pakets.
    long_description=README,  # The long description for the package.  # Die lange Beschreibung für das Paket.
    long_description_content_type='text/markdown',  # Specifies the content type for long description.  # Gibt den Inhaltstyp für die lange Beschreibung an.
    keywords='reinforcement learning, robot learning, trackmania, self driving, roborace',  # Keywords for the package.  # Schlüsselwörter für das Paket.
    url='https://github.com/trackmania-rl/tmrl',  # URL for the package repository.  # URL zum Repository des Pakets.
    download_url='https://github.com/trackmania-rl/tmrl/archive/refs/tags/v0.6.6.tar.gz',  # URL to download the package.  # URL zum Herunterladen des Pakets.
    author='Yann Bouteiller, Edouard Geze',  # Authors of the package.  # Autoren des Pakets.
    author_email='yann.bouteiller@polymtl.ca, edouard.geze@hotmail.fr',  # Authors' email addresses.  # E-Mail-Adressen der Autoren.
    license='MIT',  # License type for the package.  # Lizenztyp für das Paket.
    install_requires=install_req,  # List of dependencies required for the package.  # Liste der Abhängigkeiten, die für das Paket erforderlich sind.
    classifiers=[  # Classifiers for the package.  # Klassifikatoren für das Paket.
            'Development Status :: 4 - Beta',  # Development status of the package.  # Entwicklungsstatus des Pakets.
            'Intended Audience :: Developers',  # Intended audience for the package.  # Zielgruppe des Pakets.
            'Intended Audience :: Education',  # Intended audience for education.  # Zielgruppe für Bildung.
            'Intended Audience :: Information Technology',  # Intended audience for IT professionals.  # Zielgruppe für IT-Fachleute.
            'Intended Audience :: Science/Research',  # Intended audience for science and research.  # Zielgruppe für Wissenschaft und Forschung.
            'License :: OSI Approved :: MIT License',  # License type.  # Lizenztyp.
            'Programming Language :: Python',  # Programming language used.  # Verwendete Programmiersprache.
            'Topic :: Games/Entertainment',  # Package topic.  # Thema des Pakets.
            'Topic :: Scientific/Engineering :: Artificial Intelligence',  # Package topic related to AI.  # Thema des Pakets im Bereich KI.
        ],  
    include_package_data=True,  # Includes package data in the distribution.  # Schließt Paketdaten in die Verteilung ein.
    extras_require={},  # Additional optional requirements.  # Zusätzliche optionale Anforderungen.
    scripts=[],  # Defines the scripts to be included in the package.  # Definiert die Skripte, die im Paket enthalten sein sollen.
    packages=find_packages(exclude=("tests", )),  # Finds and includes all packages, excluding tests.  # Findet und schließt alle Pakete ein, mit Ausnahme von Tests.
)
