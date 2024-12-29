# standard library imports
import logging  # Provides logging functionality for debugging and diagnostics.  # Bietet Logging-Funktionalität für Debugging und Diagnosen.
import os  # OS-level operations such as file paths and environment variables.  # Betriebssystemoperationen wie Dateipfade und Umgebungsvariablen.
from pathlib import Path  # Object-oriented file system paths.  # Objektorientierte Dateisystempfade.
import json  # Reading and writing JSON files.  # Lesen und Schreiben von JSON-Dateien.
import platform  # Platform-specific information (e.g., OS type).  # Plattform-spezifische Informationen (z. B. Betriebssystemtyp).
from packaging import version  # Handles software version comparison.  # Verarbeitet Softwareversionsvergleiche.

__compatibility__ = "0.6.0"  # Minimum compatible version of the configuration file.  # Minimale kompatible Version der Konfigurationsdatei.

# TMRL FOLDER: =======================================================
SYSTEM = platform.system()  # Get the operating system (e.g., Windows or Linux).  # Ermittelt das Betriebssystem (z. B. Windows oder Linux).
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"  # OS-dependent version of the RTGym interface.  # Betriebssystemabhängige Version der RTGym-Schnittstelle.

TMRL_FOLDER = Path.home() / "TmrlData"  # Path to the main TMRL data folder.  # Pfad zum Hauptdatenordner von TMRL.

if not TMRL_FOLDER.exists():  # Check if the folder exists.  # Überprüft, ob der Ordner existiert.
    raise RuntimeError(f"Missing folder: {TMRL_FOLDER}")  # Raise an error if the folder is missing.  # Gibt einen Fehler aus, wenn der Ordner fehlt.

CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"  # Folder for model checkpoints.  # Ordner für Modellprüfpunkte.
DATASET_FOLDER = TMRL_FOLDER / "dataset"  # Folder for datasets.  # Ordner für Datensätze.
REWARD_FOLDER = TMRL_FOLDER / "reward"  # Folder for reward data.  # Ordner für Belohnungsdaten.
WEIGHTS_FOLDER = TMRL_FOLDER / "weights"  # Folder for model weights.  # Ordner für Modellgewichte.
CONFIG_FOLDER = TMRL_FOLDER / "config"  # Folder for configuration files.  # Ordner für Konfigurationsdateien.

CONFIG_FILE = TMRL_FOLDER / "config" / "config.json"  # Path to the main configuration file.  # Pfad zur Hauptkonfigurationsdatei.
with open(CONFIG_FILE) as f:  # Open the configuration file.  # Öffnet die Konfigurationsdatei.
    TMRL_CONFIG = json.load(f)  # Load the configuration as a dictionary.  # Lädt die Konfiguration als Wörterbuch.

# VERSION CHECK: =====================================================
__err_msg = "Perform a clean installation:\n(1) Uninstall TMRL,\n(2) Delete the TmrlData folder,\n(3) Reinstall TMRL."  # Error message for version mismatch.  # Fehlermeldung bei Versionsinkompatibilität.
assert "__VERSION__" in TMRL_CONFIG, "config.json is outdated. " + __err_msg  # Ensure version information exists in the config.  # Stellt sicher, dass Versionsinformationen in der Konfiguration vorhanden sind.
CONFIG_VERSION = TMRL_CONFIG["__VERSION__"]  # Extract the version from the configuration.  # Extrahiert die Version aus der Konfiguration.
assert version.parse(CONFIG_VERSION) >= version.parse(__compatibility__), \
    f"config.json version ({CONFIG_VERSION}) must be >= {__compatibility__}. " + __err_msg  # Check compatibility of the config version.  # Überprüft die Kompatibilität der Konfigurationsversion.

# GENERAL: ===========================================================
RUN_NAME = TMRL_CONFIG["RUN_NAME"]  # Name of the current experiment or run.  # Name des aktuellen Experiments oder Durchlaufs.
BUFFERS_MAXLEN = TMRL_CONFIG["BUFFERS_MAXLEN"]  # Maximum length for local buffers.  # Maximale Länge für lokale Puffer.
RW_MAX_SAMPLES_PER_EPISODE = TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]  # Maximum timesteps per episode.  # Maximale Zeitstufen pro Episode.

PRAGMA_RNN = False  # Use RNN (True) or MLP (False) in the model.  # Verwendet RNN (True) oder MLP (False) im Modell.

CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]  # Use CUDA for training (True) or CPU (False).  # Verwendet CUDA (True) oder CPU (False) für das Training.
CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]  # Use CUDA for inference (True) or CPU (False).  # Verwendet CUDA (True) oder CPU (False) für die Inferenz.

PRAGMA_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]  # Use gamepad (True) or keyboard (False).  # Verwendet Gamepad (True) oder Tastatur (False).

LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]  # RolloutWorker is on the same machine as the Server.  # RolloutWorker ist auf derselben Maschine wie der Server.
LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]  # Trainer is on the same machine as the Server.  # Trainer ist auf derselben Maschine wie der Server.
PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]  # Public IP address of the server.  # Öffentliche IP-Adresse des Servers.

SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"  # Server IP for workers.  # Server-IP für Worker.
SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"  # Server IP for trainers.  # Server-IP für Trainer.

# ENVIRONMENT: =======================================================
ENV_CONFIG = TMRL_CONFIG["ENV"]  # Configuration for the simulation environment.  # Konfiguration für die Simulationsumgebung.
RTGYM_INTERFACE = ENV_CONFIG["RTGYM_INTERFACE"]  # RTGym interface type.  # RTGym-Schnittstellentyp.
PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")  # Use LIDAR if the interface name ends with "LIDAR".  # Verwendet LIDAR, wenn der Schnittstellenname mit "LIDAR" endet.
PRAGMA_PROGRESS = RTGYM_INTERFACE.endswith("LIDARPROGRESS")  # Check for progress in the LIDAR interface.  # Überprüft Fortschritt in der LIDAR-Schnittstelle.
if PRAGMA_PROGRESS:  # Enable LIDAR if progress mode is active.  # Aktiviert LIDAR, wenn Fortschrittsmodus aktiv ist.
    PRAGMA_LIDAR = True
LIDAR_BLACK_THRESHOLD = [55, 55, 55]  # Color threshold for LIDAR processing.  # Farbgrenzwert für die LIDAR-Verarbeitung.
REWARD_CONFIG = ENV_CONFIG["REWARD_CONFIG"]  # Configuration for reward calculations.  # Konfiguration für Belohnungsberechnungen.
SLEEP_TIME_AT_RESET = ENV_CONFIG["SLEEP_TIME_AT_RESET"]  # Sleep time after environment reset.  # Wartezeit nach dem Zurücksetzen der Umgebung.
IMG_HIST_LEN = ENV_CONFIG["IMG_HIST_LEN"]  # Image history length (e.g., for state stacking).  # Länge der Bildhistorie (z. B. für Zustandsstapelung).
ACT_BUF_LEN = ENV_CONFIG["RTGYM_CONFIG"]["act_buf_len"]  # Length of action buffer.  # Länge des Aktionspuffers.
WINDOW_WIDTH = ENV_CONFIG["WINDOW_WIDTH"]  # Simulation window width.  # Breite des Simulationsfensters.
WINDOW_HEIGHT = ENV_CONFIG["WINDOW_HEIGHT"]  # Simulation window height.  # Höhe des Simulationsfensters.
GRAYSCALE = ENV_CONFIG["IMG_GRAYSCALE"] if "IMG_GRAYSCALE" in ENV_CONFIG else False  # Use grayscale images if specified.  # Verwendet Graustufenbilder, wenn angegeben.
IMG_WIDTH = ENV_CONFIG["IMG_WIDTH"] if "IMG_WIDTH" in ENV_CONFIG else 64  # Image width.  # Bildbreite.
IMG_HEIGHT = ENV_CONFIG["IMG_HEIGHT"] if "IMG_HEIGHT" in ENV_CONFIG else 64  # Image height.  # Bildhöhe.
LINUX_X_OFFSET = ENV_CONFIG["LINUX_X_OFFSET"] if "LINUX_X_OFFSET" in ENV_CONFIG else 64  # X offset for Linux.  # X-Versatz für Linux.
LINUX_Y_OFFSET = ENV_CONFIG["LINUX_Y_OFFSET"] if "LINUX_Y_OFFSET" in ENV_CONFIG else 70  # Y offset for Linux.  # Y-Versatz für Linux.
IMG_SCALE_CHECK_ENV = ENV_CONFIG["IMG_SCALE_CHECK_ENV"] if "IMG_SCALE_CHECK_ENV" in ENV_CONFIG else 1.0  # Image scale factor.  # Bildskalierungsfaktor.

# DEBUGGING AND BENCHMARKING: ===================================
CRC_DEBUG = False  # Enable CRC debugging for networking consistency.  # Aktiviert CRC-Debug

ging für Netzwerkkonsistenz.
CRC_DEBUG_SAMPLES = 100  # Number of samples collected during CRC debugging.  # Anzahl der gesammelten Proben im CRC-Debugging.
PROFILE_TRAINER = False  # Enable profiling for trainer epochs.  # Aktiviert die Profilierung für Trainer-Epochen.
SYNCHRONIZE_CUDA = False  # Enable CUDA synchronization for profiling.  # Aktiviert CUDA-Synchronisation für die Profilierung.
DEBUG_MODE = TMRL_CONFIG["DEBUG_MODE"] if "DEBUG_MODE" in TMRL_CONFIG.keys() else False  # Enable debug mode based on configuration.  # Aktiviert den Debugmodus basierend auf der Konfiguration.

# FILE SYSTEM: =================================================
PATH_DATA = TMRL_FOLDER  # Root path for TMRL data.  # Stammverzeichnis für TMRL-Daten.
logging.debug(f" PATH_DATA:{PATH_DATA}")  # Log the data path for debugging.  # Protokolliert den Datenpfad zum Debuggen.

MODEL_HISTORY = TMRL_CONFIG["SAVE_MODEL_EVERY"]  # Interval for saving model history.  # Intervall für das Speichern der Modellhistorie.
MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))  # Path for the worker's model.  # Pfad für das Modell des Arbeiters.
MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))  # Path for model history.  # Pfad für die Modellhistorie.
MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))  # Path for the trainer's model.  # Pfad für das Modell des Trainers.
CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))  # Path for model checkpoints.  # Pfad für Modellprüfpunkte.
DATASET_PATH = str(DATASET_FOLDER)  # Path for dataset storage.  # Pfad für die Datensatzspeicherung.
REWARD_PATH = str(REWARD_FOLDER / "reward.pkl")  # Path for reward file storage.  # Pfad für die Speicherung von Belohnungsdateien.

# WANDB: =======================================================
WANDB_RUN_ID = RUN_NAME  # Weights and Biases run identifier.  # Weights and Biases Lauf-ID.
WANDB_PROJECT = TMRL_CONFIG["WANDB_PROJECT"]  # Name of the WANDB project.  # Name des WANDB-Projekts.
WANDB_ENTITY = TMRL_CONFIG["WANDB_ENTITY"]  # Name of the WANDB entity.  # Name der WANDB-Entität.
WANDB_KEY = TMRL_CONFIG["WANDB_KEY"]  # API key for WANDB authentication.  # API-Schlüssel für WANDB-Authentifizierung.

os.environ['WANDB_API_KEY'] = WANDB_KEY  # Set the WANDB API key in the environment.  # Legt den WANDB-API-Schlüssel in der Umgebung fest.

# NETWORKING: ==================================================
PRINT_BYTESIZES = True  # Print the size of network data packets.  # Gibt die Größe der Netzwerkdatenpakete aus.

PORT = TMRL_CONFIG["PORT"]  # Port for server communication.  # Port für die Serverkommunikation.
LOCAL_PORT_SERVER = TMRL_CONFIG["LOCAL_PORT_SERVER"]  # Local port for the server.  # Lokaler Port für den Server.
LOCAL_PORT_TRAINER = TMRL_CONFIG["LOCAL_PORT_TRAINER"]  # Local port for the trainer.  # Lokaler Port für den Trainer.
LOCAL_PORT_WORKER = TMRL_CONFIG["LOCAL_PORT_WORKER"]  # Local port for the worker.  # Lokaler Port für den Arbeiter.
PASSWORD = TMRL_CONFIG["PASSWORD"]  # Password for network authentication.  # Passwort für die Netzwerkauthentifizierung.
SECURITY = "TLS" if TMRL_CONFIG["TLS"] else None  # Use TLS security if enabled.  # Verwendet TLS-Sicherheit, wenn aktiviert.
CREDENTIALS_DIRECTORY = TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] if TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] != "" else None  # Path to TLS credentials.  # Pfad zu TLS-Zertifikaten.
HOSTNAME = TMRL_CONFIG["TLS_HOSTNAME"]  # Hostname for TLS communication.  # Hostname für TLS-Kommunikation.
NB_WORKERS = None if TMRL_CONFIG["NB_WORKERS"] < 0 else TMRL_CONFIG["NB_WORKERS"]  # Number of workers (None for unlimited).  # Anzahl der Arbeiter (None für unbegrenzt).

BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]  # Socket buffer size in bytes.  # Größe des Socket-Puffers in Bytes.
HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]  # Fixed header size for network messages.  # Feste Kopfzeilengröße für Netzwerkmeldungen.
