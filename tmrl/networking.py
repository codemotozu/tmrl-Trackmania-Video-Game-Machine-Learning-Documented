# standard library imports
import datetime  # Provides classes for manipulating dates and times.  # Stellt Klassen zum Bearbeiten von Datums- und Zeitangaben zur Verfügung.
import os  # Provides functions to interact with the operating system.  # Bietet Funktionen zur Interaktion mit dem Betriebssystem.
import socket  # Provides low-level networking interfaces.  # Bietet Schnittstellen für Netzwerkanwendungen auf niedriger Ebene.
import time  # Provides functions for working with time.  # Bietet Funktionen zur Arbeit mit Zeitangaben.
import atexit  # Provides a way to register functions to be called upon normal program termination.  # Ermöglicht das Registrieren von Funktionen, die beim normalen Programmende aufgerufen werden.
import json  # Provides functions for working with JSON data.  # Bietet Funktionen zur Arbeit mit JSON-Daten.
import shutil  # Provides functions for file operations like copying and removing files.  # Bietet Funktionen für Dateioperationen wie Kopieren und Entfernen von Dateien.
import tempfile  # Provides functions to create temporary files and directories.  # Bietet Funktionen zum Erstellen temporärer Dateien und Verzeichnisse.
import itertools  # Provides functions for creating iterators for efficient looping.  # Bietet Funktionen zur Erstellung von Iteratoren für effizientes Schleifen.
from os.path import exists  # Checks if a given path exists.  # Überprüft, ob ein angegebener Pfad existiert.

# third-party imports
import numpy as np  # Imports the numpy library, used for numerical computations.  # Importiert die numpy-Bibliothek, die für numerische Berechnungen verwendet wird.
from requests import get  # Imports the `get` function from the requests library to make HTTP requests.  # Importiert die `get`-Funktion aus der Requests-Bibliothek, um HTTP-Anfragen zu stellen.
from tlspyo import Relay, Endpoint  # Imports `Relay` and `Endpoint` from the tlspyo library for network communication.  # Importiert `Relay` und `Endpoint` aus der tlspyo-Bibliothek für die Netzwerkkommunikation.

# local imports
from tmrl.actor import ActorModule  # Imports the ActorModule from the local tmrl actor module.  # Importiert das ActorModule aus dem lokalen tmrl-Modul für Akteure.
from tmrl.util import dump, load, partial_to_dict  # Imports utility functions from the tmrl module.  # Importiert Hilfsfunktionen aus dem tmrl-Modul.
import tmrl.config.config_constants as cfg  # Imports configuration constants from the tmrl module.  # Importiert Konfigurationskonstanten aus dem tmrl-Modul.
import tmrl.config.config_objects as cfg_obj  # Imports configuration objects from the tmrl module.  # Importiert Konfigurationsobjekte aus dem tmrl-Modul.

import logging  # Imports the logging module for logging messages.  # Importiert das Logging-Modul, um Nachrichten zu protokollieren.


__docformat__ = "google"  # Specifies the format for docstrings (Google style).  # Gibt das Format für Docstrings an (Google-Stil).

# PRINT: ============================================

def print_with_timestamp(s):
    x = datetime.datetime.now()  # Gets the current date and time.  # Holt das aktuelle Datum und die Uhrzeit.
    sx = x.strftime("%x %X ")  # Formats the date and time as a string.  # Formatiert das Datum und die Uhrzeit als Zeichenkette.
    logging.info(sx + str(s))  # Logs the message with a timestamp.  # Protokolliert die Nachricht mit einem Zeitstempel.

def print_ip():
    public_ip = get('http://api.ipify.org').text  # Gets the public IP address of the machine.  # Holt die öffentliche IP-Adresse des Rechners.
    local_ip = socket.gethostbyname(socket.gethostname())  # Gets the local IP address of the machine.  # Holt die lokale IP-Adresse des Rechners.
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")  # Prints the IP addresses with a timestamp.  # Gibt die IP-Adressen mit Zeitstempel aus.

# BUFFER: ===========================================

class Buffer:
    """
    Buffer of training samples.

    `Server`, `RolloutWorker` and `Trainer` all have their own `Buffer` to store and send training samples.

    Samples are tuples of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
    """
    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN):
        """
        Args:
            maxlen (int): buffer length
        """
        self.memory = []  # Initializes the buffer as an empty list.  # Initialisiert den Puffer als leere Liste.
        self.stat_train_return = 0.0  # stores the train return  # Speichert den Trainingsrückgabewert.
        self.stat_test_return = 0.0  # stores the test return  # Speichert den Testrückgabewert.
        self.stat_train_steps = 0  # stores the number of steps per training episode  # Speichert die Anzahl der Schritte pro Trainingseinheit.
        self.stat_test_steps = 0  # stores the number of steps per test episode  # Speichert die Anzahl der Schritte pro Testeinheit.
        self.maxlen = maxlen  # Sets the maximum length of the buffer.  # Setzt die maximale Länge des Puffers.

    def clip_to_maxlen(self):
        lenmem = len(self.memory)  # Gets the current length of the memory.  # Holt die aktuelle Länge des Speichers.
        if lenmem > self.maxlen:  # Checks if the buffer exceeds the maximum length.  # Überprüft, ob der Puffer die maximale Länge überschreitet.
            print_with_timestamp("buffer overflow. Discarding old samples.")  # Logs a message indicating the overflow.  # Protokolliert eine Nachricht, die den Überlauf anzeigt.
            self.memory = self.memory[(lenmem - self.maxlen):]  # Trims the memory to the maximum length.  # Kürzt den Speicher auf die maximale Länge.

    def append_sample(self, sample):
        """
        Appends `sample` to the buffer.

        Args:
            sample (Tuple): a training sample of the form (`act`, `new_obs`, `rew`, `terminated`, `truncated`, `info`)
        """
        self.memory.append(sample)  # Appends the sample to the memory.  # Fügt das Sample dem Speicher hinzu.
        self.clip_to_maxlen()  # Ensures the buffer does not exceed its maximum length.  # Stellt sicher, dass der Puffer die maximale Länge nicht überschreitet.

    def clear(self):
        """
        Clears the buffer but keeps train and test returns.
        """
        self.memory = []  # Clears the memory.  # Leert den Speicher.

    def __len__(self):
        return len(self.memory)  # Returns the current length of the memory.  # Gibt die aktuelle Länge des Speichers zurück.

    def __iadd__(self, other):
        self.memory += other.memory  # Merges the memory with another buffer's memory.  # Fügt den Speicher eines anderen Puffers hinzu.
        self.clip_to_maxlen()  # Ensures the merged buffer does not exceed its maximum length.  # Stellt sicher, dass der zusammengeführte Puffer die maximale Länge nicht überschreitet.
        self.stat_train_return = other.stat_train_return  # Updates the training return value.  # Aktualisiert den Trainingsrückgabewert.
        self.stat_test_return = other.stat_test_return  # Updates the test return value.  # Aktualisiert den Testrückgabewert.
        self.stat_train_steps = other.stat_train_steps  # Updates the training steps count.  # Aktualisiert die Anzahl der Trainingseinheiten.
        self.stat_test_steps = other.stat_test_steps  # Updates the test steps count.  # Aktualisiert die Anzahl der Testeinheiten.
        return self  # Returns the updated buffer.  # Gibt den aktualisierten Puffer zurück.


class Server:  # Defines the Server class.  # Definiert die Server-Klasse.
    """
    Central server.  # Zentraler Server.

    The `Server` lets 1 `Trainer` and n `RolloutWorkers` connect.  # Der `Server` ermöglicht es einem `Trainer` und n `RolloutWorkers`, sich zu verbinden.
    It buffers experiences sent by workers and periodically sends these to the trainer.  # Er speichert Erfahrungen, die von Arbeitern gesendet werden, und sendet diese regelmäßig an den Trainer.
    It also receives the weights from the trainer and broadcasts these to the connected workers.  # Er empfängt auch die Gewichte vom Trainer und sendet diese an die verbundenen Arbeiter.
    """
    
    def __init__(self,
                 port=cfg.PORT,  # Port für den Server.  # Port for the server.
                 password=cfg.PASSWORD,  # Passwort für den Server.  # Password for the server.
                 local_port=cfg.LOCAL_PORT_SERVER,  # Lokaler Kommunikationsport des Servers.  # Local communication port of the server.
                 header_size=cfg.HEADER_SIZE,  # Größe des Headers in Bytes.  # Header size in bytes.
                 security=cfg.SECURITY,  # Sicherheitsoption für den Server.  # Security option for the server.
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,  # Verzeichnis für Anmeldedaten.  # Directory for credentials.
                 max_workers=cfg.NB_WORKERS):  # Maximale Anzahl an Arbeitern.  # Maximum number of workers.
        """
        Args:  # Argumente für die Server-Klasse.  # Arguments for the Server class:
            port (int): tlspyo public port  # Öffentlicher Port für tlspyo.  # Public port for tlspyo.
            password (str): tlspyo password  # Passwort für tlspyo.  # Password for tlspyo.
            local_port (int): tlspyo local communication port  # Lokaler Kommunikationsport für tlspyo.  # Local communication port for tlspyo.
            header_size (int): tlspyo header size (bytes)  # Größe des Headers in Bytes für tlspyo.  # Header size in bytes for tlspyo.
            security (Union[str, None]): tlspyo security type (None or "TLS")  # Sicherheitsoption für tlspyo.  # Security option for tlspyo.
            keys_dir (str): tlspyo credentials directory  # Verzeichnis für Anmeldedaten von tlspyo.  # Directory for tlspyo credentials.
            max_workers (int): max number of accepted workers  # Maximale Anzahl akzeptierter Arbeiter.  # Maximum number of accepted workers.
        """
        
        self.__relay = Relay(port=port,  # Initialisiert das Relay mit dem Port.  # Initializes the relay with the port.
                             password=password,  # Initialisiert das Relay mit dem Passwort.  # Initializes the relay with the password.
                             accepted_groups={  # Definiert akzeptierte Gruppen für den Relay.  # Defines accepted groups for the relay.
                                 'trainers': {  # Trainer-Gruppe.  # Trainer group.
                                     'max_count': 1,  # Maximale Anzahl an Trainern.  # Maximum number of trainers.
                                     'max_consumables': None},  # Maximale Anzahl an Verbrauchsmaterialien für Trainer.  # Maximum consumables for trainers.
                                 'workers': {  # Arbeiter-Gruppe.  # Workers group.
                                     'max_count': max_workers,  # Maximale Anzahl an Arbeitern.  # Maximum number of workers.
                                     'max_consumables': None}},  # Maximale Anzahl an Verbrauchsmaterialien für Arbeiter.  # Maximum consumables for workers.
                             local_com_port=local_port,  # Definiert den lokalen Kommunikationsport für den Relay.  # Defines the local communication port for the relay.
                             header_size=header_size,  # Setzt die Header-Größe.  # Sets the header size.
                             security=security,  # Setzt die Sicherheitsoption.  # Sets the security option.
                             keys_dir=keys_dir)  # Setzt das Verzeichnis für die Anmeldedaten.  # Sets the directory for credentials.



# TRAINER: ==========================================


class TrainerInterface:  # Defines the TrainerInterface class, which handles communication with the server.  # Definiert die TrainerInterface-Klasse, die die Kommunikation mit dem Server verwaltet.
    """
    This is the trainer's network interface  # This docstring explains that this class serves as the network interface for the trainer.  # Diese Dokumentationszeichenkette erklärt, dass diese Klasse als Netzwerk-Schnittstelle für den Trainer dient.
    This connects to the server  # Describes that the class connects to a server.  # Beschreibt, dass die Klasse mit einem Server verbunden wird.
    This receives samples batches and sends new weights  # Explains that the class receives batches of data and sends new weights.  # Erklärt, dass die Klasse Datenbatches empfängt und neue Gewichtungen sendet.
    """
    def __init__(self,  # The constructor initializes the TrainerInterface object.  # Der Konstruktor initialisiert das TrainerInterface-Objekt.
                 server_ip=None,  # Optional parameter for the server's IP address.  # Optionaler Parameter für die IP-Adresse des Servers.
                 server_port=cfg.PORT,  # Server port defined in config.  # Serverport, der in der Konfiguration definiert ist.
                 password=cfg.PASSWORD,  # Password for authentication.  # Passwort zur Authentifizierung.
                 local_com_port=cfg.LOCAL_PORT_TRAINER,  # Local communication port for the trainer.  # Lokaler Kommunikationsport für den Trainer.
                 header_size=cfg.HEADER_SIZE,  # Size of the header in communication packets.  # Größe des Headers in den Kommunikationspaketen.
                 max_buf_len=cfg.BUFFER_SIZE,  # Maximum buffer size.  # Maximale Pufferspeichergröße.
                 security=cfg.SECURITY,  # Security settings.  # Sicherheitseinstellungen.
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,  # Directory for storing credentials.  # Verzeichnis zum Speichern von Anmeldeinformationen.
                 hostname=cfg.HOSTNAME,  # Hostname for the machine.  # Hostname für die Maschine.
                 model_path=cfg.MODEL_PATH_TRAINER):  # Path to store the model.  # Pfad zum Speichern des Modells.
        
        self.model_path = model_path  # Sets the model path.  # Setzt den Modellpfad.
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'  # Uses the provided IP or defaults to localhost if not provided.  # Verwendet die angegebene IP oder setzt auf localhost, wenn nicht angegeben.
        self.__endpoint = Endpoint(ip_server=self.server_ip,  # Initializes an endpoint for communication with the server.  # Initialisiert einen Endpunkt für die Kommunikation mit dem Server.
                                   port=server_port,  # Sets the server's port.  # Setzt den Serverport.
                                   password=password,  # Passes the password for authentication.  # Übergibt das Passwort zur Authentifizierung.
                                   groups="trainers",  # Defines the group for communication.  # Definiert die Gruppe für die Kommunikation.
                                   local_com_port=local_com_port,  # Local communication port for the trainer.  # Lokaler Kommunikationsport für den Trainer.
                                   header_size=header_size,  # Defines the header size in communication packets.  # Definiert die Headergröße in den Kommunikationspaketen.
                                   max_buf_len=max_buf_len,  # Sets the buffer size.  # Setzt die Puffergröße.
                                   security=security,  # Sets the security settings.  # Setzt die Sicherheitseinstellungen.
                                   keys_dir=keys_dir,  # Specifies the directory for credentials.  # Gibt das Verzeichnis für Anmeldeinformationen an.
                                   hostname=hostname)  # Specifies the machine's hostname.  # Gibt den Hostnamen der Maschine an.
        
        print_with_timestamp(f"server IP: {self.server_ip}")  # Prints the server's IP address with a timestamp.  # Gibt die IP-Adresse des Servers mit einem Zeitstempel aus.
        
        self.__endpoint.notify(groups={'trainers': -1})  # Sends a notification to the 'trainers' group to retrieve all data.  # Sendet eine Benachrichtigung an die Gruppe 'trainers', um alle Daten abzurufen.

    def broadcast_model(self, model: ActorModule):  # Defines a method to broadcast the model's weights to workers.  # Definiert eine Methode, um die Gewichtungen des Modells an die Arbeiter zu übertragen.
        """
        model must be an ActorModule  # Ensures that the model is of type ActorModule.  # Stellt sicher, dass das Modell vom Typ ActorModule ist.
        broadcasts the model's weights to all connected RolloutWorkers  # Sends the model's weights to all connected workers.  # Überträgt die Gewichtungen des Modells an alle verbundenen Arbeiter.
        """
        model.save(self.model_path)  # Saves the model at the specified path.  # Speichert das Modell am angegebenen Pfad.
        with open(self.model_path, 'rb') as f:  # Opens the saved model file in read-binary mode.  # Öffnet die gespeicherte Modell-Datei im Lese-Binärmodus.
            weights = f.read()  # Reads the weights from the model file.  # Liest die Gewichtungen aus der Modell-Datei.
        self.__endpoint.broadcast(weights, "workers")  # Sends the model's weights to the workers.  # Überträgt die Gewichtungen des Modells an die Arbeiter.

    def retrieve_buffer(self):  # Defines a method to retrieve training sample buffers from the server.  # Definiert eine Methode, um Trainingsprobenpuffer vom Server abzurufen.
        """
        returns the TrainerInterface's buffer of training samples  # Returns the collected training samples.  # Gibt die gesammelten Trainingsproben zurück.
        """
        buffers = self.__endpoint.receive_all()  # Receives all the buffers from the server.  # Empfängt alle Puffer vom Server.
        res = Buffer()  # Initializes an empty buffer to store the samples.  # Initialisiert einen leeren Puffer, um die Proben zu speichern.
        for buf in buffers:  # Iterates over each received buffer.  # Iteriert über jeden empfangenen Puffer.
            res += buf  # Adds the current buffer to the result buffer.  # Fügt den aktuellen Puffer dem Ergebnis-Puffer hinzu.
        self.__endpoint.notify(groups={'trainers': -1})  # Sends a notification to the 'trainers' group to retrieve more data.  # Sendet eine Benachrichtigung an die Gruppe 'trainers', um mehr Daten abzurufen.
        return res  # Returns the complete buffer of training samples.  # Gibt den vollständigen Puffer der Trainingsproben zurück.

def log_environment_variables():  # Defines a method to log environment variables.  # Definiert eine Methode zum Protokollieren von Umgebungsvariablen.
    """
    add certain relevant environment variables to our config  # Adds specific environment variables to the configuration.  # Fügt bestimmte Umgebungsvariablen zur Konfiguration hinzu.
    usage: `LOG_VARIABLES='HOME JOBID' python ...`  # Usage example for setting which variables to log.  # Beispiel für die Verwendung, um anzugeben, welche Variablen protokolliert werden sollen.
    """
    return {k: os.environ.get(k, '') for k in os.environ.get('LOG_VARIABLES', '').strip().split()}  # Retrieves and returns the specified environment variables.  # Ruft die angegebenen Umgebungsvariablen ab und gibt sie zurück.

def load_run_instance(checkpoint_path):  # Defines a function to load a training run instance from a checkpoint.  # Definiert eine Funktion zum Laden einer Trainingsdurchführung von einem Checkpoint.
    """
    Default function used to load trainers from checkpoint path  # Default function to load a trainer from a checkpoint path.  # Standardfunktion zum Laden eines Trainers vom Checkpoint-Pfad.
    Args:  # Explains the arguments.  # Erklärt die Argumente.
        checkpoint_path: the path where instances of run_cls are checkpointed  # Path where the run instance is saved.  # Pfad, an dem die Run-Instanz gespeichert wird.
    Returns:  # Explains the return value.  # Erklärt den Rückgabewert.
        An instance of run_cls loaded from checkpoint_path  # Returns a loaded instance of the run_cls.  # Gibt eine geladene Instanz von run_cls zurück.
    """
    return load(checkpoint_path)  # Loads the checkpointed instance from the given path.  # Lädt die checkpointed Instanz vom angegebenen Pfad.

def dump_run_instance(run_instance, checkpoint_path):  # Defines a function to dump the training instance to a checkpoint path.  # Definiert eine Funktion, um die Trainingsinstanz in einen Checkpoint-Pfad zu speichern.
    """
    Default function used to dump trainers to checkpoint path  # Default function to save a trainer to a checkpoint path.  # Standardfunktion zum Speichern eines Trainers an einem Checkpoint-Pfad.
    Args:  # Explains the arguments.  # Erklärt die Argumente.
        run_instance: the instance of run_cls to checkpoint  # The instance of run_cls to be saved.  # Die Instanz von run_cls, die gespeichert werden soll.
        checkpoint_path: the path where instances of run_cls are checkpointed  # Path where the instance will be saved.  # Pfad, an dem die Instanz gespeichert wird.
    """
    dump(run_instance, checkpoint_path)  # Dumps the run instance to the checkpoint path.  # Speichert die Run-Instanz im Checkpoint-Pfad.

def iterate_epochs(run_cls,  # Defines the main training loop method.  # Definiert die Haupttrainingsschleifen-Methode.
                   interface: TrainerInterface,  # Takes TrainerInterface as an argument to handle communication.  # Nimmt TrainerInterface als Argument, um die Kommunikation zu verwalten.
                   checkpoint_path: str,  # Path for saving checkpoints.  # Pfad zum Speichern von Checkpoints.
                   dump_run_instance_fn=dump_run_instance,  # Default function to dump the run instance to a checkpoint.  # Standardfunktion zum Speichern der Run-Instanz in einem Checkpoint.
                   load_run_instance_fn=load_run_instance,  # Default function to load the run instance from a checkpoint.  # Standardfunktion zum Laden der Run-Instanz aus einem Checkpoint.
                   epochs_between_checkpoints=1,  # Number of epochs between saving checkpoints.  # Anzahl der Epochen zwischen dem Speichern von Checkpoints.
                   updater_fn=None):  # Optional function to update the run instance before saving it.  # Optionale Funktion, um die Run-Instanz vor dem Speichern zu aktualisieren.
    """
    Main training loop (remote)  # Main function that handles the training loop, including checkpointing.  # Hauptfunktion, die die Trainingsschleife verwaltet, einschließlich Checkpoints.
    The run_cls instance is saved in checkpoint_path at the end of each epoch  # Saves the run_cls instance at the end of each epoch.  # Speichert die Run-Instanz am Ende jeder Epoche im Checkpoint-Pfad.
    The model weights are sent to the RolloutWorker every model_checkpoint_interval epochs  # Sends model weights to RolloutWorkers periodically.  # Überträgt Modellgewichtungen periodisch an die Rollout-Arbeiter.
    Generator yielding episode statistics (list of pd.Series) while running and checkpointing  # Yields statistics about each episode as the training progresses.  # Gibt Statistiken zu jeder Episode während des Trainings und Checkpoints zurück.
    """
    checkpoint_path = checkpoint_path or tempfile.mktemp("_remove_on_exit")  # Creates a temporary path if no checkpoint path is provided.  # Erstellt einen temporären Pfad, wenn kein Checkpoint-Pfad angegeben wird.

    try:
        logging.debug(f"checkpoint_path: {checkpoint_path}")  # Logs the checkpoint path.  # Protokolliert den Checkpoint-Pfad.
        if not exists(checkpoint_path):  # Checks if the checkpoint path exists.  # Überprüft, ob der Checkpoint-Pfad existiert.
            logging.info(f"=== specification ".ljust(70, "="))  # Logs a message about the specification.  # Protokolliert eine Nachricht zur Spezifikation.
            run_instance = run_cls()  # Creates a new instance of the run_cls.  # Erstellt eine neue Instanz der run_cls.
            dump_run_instance_fn(run_instance, checkpoint_path)  # Dumps the new instance to the checkpoint path.  # Speichert die neue Instanz im Checkpoint-Pfad.
            logging.info(f"")  # Logs an empty message.  # Protokolliert eine leere Nachricht.
        else:
            logging.info(f"Loading checkpoint...")  # Logs that the checkpoint is being loaded.  # Protokolliert, dass der Checkpoint geladen wird.
            t1 = time.time()  # Records the time before loading the checkpoint.  # Zeichnet die Zeit vor dem Laden des Checkpoints auf.
            run_instance = load_run_instance_fn(checkpoint_path)  # Loads the checkpoint from the given path.  # Lädt den Checkpoint vom angegebenen Pfad.
            logging.info(f" Loaded checkpoint in {time.time() - t1} seconds.")  # Logs the time taken to load the checkpoint.  # Protokolliert die Zeit, die zum Laden des Checkpoints benötigt wurde.
            if updater_fn is not None:  # Checks if an updater function is provided.  # Überprüft, ob eine Aktualisierungsfunktion bereitgestellt wird.
                logging.info(f"Updating checkpoint...")  # Logs that the checkpoint is being updated.  # Protokolliert, dass der Checkpoint aktualisiert wird.
                t1 = time.time()  # Records the time before updating the checkpoint.  # Zeichnet die Zeit vor der Aktualisierung des Checkpoints auf.
                run_instance = updater_fn(run_instance, run_cls)  # Updates the run instance.  # Aktualisiert die Run-Instanz.
                logging.info(f"Checkpoint updated in {time.time() - t1} seconds.")  # Logs the time taken to update the checkpoint.  # Protokolliert die Zeit, die für die Aktualisierung des Checkpoints benötigt wurde.

        while run_instance.epoch < run_instance.epochs:  # Starts the training loop for each epoch.  # Startet die Trainingsschleife für jede Epoche.
            yield run_instance.run_epoch(interface=interface)  # Runs an epoch and yields the statistics.  # Führt eine Epoche aus und gibt die Statistiken zurück.
            if run_instance.epoch % epochs_between_checkpoints == 0:  # Checks if it's time to save a checkpoint.  # Überprüft, ob es Zeit ist, einen Checkpoint zu speichern.
                logging.info(f" saving checkpoint...")  # Logs that a checkpoint is being saved.  # Protokolliert, dass ein Checkpoint gespeichert wird.
                t1 = time.time()  # Records the time before saving the checkpoint.  # Zeichnet die Zeit vor dem Speichern des Checkpoints auf.
                dump_run_instance_fn(run_instance, checkpoint_path)  # Saves the checkpoint.  # Speichert den Checkpoint.
                logging.info(f" saved checkpoint in {time.time() - t1} seconds.")  # Logs the time taken to save the checkpoint.  # Protokolliert die Zeit, die zum Speichern des Checkpoints benötigt wurde.
    finally:
        if checkpoint_path.endswith("_remove_on_exit") and exists(checkpoint_path):  # Checks if the temporary checkpoint path exists and needs to be removed.  # Überprüft, ob der temporäre Checkpoint-Pfad existiert und entfernt werden muss.
            os.remove(checkpoint_path)  # Removes the checkpoint file.  # Entfernt die Checkpoint-Datei.





def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    
    saves config and stats to https://wandb.com
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance  # Set function to dump the run instance, default to `dump_run_instance` if not provided.  # Setze die Funktion zum Speichern der Run-Instanz, standardmäßig `dump_run_instance`, wenn nicht angegeben.
    load_run_instance_fn = load_run_instance_fn or load_run_instance  # Set function to load the run instance, default to `load_run_instance` if not provided.  # Setze die Funktion zum Laden der Run-Instanz, standardmäßig `load_run_instance`, wenn nicht angegeben.
    wandb_dir = tempfile.mkdtemp()  # Prevent wandb from polluting the home directory by creating a temporary directory.  # Verhindert, dass wandb das Home-Verzeichnis verschmutzt, indem ein temporäres Verzeichnis erstellt wird.
    atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # Registers cleanup to remove the temporary directory when the program exits.  # Registriert die Bereinigung, um das temporäre Verzeichnis beim Programmbeenden zu entfernen.
    import wandb  # Import the `wandb` library for logging metrics and configs.  # Importiert die `wandb`-Bibliothek zum Protokollieren von Metriken und Konfigurationen.
    logging.debug(f" run_cls: {run_cls}")  # Logs the class of the training run for debugging.  # Protokolliert die Klasse des Trainingslaufs zum Debuggen.
    config = partial_to_dict(run_cls)  # Converts the training class configuration into a dictionary.  # Wandelt die Konfiguration der Trainingsklasse in ein Dictionary um.
    config['environ'] = log_environment_variables()  # Logs environment variables and adds them to the configuration dictionary.  # Protokolliert Umgebungsvariablen und fügt sie dem Konfigurations-Dictionary hinzu.
    resume = checkpoint_path and exists(checkpoint_path)  # Checks if a checkpoint exists and if the `checkpoint_path` is valid.  # Überprüft, ob ein Checkpoint existiert und ob der `checkpoint_path` gültig ist.
    wandb_initialized = False  # Flag to track if wandb has been successfully initialized.  # Flag, um zu überprüfen, ob wandb erfolgreich initialisiert wurde.
    err_cpt = 0  # Error counter for handling retries.  # Fehlerzähler zur Handhabung von Wiederholungsversuchen.
    while not wandb_initialized:  # Continues trying to initialize wandb until successful.  # Versucht weiterhin, wandb zu initialisieren, bis es erfolgreich ist.
        try:
            wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)  # Initializes a wandb run with the provided configuration.  # Initialisiert einen wandb-Lauf mit der angegebenen Konfiguration.
            wandb_initialized = True  # Sets the flag to True once initialization is successful.  # Setzt das Flag auf True, wenn die Initialisierung erfolgreich ist.
        except Exception as e:  # Catches exceptions if wandb fails to initialize.  # Fängt Ausnahmen ab, wenn wandb die Initialisierung nicht abschließen kann.
            err_cpt += 1  # Increments the error counter.  # Erhöht den Fehlerzähler.
            logging.warning(f"wandb error {err_cpt}: {e}")  # Logs the error message for debugging.  # Protokolliert die Fehlermeldung zum Debuggen.
            if err_cpt > 10:  # If there are more than 10 failed attempts, it aborts the process.  # Wenn mehr als 10 fehlgeschlagene Versuche auftreten, wird der Prozess abgebrochen.
                logging.warning(f"Could not connect to wandb, aborting.")  # Logs a warning and exits.  # Protokolliert eine Warnung und beendet das Programm.
                exit()  
            else:
                time.sleep(10.0)  # Waits for 10 seconds before retrying.  # Wartet 10 Sekunden, bevor ein neuer Versuch gestartet wird.
    # logging.info(config)  # Uncomment this line to log the config for information purposes.  # Kommentiere diese Zeile ein, um die Konfiguration zu Informationszwecken zu protokollieren.
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):  # Iterates through training epochs.  # Iteriert durch die Trainings-Epochen.
        [wandb.log(json.loads(s.to_json())) for s in stats]  # Logs the stats of each epoch to wandb.  # Protokolliert die Statistiken jeder Epoche an wandb.

def run(interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
    """
    Main training loop (remote).
    """
    dump_run_instance_fn = dump_run_instance_fn or dump_run_instance  # Sets function to dump the run instance, default to `dump_run_instance` if not provided.  # Setze die Funktion zum Speichern der Run-Instanz, standardmäßig `dump_run_instance`, wenn nicht angegeben.
    load_run_instance_fn = load_run_instance_fn or load_run_instance  # Sets function to load the run instance, default to `load_run_instance` if not provided.  # Setze die Funktion zum Laden der Run-Instanz, standardmäßig `load_run_instance`, wenn nicht angegeben.
    for stats in iterate_epochs(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):  # Iterates through training epochs.  # Iteriert durch die Trainings-Epochen.
        pass  # No action here, just a placeholder.  # Keine Aktion hier, nur ein Platzhalter.



class Trainer:  # Defines the Trainer class responsible for training.  # Definiert die Trainer-Klasse, die für das Training verantwortlich ist.
    """
    Training entity.

    The `Trainer` object is where RL training happens.
    Typically, it can be located on a HPC cluster.
    """  # A docstring that describes the Trainer class.  # Eine docstring, die die Trainer-Klasse beschreibt.

    def __init__(self,  # The initializer method that sets up the Trainer instance.  # Der Initialisierer, der die Trainer-Instanz einrichtet.
                 training_cls=cfg_obj.TRAINER,  # Default training class used for training.  # Standard-Trainingsklasse, die für das Training verwendet wird.
                 server_ip=cfg.SERVER_IP_FOR_TRAINER,  # IP address of the central server.  # IP-Adresse des zentralen Servers.
                 server_port=cfg.PORT,  # Port used by the central server.  # Port, der vom zentralen Server verwendet wird.
                 password=cfg.PASSWORD,  # Password for authenticating with the server.  # Passwort zur Authentifizierung mit dem Server.
                 local_com_port=cfg.LOCAL_PORT_TRAINER,  # Local communication port used by tlspyo.  # Lokaler Kommunikationsport, der von tlspyo verwendet wird.
                 header_size=cfg.HEADER_SIZE,  # Size of headers used by tlspyo.  # Größe der Header, die von tlspyo verwendet werden.
                 max_buf_len=cfg.BUFFER_SIZE,  # Maximum buffer length for messages in tlspyo.  # Maximale Pufferlänge für Nachrichten in tlspyo.
                 security=cfg.SECURITY,  # Security type for tlspyo (None or "TLS").  # Sicherheitstyp für tlspyo (None oder "TLS").
                 keys_dir=cfg.CREDENTIALS_DIRECTORY,  # Directory for credentials (used for TLS).  # Verzeichnis für Anmeldeinformationen (wird für TLS verwendet).
                 hostname=cfg.HOSTNAME,  # Hostname for TLS.  # Hostname für TLS.
                 model_path=cfg.MODEL_PATH_TRAINER,  # Path where the local model copy will be stored.  # Pfad, in dem die lokale Modellkopie gespeichert wird.
                 checkpoint_path=cfg.CHECKPOINT_PATH,  # Path where the checkpoint will be saved.  # Pfad, in dem der Checkpoint gespeichert wird.
                 dump_run_instance_fn: callable = None,  # Function to serialize the run instance.  # Funktion zum Serialisieren der Run-Instanz.
                 load_run_instance_fn: callable = None,  # Function to deserialize the run instance.  # Funktion zum Deserialisieren der Run-Instanz.
                 updater_fn: callable = None):  # Function to update the checkpoint after training.  # Funktion zum Aktualisieren des Checkpoints nach dem Training.
        """
        Args:
            training_cls (type): training class (subclass of tmrl.training_offline.TrainingOffline)  # Training class type.  # Trainingsklassentyp.
            server_ip (str): ip of the central `Server`  # IP-Adresse des zentralen Servers.  # Die IP-Adresse des zentralen Servers.
            server_port (int): public port of the central `Server`  # Der öffentliche Port des zentralen Servers.  # Port des zentralen Servers.
            password (str): password of the central `Server`  # Passwort des zentralen Servers.  # Das Passwort des zentralen Servers.
            local_com_port (int): port used by `tlspyo` for local communication  # Der von tlspyo für die lokale Kommunikation verwendete Port.  # Port für lokale Kommunikation.
            header_size (int): number of bytes used for `tlspyo` headers  # Die Anzahl der Bytes, die für tlspyo-Header verwendet werden.  # Headergröße in Bytes.
            max_buf_len (int): maximum number of messages queued by `tlspyo`  # Maximale Anzahl von Nachrichten, die in tlspyo gepuffert werden können.  # Maximale Puffermenge.
            security (str): `tlspyo security type` (None or "TLS")  # Sicherheitstyp für tlspyo.  # Sicherheitstyp von tlspyo.
            keys_dir (str): custom credentials directory for `tlspyo` TLS security  # Verzeichnis für benutzerdefinierte Anmeldeinformationen für TLS.  # Benutzerspezifisches Verzeichnis für TLS.
            hostname (str): custom TLS hostname  # Benutzerdefinierter TLS-Hostname.  # Benutzerdefinierter TLS-Hostname.
            model_path (str): path where a local copy of the model will be saved  # Pfad, in dem das Modell lokal gespeichert wird.  # Pfad zur lokalen Modellkopie.
            checkpoint_path: path where the `Trainer` will be checkpointed (`None` = no checkpointing)  # Pfad für Checkpoint-Speicherung.  # Checkpoint-Pfad für Trainer.
            dump_run_instance_fn (callable): custom serializer (`None` = pickle.dump)  # Funktion zum Speichern der Run-Instanz.  # Eigene Serialisierungsfunktion.
            load_run_instance_fn (callable): custom deserializer (`None` = pickle.load)  # Funktion zum Laden der Run-Instanz.  # Eigene Deserialisierungsfunktion.
            updater_fn (callable): custom updater (`None` = no updater). If provided, this must be a function  # Funktion zum Aktualisieren des Checkpoints.  # Aktualisierungsfunktion für den Checkpoint.
        """
        self.checkpoint_path = checkpoint_path  # Stores the checkpoint path for the Trainer.  # Speichert den Checkpoint-Pfad für den Trainer.
        self.dump_run_instance_fn = dump_run_instance_fn  # Stores the function to dump the run instance.  # Speichert die Funktion zum Speichern der Run-Instanz.
        self.load_run_instance_fn = load_run_instance_fn  # Stores the function to load the run instance.  # Speichert die Funktion zum Laden der Run-Instanz.
        self.updater_fn = updater_fn  # Stores the function to update the checkpoint.  # Speichert die Funktion zum Aktualisieren des Checkpoints.
        self.training_cls = training_cls  # Stores the training class used for training.  # Speichert die Trainingsklasse, die für das Training verwendet wird.
        self.interface = TrainerInterface(server_ip=server_ip,  # Initializes the TrainerInterface with the provided configuration.  # Initialisiert das TrainerInterface mit der angegebenen Konfiguration.
                                          server_port=server_port,  # Initializes the server port for communication.  # Initialisiert den Server-Port für die Kommunikation.
                                          password=password,  # Initializes the password for the server.  # Initialisiert das Passwort für den Server.
                                          local_com_port=local_com_port,  # Initializes the local communication port.  # Initialisiert den lokalen Kommunikationsport.
                                          header_size=header_size,  # Initializes the header size for communication.  # Initialisiert die Headergröße für die Kommunikation.
                                          max_buf_len=max_buf_len,  # Initializes the maximum buffer length.  # Initialisiert die maximale Pufferlänge.
                                          security=security,  # Initializes the security settings for communication.  # Initialisiert die Sicherheitseinstellungen für die Kommunikation.
                                          keys_dir=keys_dir,  # Initializes the directory for TLS credentials.  # Initialisiert das Verzeichnis für TLS-Anmeldeinformationen.
                                          hostname=hostname,  # Initializes the hostname for TLS.  # Initialisiert den Hostnamen für TLS.
                                          model_path=model_path)  # Initializes the model path.  # Initialisiert den Modellpfad.

    def run(self):  # Defines the method that runs the training process.  # Definiert die Methode, die den Trainingsprozess startet.
        """
        Runs training.
        """
        run(interface=self.interface,  # Calls the run method with the provided interface and configuration.  # Ruft die run-Methode mit der angegebenen Schnittstelle und Konfiguration auf.
            run_cls=self.training_cls,  # Specifies the training class for the run.  # Gibt die Trainingsklasse für den Lauf an.
            checkpoint_path=self.checkpoint_path,  # Provides the checkpoint path.  # Gibt den Checkpoint-Pfad an.
            dump_run_instance_fn=self.dump_run_instance_fn,  # Provides the function to serialize the run instance.  # Gibt die Funktion zur Serialisierung der Run-Instanz an.
            load_run_instance_fn=self.load_run_instance_fn,  # Provides the function to deserialize the run instance.  # Gibt die Funktion zum Deserialisieren der Run-Instanz an.
            updater_fn=self.updater_fn)  # Provides the function to update the checkpoint.  # Gibt die Funktion zum Aktualisieren des Checkpoints an.

    def run_with_wandb(self,  # Defines the method that runs training with wandb logging.  # Definiert die Methode, die das Training mit wandb-Protokollierung startet.
                       entity=cfg.WANDB_ENTITY,  # Defines the wandb entity to log data.  # Definiert die wandb-Entität für die Protokollierung.
                       project=cfg.WANDB_PROJECT,  # Defines the wandb project for the run.  # Definiert das wandb-Projekt für den Lauf.
                       run_id=cfg.WANDB_RUN_ID,  # Defines the unique ID for the run in wandb.  # Definiert die eindeutige ID für den Lauf in wandb.
                       key=None):  # Optionally accepts a wandb API key.  # Optionale Eingabe eines wandb-API-Schlüssels.
        
                           
"""
Runs training while logging metrics to wandb_.
        
.. _wandb: https://wandb.ai

Args:
    entity (str): wandb entity
    project (str): wandb project
    run_id (str): name of the run
    key (str): wandb API key
"""
if key is not None:  # If an API key is provided, set it as an environment variable.  # Wenn ein API-Schlüssel bereitgestellt wird, setze ihn als Umgebungsvariable.
    os.environ['WANDB_API_KEY'] = key  # Set the environment variable 'WANDB_API_KEY' to the provided key.  # Setze die Umgebungsvariable 'WANDB_API_KEY' auf den bereitgestellten Schlüssel.
run_with_wandb(entity=entity,  # Passes the wandb entity (user or team) to the function.  # Übergibt die wandb-Entität (Benutzer oder Team) an die Funktion.
               project=project,  # Passes the project name for wandb logging.  # Übergibt den Projektnamen für die wandb-Protokollierung.
               run_id=run_id,  # Passes the unique identifier for the current run.  # Übergibt die eindeutige Kennung für den aktuellen Lauf.
               interface=self.interface,  # Passes the interface used for training.  # Übergibt die Schnittstelle, die für das Training verwendet wird.
               run_cls=self.training_cls,  # Passes the class or function that runs the training process.  # Übergibt die Klasse oder Funktion, die den Trainingsprozess ausführt.
               checkpoint_path=self.checkpoint_path,  # Passes the path where model checkpoints are saved.  # Übergibt den Pfad, wo Modell-Checkpoints gespeichert werden.
               dump_run_instance_fn=self.dump_run_instance_fn,  # Passes the function that saves the current run's state.  # Übergibt die Funktion, die den aktuellen Laufzustand speichert.
               load_run_instance_fn=self.load_run_instance_fn,  # Passes the function that loads the run's state from a checkpoint.  # Übergibt die Funktion, die den Laufzustand aus einem Checkpoint lädt.
               updater_fn=self.updater_fn)  # Passes the function that updates the model during training.  # Übergibt die Funktion, die das Modell während des Trainings aktualisiert.


# ROLLOUT WORKER: ===================================

class RolloutWorker:  # Defines the RolloutWorker class, which deploys the current policy in the environment.  # Definiert die Klasse RolloutWorker, die die aktuelle Politik in der Umgebung einsetzt.
    """Actor.  # Documentation for the class.  # Dokumentation der Klasse.

    A `RolloutWorker` deploys the current policy in the environment.  # A RolloutWorker runs the policy in the environment.  # Ein `RolloutWorker` setzt die aktuelle Politik in der Umgebung um.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.  # It can send experience data to a server.  # Ein `RolloutWorker` kann sich mit einem `Server` verbinden, an den er gepufferte Erfahrungen sendet.
    Alternatively, it may exist in standalone mode for deployment.  # It can also work in standalone mode without a server.  # Alternativ kann es im Standalone-Modus arbeiten, ohne einen Server zu benötigen.
    """
    def __init__(  # The constructor for initializing a RolloutWorker instance.  # Der Konstruktor zur Initialisierung einer RolloutWorker-Instanz.
            self,
            env_cls,  # The environment class type (usually a Gym environment).  # Der Klassentyp der Umgebung (normalerweise eine Gym-Umgebung).
            actor_module_cls,  # The actor module class that contains the policy.  # Die Actor-Modulklasse, die die Politik enthält.
            sample_compressor: callable = None,  # Optional: Function to compress experience samples for transmission.  # Optional: Funktion zur Kompression von Erfahrungssamples für die Übertragung.
            device="cpu",  # Device for running the policy (default is CPU).  # Gerät, auf dem die Politik ausgeführt wird (Standard ist CPU).
            max_samples_per_episode=np.inf,  # Maximum number of samples per episode before resetting.  # Maximale Anzahl von Samples pro Episode, bevor diese zurückgesetzt wird.
            model_path=cfg.MODEL_PATH_WORKER,  # Path where the model is stored.  # Pfad, in dem das Modell gespeichert wird.
            obs_preprocessor: callable = None,  # Optional: Function to preprocess observations.  # Optional: Funktion zur Vorverarbeitung der Beobachtungen.
            crc_debug=False,  # Debug flag for custom pipelines.  # Debug-Flag für benutzerdefinierte Pipelines.
            model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,  # Path to the history of saved models.  # Pfad zur Historie der gespeicherten Modelle.
            model_history=cfg.MODEL_HISTORY,  # Frequency of saving models.  # Häufigkeit der Modellspeicherung.
            standalone=False,  # Flag to indicate if the worker runs independently (no server).  # Flag, das angibt, ob der Worker unabhängig läuft (kein Server).
            server_ip=None,  # IP address of the server to connect to.  # IP-Adresse des Servers, zu dem eine Verbindung hergestellt werden soll.
            server_port=cfg.PORT,  # Port number for the server.  # Portnummer für den Server.
            password=cfg.PASSWORD,  # Password for secure connection.  # Passwort für die sichere Verbindung.
            local_port=cfg.LOCAL_PORT_WORKER,  # Local port for worker communication.  # Lokaler Port für die Kommunikation des Workers.
            header_size=cfg.HEADER_SIZE,  # Header size for communication.  # Header-Größe für die Kommunikation.
            max_buf_len=cfg.BUFFER_SIZE,  # Maximum buffer length for messages.  # Maximale Puffergröße für Nachrichten.
            security=cfg.SECURITY,  # Security type (e.g., TLS).  # Sicherheitstyp (z. B. TLS).
            keys_dir=cfg.CREDENTIALS_DIRECTORY,  # Directory containing security keys.  # Verzeichnis mit Sicherheitsschlüsseln.
            hostname=cfg.HOSTNAME  # Hostname for the worker.  # Hostname des Workers.
    ):
        """Initialize the worker with provided parameters.  # Initialisiert den Worker mit den angegebenen Parametern.
        """
        self.obs_preprocessor = obs_preprocessor  # Store the observation preprocessor function.  # Speichert die Funktion zur Vorverarbeitung der Beobachtungen.
        self.get_local_buffer_sample = sample_compressor  # Store the sample compressor function.  # Speichert die Funktion zur Kompression von Samples.
        self.env = env_cls()  # Initialize the environment instance.  # Initialisiert die Instanz der Umgebung.
        obs_space = self.env.observation_space  # Get the observation space of the environment.  # Holt den Beobachtungsraum der Umgebung.
        act_space = self.env.action_space  # Get the action space of the environment.  # Holt den Aktionsraum der Umgebung.
        self.model_path = model_path  # Store the path to the model.  # Speichert den Pfad zum Modell.
        self.model_path_history = model_path_history  # Store the path to the model history.  # Speichert den Pfad zur Modellhistorie.
        self.device = device  # Store the device type for running the model.  # Speichert den Gerätetyp für die Ausführung des Modells.
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to_device(self.device)  # Initialize the actor with the environment's spaces and move to the selected device.  # Initialisiert den Actor mit den Umgebungsräumen und verschiebt ihn auf das ausgewählte Gerät.
        self.standalone = standalone  # Set whether the worker is in standalone mode.  # Setzt, ob der Worker im Standalone-Modus ist.
        if os.path.isfile(self.model_path):  # Check if a model file exists at the given path.  # Überprüft, ob eine Modell-Datei am angegebenen Pfad existiert.
            logging.debug(f"Loading model from {self.model_path}")  # Log a message indicating the model is being loaded.  # Protokolliert eine Nachricht, dass das Modell geladen wird.
            self.actor = self.actor.load(self.model_path, device=self.device)  # Load the model from the file.  # Lädt das Modell aus der Datei.
        else:  # If no model file is found, log a message.  # Wenn keine Modell-Datei gefunden wird, wird eine Nachricht protokolliert.
            logging.debug(f"No model found at {self.model_path}")  # Log a message indicating no model was found.  # Protokolliert eine Nachricht, dass kein Modell gefunden wurde.
        self.buffer = Buffer()  # Initialize a buffer to store experience.  # Initialisiert einen Puffer zum Speichern von Erfahrungen.
        self.max_samples_per_episode = max_samples_per_episode  # Set the maximum samples per episode.  # Setzt die maximale Anzahl von Samples pro Episode.
        self.crc_debug = crc_debug  # Store the debug flag.  # Speichert das Debug-Flag.
        self.model_history = model_history  # Set the model history saving frequency.  # Setzt die Häufigkeit der Modellhistorien-Speicherung.
        self._cur_hist_cpt = 0  # Initialize a counter for model history.  # Initialisiert einen Zähler für die Modellhistorie.
        self.model_cpt = 0  # Initialize a counter for models.  # Initialisiert einen Zähler für Modelle.
        self.debug_ts_cpt = 0  # Initialize a counter for debug timestamps.  # Initialisiert einen Zähler für Debug-Timestamps.
        self.debug_ts_res_cpt = 0  # Initialize a counter for debug timestamp results.  # Initialisiert einen Zähler für Debug-Timestamp-Ergebnisse.
        self.server_ip = server_ip if server_ip is not None else '127.0.0.1'  # Set the server IP, defaulting to localhost.  # Setzt die Server-IP, standardmäßig auf localhost.
        print_with_timestamp(f"server IP: {self.server_ip}")  # Print the server IP with a timestamp.  # Gibt die Server-IP mit einem Timestamp aus.
        if not self.standalone:  # If not in standalone mode, initialize endpoint communication.  # Wenn nicht im Standalone-Modus, wird die Endpunktkommunikation initialisiert.
            self.__endpoint = Endpoint(ip_server=self.server_ip,  # Create an endpoint for communication with the server.  # Erstellt einen Endpunkt für die Kommunikation mit dem Server.
                                       port=server_port,  # Set the server port.  # Setzt den Server-Port.
                                       password=password,  # Set the password for secure communication.  # Setzt das Passwort für die sichere Kommunikation.
                                       groups="workers",  # Define the worker group.  # Definiert die Worker-Gruppe.
                                       local_com_port=local_port,  # Set the local communication port.  # Setzt den lokalen Kommunikationsport.
                                       header_size=header_size,  # Set the header size for messages.  # Setzt die Header-Größe für Nachrichten.
                                       max_buf_len=max_buf_len,  # Set the maximum buffer length for messages.  # Setzt die maximale Pufferlänge für Nachrichten.
                                       security=security,  # Set the security type for communication.  # Setzt den Sicherheitstyp für die Kommunikation.
                                       keys_dir=keys_dir,  # Set the directory for security keys.  # Setzt das Verzeichnis für Sicherheitsschlüssel.
                                       hostname=hostname,  # Set the hostname for the communication.  # Setzt den Hostnamen für die Kommunikation.
                                       deserializer_mode="synchronous")  # Set communication mode to synchronous.  # Setzt den Kommunikationsmodus auf synchron.
        else:  # If in standalone mode, no endpoint is needed.  # Wenn im Standalone-Modus, ist kein Endpunkt erforderlich.
            self.__endpoint = None  # No endpoint initialized for standalone mode.  # Kein Endpunkt für den Standalone-Modus initialisiert.

    def act(self, obs, test=False):  # Method for selecting an action based on the observation.  # Methode zur Auswahl einer Aktion basierend auf der Beobachtung.
        """
        Select an action based on observation `obs`  # Beschreibung der Methode zur Auswahl einer Aktion.
        Args:  # Eingabewerte der Methode:
            obs (nested structure): observation  # Observation received from the environment.  # Beobachtung, die von der Umgebung empfangen wurde.
            test (bool): directly passed to the `act()` method of the `ActorModule`  # Test flag for passing to the ActorModule's act method.  # Test-Flag, das an die Act-Methode des ActorModules weitergegeben wird.
        Returns:  # Rückgabewerte der Methode:
            numpy.array: action computed by the `ActorModule`  # The action decided by the ActorModule.  # Die Aktion, die vom ActorModule berechnet wurde.
        """
        action = self.actor.act_(obs, test=test)  # Use the actor to select an action based on the observation.  # Verwendet den Actor, um eine Aktion basierend auf der Beobachtung auszuwählen.
        return action  # Return the selected action.  # Gibt die ausgewählte Aktion zurück.




def reset(self, collect_samples):  
    """
    Starts a new episode.  
    # Startet eine neue Episode.
    """  

    obs = None  # Initialize observation as None.  # Initialisiert die Beobachtung als None.
    try:  
        # Faster than hasattr() in real-time environments  
        act = self.env.unwrapped.default_action  # Retrieve the default action from the environment.  # Ruft die Standardaktion aus der Umgebung ab.
    except AttributeError:  
        # In non-real-time environments, act is None on reset  
        act = None  # Wenn ein Fehler auftritt, setze die Aktion auf None.  
    new_obs, info = self.env.reset()  # Reset the environment and retrieve new observation and info.  # Setzt die Umgebung zurück und erhält die neue Beobachtung und Informationen.
    if self.obs_preprocessor is not None:  
        new_obs = self.obs_preprocessor(new_obs)  # Preprocess the new observation if a preprocessor is defined.  # Verarbeitet die neue Beobachtung, wenn ein Vorverarbeiter definiert ist.
    rew = 0.0  # Initialize reward to 0.0.  # Initialisiert die Belohnung mit 0.0.
    terminated, truncated = False, False  # Set termination and truncation flags to False.  # Setzt die Flags für Terminierung und Abschneidung auf False.
    if collect_samples:  
        if self.crc_debug:  
            self.debug_ts_cpt += 1  # Increment debug timestamp counter.  # Erhöht den Debug-Zeitstempel-Zähler.
            self.debug_ts_res_cpt = 0  # Reset result timestamp counter.  # Setzt den Ergebnis-Zeitstempel-Zähler zurück.
            info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)  # Add current sample data to info.  # Fügt die aktuellen Beispieldaten den Informationen hinzu.
            info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)  # Add debug timestamps.  # Fügt die Debug-Zeitstempel hinzu.
        if self.get_local_buffer_sample:  
            sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)  # Get a sample from the local buffer.  # Holt ein Beispiel aus dem lokalen Puffer.
        else:  
            sample = act, new_obs, rew, terminated, truncated, info  # Create a sample tuple.  # Erstellt ein Beispiel-Tupel.
        self.buffer.append_sample(sample)  # Append the sample to the buffer.  # Fügt das Beispiel dem Puffer hinzu.
    return new_obs, info  # Return new observation and info.  # Gibt die neue Beobachtung und Informationen zurück.

def step(self, obs, test, collect_samples, last_step=False):  
    """
    Performs a full RL transition.  
    # Führt eine vollständige RL-Übergang durch.

    A full RL transition is `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.  
    # Eine vollständige RL-Übergang ist `obs` -> `act` -> `new_obs`, `rew`, `terminated`, `truncated`, `info`.

    Note that, in the Real-Time RL setting, `act` is appended to a buffer which is part of `new_obs`.  
    # Beachten Sie, dass in Echtzeit-RL-Einstellungen `act` an einen Puffer angehängt wird, der Teil von `new_obs` ist.
    This is because it does not directly affect the new observation, due to real-time delays.  
    # Dies liegt daran, dass es die neue Beobachtung aufgrund von Echtzeitverzögerungen nicht direkt beeinflusst.
    """  

    act = self.act(obs, test=test)  # Get the action from the actor module based on the observation.  # Holt die Aktion aus dem Actor-Modul basierend auf der Beobachtung.
    new_obs, rew, terminated, truncated, info = self.env.step(act)  # Perform the action in the environment and get the new observation, reward, termination status, truncation, and info.  # Führt die Aktion in der Umgebung aus und erhält die neue Beobachtung, Belohnung, Terminierungsstatus, Abschneidung und Informationen.
    if self.obs_preprocessor is not None:  
        new_obs = self.obs_preprocessor(new_obs)  # Preprocess the new observation if needed.  # Verarbeitet die neue Beobachtung, falls erforderlich.
    if collect_samples:  
        if last_step and not terminated:  
            truncated = True  # Mark as truncated if it's the last step and not terminated.  # Markiert als abgeschnitten, wenn es der letzte Schritt ist und nicht beendet wurde.
        if self.crc_debug:  
            self.debug_ts_cpt += 1  # Increment debug timestamp counter.  # Erhöht den Debug-Zeitstempel-Zähler.
            self.debug_ts_res_cpt += 1  # Increment result timestamp counter.  # Erhöht den Ergebnis-Zeitstempel-Zähler.
            info['crc_sample'] = (obs, act, new_obs, rew, terminated, truncated)  # Add current sample data to info.  # Fügt die aktuellen Beispieldaten den Informationen hinzu.
            info['crc_sample_ts'] = (self.debug_ts_cpt, self.debug_ts_res_cpt)  # Add debug timestamps.  # Fügt die Debug-Zeitstempel hinzu.
        if self.get_local_buffer_sample:  
            sample = self.get_local_buffer_sample(act, new_obs, rew, terminated, truncated, info)  # Get sample from local buffer.  # Holt das Beispiel aus dem lokalen Puffer.
        else:  
            sample = act, new_obs, rew, terminated, truncated, info  # Create a sample tuple.  # Erstellt ein Beispiel-Tupel.
        self.buffer.append_sample(sample)  # Append the sample to the buffer.  # Fügt das Beispiel dem Puffer hinzu.
    return new_obs, rew, terminated, truncated, info  # Return new observation, reward, termination status, truncation status, and info.  # Gibt die neue Beobachtung, Belohnung, Terminierungsstatus, Abschneidung und Informationen zurück.





def collect_train_episode(self, max_samples=None):  # Defines a function to collect a training episode. / Definiert eine Funktion zum Sammeln einer Trainings-Episode.
    """  # Docstring that describes the purpose of the function. / Docstring, die den Zweck der Funktion beschreibt.
    Collects a maximum of `max_samples` training transitions (from reset to terminated or truncated)  # Collects up to `max_samples` training transitions (reset to termination or truncation). / Sammelt bis zu `max_samples` Trainings-Übergänge (von Reset bis zu Beendigung oder Abbruch).
    
    This method stores the episode and the training return in the local `Buffer` of the worker  # This method saves the episode and training return in the worker's local buffer for later use. / Diese Methode speichert die Episode und den Trainings-Rückgabewert im lokalen Buffer des Arbeiters zur späteren Verwendung.
    for sending to the `Server`.  # For sending data to the server. / Zum Senden der Daten an den Server.

    Args:  # Describes the function arguments. / Beschreibt die Funktionsargumente:
        max_samples (int): if the environment is not `terminated` after `max_samples` time steps,  # If not terminated after `max_samples`, the environment will reset. / Wenn nach `max_samples` Zeitschritten die Umgebung nicht beendet ist, wird sie zurückgesetzt.
            it is forcefully reset and `truncated` is set to True.  # The environment is forcefully reset and truncated. / Die Umgebung wird zwangsweise zurückgesetzt und als abgeschnitten markiert.
    """  
    if max_samples is None:  # Checks if `max_samples` is not provided. / Überprüft, ob `max_samples` nicht angegeben wurde.
        max_samples = self.max_samples_per_episode  # Assigns a default value if not provided. / Weist einen Standardwert zu, wenn nicht angegeben.

    iterator = range(max_samples) if max_samples != np.inf else itertools.count()  # Creates an iterator based on `max_samples`, or infinitely if max_samples is infinite. / Erstellt einen Iterator basierend auf `max_samples` oder unendlich, wenn `max_samples` unendlich ist.

    ret = 0.0  # Initializes the return value for the episode. / Initialisiert den Rückgabewert für die Episode.
    steps = 0  # Initializes the step counter. / Initialisiert den Schrittzähler.
    obs, info = self.reset(collect_samples=True)  # Resets the environment and starts a new episode. / Setzt die Umgebung zurück und startet eine neue Episode.
    for i in iterator:  # Iterates through the steps of the episode. / Iteriert durch die Schritte der Episode.
        obs, rew, terminated, truncated, info = self.step(obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1)  # Takes a step in the environment, collecting samples. / Macht einen Schritt in der Umgebung und sammelt Daten.
        ret += rew  # Adds the reward to the return value. / Fügt die Belohnung dem Rückgabewert hinzu.
        steps += 1  # Increments the step counter. / Erhöht den Schrittzähler.
        if terminated or truncated:  # Checks if the episode is terminated or truncated. / Überprüft, ob die Episode beendet oder abgeschnitten wurde.
            break  # Ends the loop if terminated or truncated. / Beendet die Schleife, wenn die Episode beendet oder abgeschnitten wurde.
    self.buffer.stat_train_return = ret  # Stores the training return in the buffer. / Speichert den Trainings-Rückgabewert im Buffer.
    self.buffer.stat_train_steps = steps  # Stores the number of steps in the buffer. / Speichert die Anzahl der Schritte im Buffer.

def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):  # Defines a function to run multiple episodes. / Definiert eine Funktion zum Ausführen mehrerer Episoden.
    """  # Docstring that describes the function. / Docstring, die die Funktion beschreibt.
    Runs `nb_episodes` episodes.  # Runs the specified number of episodes. / Führt die angegebene Anzahl an Episoden aus.

    Args:  # Describes the function arguments. / Beschreibt die Funktionsargumente:
        max_samples_per_episode (int): same as run_episode  # Specifies maximum samples per episode. / Gibt die maximalen Proben pro Episode an.
        nb_episodes (int): total number of episodes to collect  # Total number of episodes to collect. / Gesamte Anzahl der Episoden zum Sammeln.
        train (bool): same as run_episode  # Whether to train during the episodes. / Ob während der Episoden trainiert werden soll.
    """
    if max_samples_per_episode is None:  # Checks if `max_samples_per_episode` is not provided. / Überprüft, ob `max_samples_per_episode` nicht angegeben wurde.
        max_samples_per_episode = self.max_samples_per_episode  # Assigns default value if not provided. / Weist einen Standardwert zu, wenn nicht angegeben.

    iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()  # Creates an iterator for running episodes. / Erstellt einen Iterator für die Ausführung der Episoden.

    for _ in iterator:  # Loops through the number of episodes. / Schleift durch die Anzahl der Episoden.
        self.run_episode(max_samples_per_episode, train=train)  # Runs each individual episode. / Führt jede einzelne Episode aus.

def run_episode(self, max_samples=None, train=False):  # Defines a function to run a single episode. / Definiert eine Funktion, um eine einzelne Episode auszuführen.
    """  # Docstring that describes the function. / Docstring, die die Funktion beschreibt.
    Collects a maximum of `max_samples` test transitions (from reset to terminated or truncated).  # Collects up to `max_samples` test transitions. / Sammelt bis zu `max_samples` Test-Übergänge.

    Args:  # Describes the function arguments. / Beschreibt die Funktionsargumente:
        max_samples (int): At most `max_samples` samples are collected per episode.  # Specifies maximum samples to collect per episode. / Gibt die maximalen Proben an, die pro Episode gesammelt werden.
            If the episode is longer, it is forcefully reset and `truncated` is set to True.  # If the episode exceeds, it's reset and truncated. / Wenn die Episode länger ist, wird sie zurückgesetzt und abgeschnitten.
        train (bool): whether the episode is a training or a test episode.  # Indicates if the episode is for training or testing. / Gibt an, ob die Episode für Training oder Testen ist.
            `step` is called with `test=not train`.  # Calls `step` with `test=not train` depending on the mode. / Ruft `step` mit `test=not train` je nach Modus auf.
    """
    if max_samples is None:  # Checks if `max_samples` is not provided. / Überprüft, ob `max_samples` nicht angegeben wurde.
        max_samples = self.max_samples_per_episode  # Assigns default value if not provided. / Weist einen Standardwert zu, wenn nicht angegeben.

    iterator = range(max_samples) if max_samples != np.inf else itertools.count()  # Creates an iterator for episode steps. / Erstellt einen Iterator für die Schritte der Episode.

    ret = 0.0  # Initializes the return value for the episode. / Initialisiert den Rückgabewert für die Episode.
    steps = 0  # Initializes the step counter. / Initialisiert den Schrittzähler.
    obs, info = self.reset(collect_samples=False)  # Resets the environment and starts a new episode without collecting samples. / Setzt die Umgebung zurück und startet eine neue Episode ohne Proben zu sammeln.
    for _ in iterator:  # Loops through the steps of the episode. / Schleift durch die Schritte der Episode.
        obs, rew, terminated, truncated, info = self.step(obs=obs, test=not train, collect_samples=False)  # Takes a step in the environment, not collecting samples. / Macht einen Schritt in der Umgebung, ohne Proben zu sammeln.
        ret += rew  # Adds the reward to the return value. / Fügt die Belohnung dem Rückgabewert hinzu.
        steps += 1  # Increments the step counter. / Erhöht den Schrittzähler.
        if terminated or truncated:  # Checks if the episode is finished. / Überprüft, ob die Episode abgeschlossen ist.
            break  # Ends the loop if the episode is finished. / Beendet die Schleife, wenn die Episode beendet ist.
    self.buffer.stat_test_return = ret  # Stores the test return in the buffer. / Speichert den Test-Rückgabewert im Buffer.
    self.buffer.stat_test_steps = steps  # Stores the number of steps in the buffer. / Speichert die Anzahl der Schritte im Buffer.

















def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):  
    """  
    Runs the worker for `nb_episodes` episodes.  # Runs the worker for a specified number of episodes.  # Führt die Arbeitseinheit für eine bestimmte Anzahl von Episoden aus.
    Args:  
        test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;  # Interval for collecting test episodes.  # Intervall für das Sammeln von Testepisoden.
        nb_episodes (int): maximum number of train episodes to collect.  # Maximum number of training episodes.  # Maximale Anzahl an Trainingsepisoden.
        verbose (bool): whether to log INFO messages.  # Whether to print log messages.  # Ob Protokollnachrichten gedruckt werden sollen.
        expert (bool): experts send training samples without updating their model nor running test episodes.  # If true, the worker operates as an expert.  # Wenn wahr, arbeitet der Arbeiter als Experte.
    """  

    iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()  
    # Create an iterator for episodes.  # Erstellt einen Iterator für Episoden.

    if expert:  
        if not verbose:  
            for _ in iterator:  
                self.collect_train_episode(self.max_samples_per_episode)  # Collect training data for one episode.  # Sammelt Trainingsdaten für eine Episode.
                self.send_and_clear_buffer()  # Send and clear the buffer.  # Senden und Löschen des Puffers.
                self.ignore_actor_weights()  # Ignore updates to actor weights.  # Ignoriert Updates an den Gewichten des Akteurs.
        else:  
            for _ in iterator:  
                print_with_timestamp("collecting expert episode")  # Log expert episode collection.  # Protokolliert die Sammlung von Expertenepisoden.
                self.collect_train_episode(self.max_samples_per_episode)  
                print_with_timestamp("copying buffer for sending")  # Log buffer preparation.  # Protokolliert die Pufferbereitung.
                self.send_and_clear_buffer()  
                self.ignore_actor_weights()  
    elif not verbose:  
        if not test_episode_interval:  
            for _ in iterator:  
                self.collect_train_episode(self.max_samples_per_episode)  
                self.send_and_clear_buffer()  
                self.update_actor_weights(verbose=False)  # Update actor weights without logging.  # Aktualisiert Akteursgewichte ohne Protokollierung.
        else:  
            for episode in iterator:  
                if episode % test_episode_interval == 0 and not self.crc_debug:  
                    self.run_episode(self.max_samples_per_episode, train=False)  # Run a test episode.  # Führt eine Testepisode aus.
                self.collect_train_episode(self.max_samples_per_episode)  
                self.send_and_clear_buffer()  
                self.update_actor_weights(verbose=False)  
    else:  
        for episode in iterator:  
            if test_episode_interval and episode % test_episode_interval == 0 and not self.crc_debug:  
                print_with_timestamp("running test episode")  # Log running a test episode.  # Protokolliert das Ausführen einer Testepisode.
                self.run_episode(self.max_samples_per_episode, train=False)  
            print_with_timestamp("collecting train episode")  # Log train episode collection.  # Protokolliert das Sammeln von Trainingsepisoden.
            self.collect_train_episode(self.max_samples_per_episode)  
            print_with_timestamp("copying buffer for sending")  
            self.send_and_clear_buffer()  
            print_with_timestamp("checking for new weights")  # Log checking for new actor weights.  # Protokolliert die Prüfung neuer Akteursgewichte.
            self.update_actor_weights(verbose=True)  

def run_synchronous(self, test_episode_interval=0, nb_steps=np.inf, initial_steps=1, max_steps_per_update=np.inf, end_episodes=True, verbose=False):  
    """  
    Collects `nb_steps` steps while synchronizing with the Trainer.  # Collects steps while synchronizing with a trainer.  # Sammelt Schritte, während es mit einem Trainer synchronisiert.
    Args:  
        test_episode_interval (int): a test episode is collected for every `test_episode_interval` train episodes;  # Interval for test episodes.  # Intervall für Testepisoden.
        nb_steps (int): total number of steps to collect (after `initial_steps`).  # Total steps to collect.  # Gesamtschritte zum Sammeln.
        initial_steps (int): initial number of steps to collect before waiting for the first model update.  # Steps before first model update.  # Schritte vor dem ersten Modell-Update.
        max_steps_per_update (float): maximum number of steps per model received from the Server.  # Max steps per model update.  # Maximale Schritte pro Modell-Update.
        end_episodes (bool): when True, waits for episodes to end before sending samples and waiting for updates.  # Wait for episode ends if true.  # Wartet auf Episodenende, wenn wahr.
        verbose (bool): whether to log INFO messages.  # Log INFO messages or not.  # Protokolliert INFO-Nachrichten oder nicht.
    """  


# collect initial samples  # Start der Sammlung von ersten Beispieldaten

if verbose:  
    logging.info(f"Collecting {initial_steps} initial steps")  # Log-Ausgabe der Anzahl initialer Schritte, falls `verbose` aktiviert ist.  # Ausgabe der initial Schritte, wenn "verbose" aktiv ist.

iteration = 0  # Setzt die Iteration auf 0.  # Initialisiert Iteration mit 0.
done = False  # Setzt den "done"-Status auf falsch.  # Definiert `done` als nicht abgeschlossen.
while iteration < initial_steps:  # Solange die Anzahl Iterationen kleiner als `initial_steps` ist.  # Schleife für die initiale Probenanzahl.
    steps = 0  # Anzahl der Schritte pro Episode auf 0 setzen.  # Schritte pro Episode werden initialisiert.
    ret = 0.0  # Gesamter Reward wird auf 0 gesetzt.  # Belohnung wird zurückgesetzt.
    # reset
    obs, info = self.reset(collect_samples=True)  # Umgebungsstatus zurücksetzen und Infos initialisieren.  # Reset der Umgebung und Sammlung der Daten starten.
    done = False  # "done"-Status erneut auf falsch setzen.  # Markiert die Episode als nicht abgeschlossen.
    iteration += 1  # Iteration um 1 erhöhen.  # Erhöht den Zähler für Iterationen.
    # episode
    while not done and (end_episodes or iteration < initial_steps):  # Solange Episode nicht abgeschlossen ist und Bedingungen erfüllt sind.  # Steuert Schleife abhängig vom Status `done`.
        # step
        obs, rew, terminated, truncated, info = self.step(  # Einen Schritt in der Umgebung ausführen.  # Führt Simulation und Rückgabewerte aus.
            obs=obs, test=False, collect_samples=True, last_step=steps == self.max_samples_per_episode - 1)  # Verschiedene Parameter für den Schritt.  # Logik für letzte Schritt definieren.
        iteration += 1  # Iteration inkrementieren.  # Iterationsschleife erhöhen.
        steps += 1  # Schritte in der Episode inkrementieren.  # Schritte der Episode erhöhen.
        ret += rew  # Belohnung zur Gesamtsumme hinzufügen.  # Addiert den Reward zu ret.
        done = terminated or truncated  # Beendet die Episode, wenn beendet oder abgeschnitten.  # Bestimmt Abschlusszustand.
    # send the collected samples to the Server
    self.buffer.stat_train_return = ret  # Belohnung in den Puffer schreiben.  # Reward zur Statistik hinzufügen.
    self.buffer.stat_train_steps = steps  # Schritte in den Puffer schreiben.  # Schritte in die Statistik einfügen.
    if verbose:  
        logging.info(f"Sending buffer (initial steps)")  # Loggt das Senden des Puffers.  # Debug-Informationen ausgeben.
    self.send_and_clear_buffer()  # Puffer senden und leeren.  # Pufferspeicher leeren.

i_model = 1  # Modell-Index auf 1 setzen.  # Start-Modellnummer setzen.

# wait for the first updated model if required here
ratio = (iteration + 1) / i_model  # Berechnung des Verhältnisses zwischen Iteration und Modell.  # Bestimmt Verhältnis Iteration/Modell.
while ratio > max_steps_per_update:  # Solange das Verhältnis größer als der maximale Schwellenwert ist.  # Überprüft Grenzwert.
    if verbose:  
        logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates")  # Debug-Ausgabe, falls Verhältnis überschritten.  # Statusmeldung bei Überschreitung.
    self.send_and_clear_buffer()  # Senden und Leeren des Puffers.  # Sammelt neuen Datensatz.
    i_model += self.update_actor_weights(verbose=verbose, blocking=True)  # Modell-Parameter aktualisieren.  # Modellparameter synchronisieren.
    ratio = (iteration + 1) / i_model  # Neues Verhältnis berechnen.  # Neues Verhältnis prüfen.

# collect further samples while synchronizing with the Trainer
iteration = 0  # Iteration auf 0 zurücksetzen.  # Iterationen zurücksetzen.
episode = 0  # Episodenzähler auf 0 setzen.  # Episoden starten.
steps = 0  # Schrittzähler auf 0 setzen.  # Schritte zurücksetzen.
ret = 0.0  # Gesamten Reward auf 0 setzen.  # Belohnung auf Anfangszustand setzen.

while iteration < nb_steps:  # Solange die Iteration kleiner ist als `nb_steps`.  # Schleife für alle Schritte.
    if done:  
        # test episode
        if test_episode_interval > 0 and episode % test_episode_interval == 0 and end_episodes:  # Überprüft Testepisoden-Bedingung.  # Führt Test durch, wenn Intervall erreicht.
            if verbose:  
                print_with_timestamp("running test episode")  # Debug-Ausgabe für Test.  # Meldung über Testlauf.
            self.run_episode(self.max_samples_per_episode, train=False)  # Testepisode ausführen.  # Testschritte simulieren.
        # reset
        obs, info = self.reset(collect_samples=True)  # Umgebung zurücksetzen.  # Neue Episode initialisieren.
        done = False  # Status zurücksetzen.  # Markiert Episode als nicht abgeschlossen.
        iteration += 1  # Iteration erhöhen.  # Iteration aktualisieren.
        steps = 0  # Schrittzähler auf 0 setzen.  # Zähler wieder auf 0 setzen.
        ret = 0.0  # Reward zurücksetzen.  # Belohnung zurücksetzen.
        episode += 1  # Episodenzähler erhöhen.  # Zähler für Episoden inkrementieren.

    while not done and (end_episodes or ratio <= max_steps_per_update):  # Schleife für Episode.  # Schleife abhängig von Abschlusszustand.
        # step
        obs, rew, terminated, truncated, info = self.step(  # Ausführen eines Schrittes.  # Einzelner Schritt in der Simulation.
            obs=obs, test=False, collect_samples=True, last_step=steps == self.max_samples_per_episode - 1)  # Schrittparameter.  # Parameter für Schrittlogik.
        iteration += 1  # Iteration erhöhen.  # Inkrementiert Iteration.
        steps += 1  # Schritte erhöhen.  # Schrittzähler erhöhen.
        ret += rew  # Reward akkumulieren.  # Belohnungssumme aktualisieren.
        done = terminated or truncated  # Abschlusszustand bestimmen.  # Überprüft Abschluss.

        if not end_episodes:  
            # check model and send samples after each step
            ratio = (iteration + 1) / i_model  # Berechnung des Verhältnisses nach jedem Schritt.  # Aktuelles Verhältnis bestimmen.
            while ratio > max_steps_per_update:  # Überprüfen, ob Schwellenwert überschritten.  # Maximalwert kontrollieren.
                if verbose:  
                    logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (no eoe)")  # Debug-Log.  # Debug-Status ausgeben.
                if not done:  
                    if verbose:  
                        logging.info(f"Sending buffer (no eoe)")  # Buffer senden.  # Debug: Puffer senden.
                    self.send_and_clear_buffer()  # Senden und Leeren.  # Buffer senden.
                i_model += self.update_actor_weights(verbose=verbose, blocking=True)  # Gewichtung aktualisieren.  # Parameter synchronisieren.
                ratio = (iteration + 1) / i_model  # Neues Verhältnis berechnen.  # Verhältnis aktualisieren.

    if end_episodes:  
        # check model and send samples only after episodes end
        ratio = (iteration + 1) / i_model  # Berechnen des Verhältnisses nach Episode.  # Verhältnis prüfen.
        while ratio > max_steps_per_update:  # Schleife für Schwellenwertprüfung.  # Kontrolliert maximal zulässiges Verhältnis.
            if verbose:  
                logging.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates (eoe)")  # Debug-Status ausgeben.  # Statusmeldung zu Überschreitungen.
            if not done:  
                if verbose:  
                    logging.info(f"Sending buffer (eoe)")  # Debug-Log für Episodenende.  # Sendepuffer bei Episodenende.
                self.send_and_clear_buffer()  # Senden und Puffer leeren.  # Pufferspeicher abschicken.
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)  # Gewichtungen aktualisieren.  # Synchronisation durchführen.
            ratio = (iteration + 1) / i_model  # Verhältnis neu berechnen.  # Verhältnis aktualisieren.

    self.buffer.stat_train_return = ret  # Belohnung in Puffer schreiben.  # Statistik für Training aktualisieren.
    self.buffer.stat_train_steps = steps  # Schritte in Puffer schreiben.  # Anzahl Schritte aktualisieren.
    if verbose:  
        logging.info(f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}")  # Debug-Ausgabe.  # Zusätzliche Debug-Daten.
    self.send_and_clear_buffer()  # Puffer senden und leeren.  # Bufferspeicher finalisieren.


     
def run_env_benchmark(self, nb_steps, test=False, verbose=True):  
    # Benchmarks the environment.  # Führt einen Benchmark der Umgebung durch.
    """
    Benchmarks the environment.

    This method is only compatible with rtgym_ environments.
    Furthermore, the `"benchmark"` option of the rtgym configuration dictionary must be set to `True`.

    .. _rtgym: https://github.com/yannbouteiller/rtgym

    Args:
        nb_steps (int): number of steps to perform to compute the benchmark
        test (int): whether the actor is called in test or train mode
        verbose (bool): whether to log INFO messages
    """
    if nb_steps == np.inf or nb_steps < 0:  
        # Checks if the number of steps is infinite or negative.  # Überprüft, ob die Anzahl der Schritte unendlich oder negativ ist.
        raise RuntimeError(f"Invalid number of steps: {nb_steps}")  
        # Raises an error if the number of steps is invalid.  # Gibt einen Fehler aus, wenn die Schrittanzahl ungültig ist.

    obs, info = self.reset(collect_samples=False)  
    # Resets the environment and initializes observation and info.  # Setzt die Umgebung zurück und initialisiert Observation und Info.
    for _ in range(nb_steps):  
        # Loops for the specified number of steps.  # Schleife für die angegebene Anzahl von Schritten.
        obs, rew, terminated, truncated, info = self.step(obs=obs, test=test, collect_samples=False)  
        # Executes a step in the environment.  # Führt einen Schritt in der Umgebung aus.
        if terminated or truncated:  
            # Breaks the loop if the episode ends.  # Bricht die Schleife ab, wenn die Episode endet.
            break  
    res = self.env.benchmarks()  
    # Retrieves benchmark results from the environment.  # Ruft Benchmark-Ergebnisse aus der Umgebung ab.
    if verbose:  
        # Logs benchmark results if verbose mode is enabled.  # Protokolliert Benchmark-Ergebnisse, wenn der Verbose-Modus aktiviert ist.
        print_with_timestamp(f"Benchmark results:\n{res}")  
        # Prints the benchmark results.  # Gibt die Benchmark-Ergebnisse aus.
    return res  
    # Returns the benchmark results.  # Gibt die Benchmark-Ergebnisse zurück.

def send_and_clear_buffer(self):  
    # Sends the buffered samples to the Server.  # Sendet die gepufferten Proben an den Server.
    self.__endpoint.produce(self.buffer, "trainers")  
    # Sends the buffer contents to the server for trainers.  # Sendet den Inhalt des Puffers an den Server für Trainer.
    self.buffer.clear()  
    # Clears the buffer.  # Löscht den Puffer.

def update_actor_weights(self, verbose=True, blocking=False):  
    # Updates the actor with new weights from the Server.  # Aktualisiert den Actor mit neuen Gewichten vom Server.
    """
    Updates the actor with new weights received from the `Server` when available.

    Args:
        verbose (bool): whether to log INFO messages.
        blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

    Returns:
        int: number of new actor models received from the Server (the latest is used).
    """
    weights_list = self.__endpoint.receive_all(blocking=blocking)  
    # Receives all weights from the server.  # Empfängt alle Gewichte vom Server.
    nb_received = len(weights_list)  
    # Counts the number of weights received.  # Zählt die Anzahl der empfangenen Gewichte.
    if nb_received > 0:  
        # If weights were received.  # Wenn Gewichte empfangen wurden.
        weights = weights_list[-1]  
        # Takes the latest weights.  # Nimmt die neuesten Gewichte.
        with open(self.model_path, 'wb') as f:  
            # Opens the model path file for writing.  # Öffnet die Modelldatei zum Schreiben.
            f.write(weights)  
            # Writes the weights to the file.  # Schreibt die Gewichte in die Datei.
        if self.model_history:  
            # If model history is enabled.  # Wenn die Modellhistorie aktiviert ist.
            self._cur_hist_cpt += 1  
            # Increments the history counter.  # Erhöht den Historienzähler.
            if self._cur_hist_cpt == self.model_history:  
                # If history counter reaches the limit.  # Wenn der Historienzähler das Limit erreicht.
                x = datetime.datetime.now()  
                # Gets the current datetime.  # Ruft das aktuelle Datum und die Uhrzeit ab.
                with open(self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod", 'wb') as f:  
                    # Saves the weights to a history file.  # Speichert die Gewichte in einer Historiendatei.
                    f.write(weights)  
                    # Writes the weights to the history file.  # Schreibt die Gewichte in die Historiendatei.
                self._cur_hist_cpt = 0  
                # Resets the history counter.  # Setzt den Historienzähler zurück.
                if verbose:  
                    # Logs the history save action if verbose mode is enabled.  # Protokolliert das Speichern der Historie, wenn der Verbose-Modus aktiviert ist.
                    print_with_timestamp("model weights saved in history")  
        self.actor = self.actor.load(self.model_path, device=self.device)  
        # Loads the updated weights into the actor.  # Lädt die aktualisierten Gewichte in den Actor.
        if verbose:  
            # Logs the weight update if verbose mode is enabled.  # Protokolliert das Gewicht-Update, wenn der Verbose-Modus aktiviert ist.
            print_with_timestamp("model weights have been updated")  
    return nb_received  
    # Returns the number of weights received.  # Gibt die Anzahl der empfangenen Gewichte zurück.

def ignore_actor_weights(self):  
    # Clears the buffer of received weights.  # Löscht den Puffer der empfangenen Gewichte.
    """
    Clears the buffer of weights received from the `Server`.

    This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

    Returns:
        int: number of new (ignored) actor models received from the Server.
    """
    weights_list = self.__endpoint.receive_all(blocking=False)  
    # Receives all weights non-blockingly.  # Empfängt alle Gewichte nicht blockierend.
    nb_received = len(weights_list)  
    # Counts the number of ignored weights.  # Zählt die Anzahl der ignorierten Gewichte.
    return nb_received  
    # Returns the count of ignored weights.  # Gibt die Anzahl der ignorierten Gewichte zurück.
