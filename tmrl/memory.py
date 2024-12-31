# standard library imports
import os  # Importing the os module for interacting with the operating system.  # Importiert das os-Modul für die Interaktion mit dem Betriebssystem.
import pickle  # Importing pickle for serializing and deserializing Python objects.  # Importiert pickle für das Serialisieren und Deserialisieren von Python-Objekten.
import zlib  # Importing zlib for compression and CRC32 checks.  # Importiert zlib für Komprimierung und CRC32-Prüfungen.
from abc import ABC, abstractmethod  # Importing ABC (Abstract Base Classes) and abstractmethod for creating abstract classes.  # Importiert ABC (Abstrakte Basisklassen) und abstractmethod für die Erstellung abstrakter Klassen.
from pathlib import Path  # Importing Path from pathlib to work with filesystem paths in a convenient way.  # Importiert Path aus pathlib, um auf einfache Weise mit Dateisystempfaden zu arbeiten.
from random import randint  # Importing randint for generating random integers.  # Importiert randint für die Erzeugung von Zufallszahlen.
import logging  # Importing logging for logging messages during execution.  # Importiert logging für die Protokollierung von Nachrichten während der Ausführung.

# third-party imports
import numpy as np  # Importing numpy for numerical operations.  # Importiert numpy für numerische Operationen.
# from torch.utils.data import DataLoader, Dataset, Sampler  # Importing PyTorch utilities for handling datasets and data loading (commented out).  # Importiert PyTorch-Tools für den Umgang mit Datensätzen und das Laden von Daten (auskommentiert).

# local imports
from tmrl.util import collate_torch  # Importing collate_torch from a local utility module.  # Importiert collate_torch aus einem lokalen Hilfsmodul.

__docformat__ = "google"  # Specifies the documentation format to be Google style.  # Gibt das Dokumentationsformat als Google-Stil an.

def check_samples_crc(original_po, original_a, original_o, original_r, original_d, original_t, rebuilt_po, rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t, debug_ts, debug_ts_res):
    # Check if the original and rebuilt samples match and verify CRC32 checksum.
    # Überprüft, ob die ursprünglichen und wiederaufgebauten Proben übereinstimmen und validiert die CRC32-Prüfziffer.

    assert original_po is None or str(original_po) == str(rebuilt_po), f"previous observations don't match:\noriginal:\n{original_po}\n!= rebuilt:\n{rebuilt_po}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure previous observations match.  # Stellt sicher, dass die vorherigen Beobachtungen übereinstimmen.
    assert str(original_a) == str(rebuilt_a), f"actions don't match:\noriginal:\n{original_a}\n!= rebuilt:\n{rebuilt_a}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure actions match.  # Stellt sicher, dass die Aktionen übereinstimmen.
    assert str(original_o) == str(rebuilt_o), f"observations don't match:\noriginal:\n{original_o}\n!= rebuilt:\n{rebuilt_o}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure observations match.  # Stellt sicher, dass die Beobachtungen übereinstimmen.
    assert str(original_r) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{original_r}\n!= rebuilt:\n{rebuilt_r}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure rewards match.  # Stellt sicher, dass die Belohnungen übereinstimmen.
    assert str(original_d) == str(rebuilt_d), f"terminated don't match:\noriginal:\n{original_d}\n!= rebuilt:\n{rebuilt_d}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure termination flags match.  # Stellt sicher, dass die Beendigungsflags übereinstimmen.
    assert str(original_t) == str(rebuilt_t), f"truncated don't match:\noriginal:\n{original_t}\n!= rebuilt:\n{rebuilt_t}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Ensure truncation flags match.  # Stellt sicher, dass die Abschneide-Flags übereinstimmen.
    
    original_crc = zlib.crc32(str.encode(str((original_a, original_o, original_r, original_d, original_t))))  # Calculate the CRC32 for the original sample.  # Berechnet die CRC32 für die ursprüngliche Probe.
    crc = zlib.crc32(str.encode(str((rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t))))  # Calculate the CRC32 for the rebuilt sample.  # Berechnet die CRC32 für die wiederaufgebaute Probe.
    
    assert crc == original_crc, f"CRC failed: new crc:{crc} != old crc:{original_crc}.\nEither the custom pipeline is corrupted, or crc_debug is False in the rollout worker.\noriginal sample:\n{(original_a, original_o, original_r, original_d)}\n!= rebuilt sample:\n{(rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d)}\nTime step: {debug_ts}, since reset: {debug_ts_res}"  # Assert that the CRC32 values match.  # Stellt sicher, dass die CRC32-Werte übereinstimmen.
    
    print(f"DEBUG: CRC check passed. Time step: {debug_ts}, since reset: {debug_ts_res}")  # Print debug message when CRC check passes.  # Gibt eine Debug-Nachricht aus, wenn die CRC-Prüfung bestanden wurde.

class Memory(ABC):  # Define the Memory class as an abstract class (interface for replay buffer).  # Definiert die Memory-Klasse als abstrakte Klasse (Schnittstelle für den Replay-Puffer).
    """
    Interface implementing the replay buffer.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """
    # Explanation: The Memory class is meant to manage a buffer of samples for reinforcement learning. It implements an abstract base class.
    # Erklärung: Die Memory-Klasse soll einen Puffer von Proben für verstärkendes Lernen verwalten. Sie implementiert eine abstrakte Basisklasse.

    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False):  # Initialize the memory with parameters.  # Initialisiert den Speicher mit Parametern.
        """
        Args:
            device (str): output tensors will be collated to this device  # Gerät, auf dem die Ausgabetensoren gesammelt werden.
            nb_steps (int): number of steps per round  # Anzahl der Schritte pro Runde.
            sample_preprocessor (callable): can be used for data augmentation  # Kann für Datenaugmentation verwendet werden.
            memory_size (int): size of the circular buffer  # Größe des zirkulären Puffers.
            batch_size (int): batch size of the output tensors  # Batch-Größe der Ausgabetensoren.
            dataset_path (str): an offline dataset may be provided here to initialize the memory  # Ein Offline-Datensatz kann hier bereitgestellt werden, um den Speicher zu initialisieren.
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline  # Normalerweise False, True bei Verwendung der CRC-Debugging-Funktion der Pipeline.
        """
        self.nb_steps = nb_steps  # Set the number of steps per round.  # Setzt die Anzahl der Schritte pro Runde.
        self.device = device  # Set the device for tensor output.  # Setzt das Gerät für die Tensor-Ausgabe.
        self.batch_size = batch_size  # Set the batch size.  # Setzt die Batch-Größe.
        self.memory_size = memory_size  # Set the size of the memory buffer.  # Setzt die Größe des Speicherpuffers.
        self.sample_preprocessor = sample_preprocessor  # Set the optional sample preprocessor.  # Setzt den optionalen Proben-Vorprozessor.
        self.crc_debug = crc_debug  # Set CRC debugging flag.  # Setzt das CRC-Debugging-Flag.

        # These stats are here because they reach the trainer along with the buffer:  # Diese Statistiken sind hier, weil sie zusammen mit dem Puffer den Trainer erreichen.
        self.stat_test_return = 0.0  # Initialize test return stat.  # Initialisiert die Test-Rückgabestatistik.
        self.stat_train_return = 0.0  # Initialize training return stat.  # Initialisiert die Trainings-Rückgabestatistik.
        self.stat_test_steps = 0  # Initialize test steps stat.  # Initialisiert die Test-Schritte-Statistik.
        self.stat_train_steps = 0  # Initialize training steps stat.  # Initialisiert die Trainings-Schritte-Statistik.

        # init memory  # Initialisiert den Speicher.
        self.path = Path(dataset_path)  # Convert dataset path to Path object.  # Konvertiert den Datensatzpfad in ein Path-Objekt.
        logging.debug(f"Memory self.path:{self.path}")  # Log the dataset path.  # Protokolliert den Datensatzpfad.
        if os.path.isfile(self.path / 'data.pkl'):  # Check if 'data.pkl' exists.  # Überprüft, ob 'data.pkl' existiert.
            with open(self.path / 'data.pkl', 'rb') as f:  # Open 'data.pkl' file for reading.  # Öffnet die Datei 'data.pkl' zum Lesen.
                self.data = list(pickle.load(f))  # Deserialize and load data into memory.  # Deserialisiert und lädt Daten in den Speicher.
        else:  # If file doesn't exist.  # Wenn die Datei nicht existiert.
            logging.info("no data found, initializing empty replay memory")  # Log that no data was found.  # Protokolliert, dass keine Daten gefunden wurden.
            self.data = []  # Initialize empty memory.  # Initialisiert einen leeren Speicher.

        if len(self) > self.memory_size:  # Check if memory size exceeds the limit.  # Überprüft, ob die Speichergröße das Limit überschreitet.
            # TODO: crop to memory_size  # To be implemented: crop data if size exceeds memory.  # TODO: Daten kürzen, wenn die Größe den Speicher überschreitet.
            logging.warning(f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")  # Log warning.  # Gibt eine Warnung aus.

    def __iter__(self):  # Define an iterator for the memory class.  # Definiert einen Iterator für die Memory-Klasse.
        for _ in range(self.nb_steps):  # Loop through nb_steps.  # Schleife durch nb_steps.
            yield self.sample()  # Yield a sample from memory.  # Gibt eine Probe aus dem Speicher zurück.


@abstractmethod  # Marks the following method as abstract, meaning it must be implemented by subclasses.  # bezeichnet die folgende Methode als abstrakt, was bedeutet, dass sie von Unterklassen implementiert werden muss.
def append_buffer(self, buffer):  # Method to append a buffer to memory.  # Methode zum Hinzufügen eines Puffers zum Speicher.
    """
    Must append a Buffer object to the memory.  # Muss ein Buffer-Objekt zum Speicher hinzufügen.
    
    Args:  # Argumente:
        buffer (tmrl.networking.Buffer): the buffer of samples to append.  # buffer (tmrl.networking.Buffer): der Puffer von Proben, der hinzugefügt werden soll.
    """
    raise NotImplementedError  # Raises an error if not implemented.  # Löst einen Fehler aus, wenn es nicht implementiert ist.

@abstractmethod  # Another abstract method.  # Eine weitere abstrakte Methode.
def __len__(self):  # Method to return the length of memory.  # Methode zum Zurückgeben der Länge des Speichers.
    """
    Must return the length of the memory.  # Muss die Länge des Speichers zurückgeben.
    
    Returns:  # Gibt zurück:
        int: the maximum `item` argument of `get_transition`  # int: das maximale `item`-Argument von `get_transition`.
    """
    raise NotImplementedError  # If not implemented, raises an error.  # Löst einen Fehler aus, wenn es nicht implementiert ist.

@abstractmethod  # Marks the method as abstract.  # Kennzeichnet die Methode als abstrakt.
def get_transition(self, item):  # Method to get a transition sample from memory.  # Methode, um eine Übergangsprobe aus dem Speicher zu erhalten.
    """
    Must return a transition.  # Muss eine Übergangsprobe zurückgeben.
    
    `info` is required in each sample for CRC debugging. The 'crc' key is important for this.  # `info` wird in jeder Probe für das CRC-Debugging benötigt. Der 'crc'-Schlüssel ist dafür wichtig.
    
    Args:  # Argumente:
        item (int): the index where to sample  # item (int): der Index, von dem die Probe entnommen wird.
    
    Returns:  # Gibt zurück:
        Tuple: (prev_obs, prev_act, rew, obs, terminated, truncated, info)  # Tupel: (prev_obs, prev_act, rew, obs, terminated, truncated, info)
    """
    raise NotImplementedError  # If not implemented, raises an error.  # Löst einen Fehler aus, wenn es nicht implementiert ist.

@abstractmethod  # Abstract method marker.  # Kennzeichen für eine abstrakte Methode.
def collate(self, batch, device):  # Method to collate training samples into tensors.  # Methode zum Zusammenführen von Trainingsproben in Tensoren.
    """
    Must collate `batch` onto `device`.  # Muss `batch` auf `device` zusammenfassen.
    
    `batch` is a list of training samples.  # `batch` ist eine Liste von Trainingsproben.
    The length of `batch` is `batch_size`.  # Die Länge von `batch` entspricht der `batch_size`.
    Each training sample in the list is of the form `(prev_obs, new_act, rew, new_obs, terminated, truncated)`.  # Jede Trainingsprobe in der Liste hat die Form `(prev_obs, new_act, rew, new_obs, terminated, truncated)`.
    These samples must be collated into 6 tensors of batch dimension `batch_size`.  # Diese Proben müssen in 6 Tensoren der Batch-Dimension `batch_size` zusammengefasst werden.
    These tensors should be collated onto the device indicated by the `device` argument.  # Diese Tensoren sollten auf das durch das Argument `device` angegebene Gerät übertragen werden.
    Then, your implementation must return a single tuple containing these 6 tensors.  # Dann muss Ihre Implementierung ein einziges Tupel zurückgeben, das diese 6 Tensoren enthält.
    
    Args:  # Argumente:
        batch (list): list of `(prev_obs, new_act, rew, new_obs, terminated, truncated)` tuples  # batch (Liste): Liste von Tupeln `(prev_obs, new_act, rew, new_obs, terminated, truncated)`
        device: device onto which the list needs to be collated into batches `batch_size`  # device: Gerät, auf das die Liste in Batches mit `batch_size` zusammengefasst werden muss.
    
    Returns:  # Gibt zurück:
        Tuple of tensors:  # Tupel von Tensoren:
        (prev_obs_tens, new_act_tens, rew_tens, new_obs_tens, terminated_tens, truncated_tens)  # (prev_obs_tens, new_act_tens, rew_tens, new_obs_tens, terminated_tens, truncated_tens)
        collated on device `device`, each of batch dimension `batch_size`  # Zusammengeführt auf dem Gerät `device`, jede mit der Batch-Dimension `batch_size`
    """
    raise NotImplementedError  # If not implemented, raises an error.  # Löst einen Fehler aus, wenn es nicht implementiert ist.

def sample(self):  # Method to sample a batch of transitions.  # Methode zum Abtasten einer Batch von Übergängen.
    indices = self.sample_indices()  # Get indices of sampled transitions.  # Hole Indizes der abgetasteten Übergänge.
    batch = [self[idx] for idx in indices]  # Create the batch from the sampled indices.  # Erstelle die Batch aus den abgetasteten Indizes.
    batch = self.collate(batch, self.device)  # Collate the batch into tensors on the device.  # Fasse die Batch auf dem Gerät zu Tensoren zusammen.
    return batch  # Return the collated batch.  # Gib die zusammengefasste Batch zurück.

def append(self, buffer):  # Method to append a buffer to memory.  # Methode zum Hinzufügen eines Puffers zum Speicher.
    if len(buffer) > 0:  # Check if buffer is not empty.  # Überprüfe, ob der Puffer nicht leer ist.
        self.stat_train_return = buffer.stat_train_return  # Copy training statistics from buffer.  # Kopiere Trainingsstatistiken aus dem Puffer.
        self.stat_test_return = buffer.stat_test_return  # Copy test statistics from buffer.  # Kopiere Teststatistiken aus dem Puffer.
        self.stat_train_steps = buffer.stat_train_steps  # Copy training steps count from buffer.  # Kopiere die Anzahl der Trainingsschritte aus dem Puffer.
        self.stat_test_steps = buffer.stat_test_steps  # Copy test steps count from buffer.  # Kopiere die Anzahl der Testschritte aus dem Puffer.
        self.append_buffer(buffer)  # Append the buffer to memory.  # Füge den Puffer dem Speicher hinzu.

def __getitem__(self, item):  # Method to get a specific transition by index.  # Methode zum Abrufen einer bestimmten Übergangsprobe nach Index.
    prev_obs, new_act, rew, new_obs, terminated, truncated, info = self.get_transition(item)  # Get the transition data.  # Hole die Übergangsdatensätze.
    if self.crc_debug:  # If CRC debugging is enabled.  # Wenn CRC-Debugging aktiviert ist.
        po, a, o, r, d, t = info['crc_sample']  # Extract CRC sample data.  # Extrahiere CRC-Proben-Daten.
        debug_ts, debug_ts_res = info['crc_sample_ts']  # Extract CRC timestamp data.  # Extrahiere CRC-Zeitstempel-Daten.
        check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts, debug_ts_res)  # Debug CRC samples.  # Debugge CRC-Proben.
    if self.sample_preprocessor is not None:  # If a sample preprocessor is defined.  # Wenn ein Proben-Präprozessor definiert ist.
        prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, terminated, truncated)  # Preprocess the samples.  # Verarbeite die Proben vor.
    terminated = np.float32(terminated)  # Convert to float32 for consistency.  # Konvertiere in float32 für Konsistenz.
    truncated = np.float32(truncated)  # Convert to float32 for consistency.  # Konvertiere in float32 für Konsistenz.
    return prev_obs, new_act, rew, new_obs, terminated, truncated  # Return the processed sample.  # Gib die verarbeitete Probe zurück.

def sample_indices(self):  # Method to generate random indices for sampling.  # Methode zum Erzeugen zufälliger Indizes zum Abtasten.
    return (randint(0, len(self) - 1) for _ in range(self.batch_size))  # Generate random indices for the batch size.  # Erzeuge zufällige Indizes für die Batch-Größe.

class TorchMemory(Memory, ABC):  # TorchMemory class inherits from Memory and ABC.  # TorchMemory-Klasse erbt von Memory und ABC.
    """
    Partial implementation of the `Memory` class collating samples into batched torch tensors.  # Teilweise Implementierung der `Memory`-Klasse, die Proben in gebatchte Torch-Tensoren zusammenfasst.
    
    .. note::  # Hinweis:
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.  # Wenn `__init__` überschrieben wird, vergessen Sie nicht, `super().__init__` in der Unterklasse aufzurufen.
       Your `__init__` method needs to take at least all the arguments of the superclass.  # Ihre `__init__`-Methode muss mindestens alle Argumente der Oberklasse annehmen.
    """
    def __init__(self,  # Constructor for the TorchMemory class.  # Konstruktor für die TorchMemory-Klasse.
                 device,  # Device to store tensors.  # Gerät, auf dem die Tensoren gespeichert werden.
                 nb_steps,  # Number of steps per round.  # Anzahl der Schritte pro Runde.
                 sample_preprocessor: callable = None,  # Optional callable for data preprocessing.  # Optionaler Aufruf zur Datenvorverarbeitung.
                 memory_size=1000000,  # Memory buffer size.  # Puffergröße des Speichers.
                 batch_size=256,  # Size of each batch.  # Größe jedes Batches.
                 dataset_path="",  # Path to an offline dataset.  # Pfad zu einem Offline-Datensatz.
                 crc_debug=False):  # Enable CRC debugging.  # Aktiviert CRC-Debugging.
        """
        Args:  # Argumente:
            device (str): output tensors will be collated to this device  # device (str): Ausgabetensoren werden auf diesem Gerät zusammengefasst.
            nb_steps (int): number of steps per round  # nb_steps (int): Anzahl der Schritte pro Runde.
            sample_preprocessor (callable): can be used for data augmentation  # sample_preprocessor (callable): kann für Datenaugmentation verwendet werden.
            memory_size (int): size of the circular buffer  # memory_size (int): Größe des zirkulären Puffers.
            batch_size (int): batch size of the output tensors  # batch_size (int): Batch-Größe der Ausgabetensoren.
            dataset_path (str): an offline dataset may be provided here to initialize the memory  # dataset_path (str): Ein Offline-Datensatz kann hier angegeben werden, um den Speicher zu initialisieren.
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline  # crc_debug (bool): Normalerweise False, True, wenn CRC-Debugging der Pipeline verwendet wird.
        """
        super().__init__(memory_size=memory_size,  # Calls the constructor of the parent class Memory.  # Ruft den Konstruktor der Elternklasse Memory auf.
                         batch_size=batch_size,  # Batch size for the memory.  # Batch-Größe für den Speicher.
                         dataset_path=dataset_path,  # Path to dataset.  # Pfad zum Datensatz.
                         nb_steps=nb_steps,  # Steps per round.  # Schritte pro Runde.
                         sample_preprocessor=sample_preprocessor,  # Optional preprocessor for samples.  # Optionaler Präprozessor für Proben.
                         crc_debug=crc_debug,  # CRC Debugging flag.  # CRC-Debugging-Flag.
                         device=device)  # Device for tensors.  # Gerät für Tensoren.

    def collate(self, batch, device):  # Collates samples into torch tensors.  # Fasst Proben in Torch-Tensoren zusammen.
        return collate_torch(batch, device)  # Uses the collate_torch function to collate the batch.  # Verwendet die Funktion collate_torch, um die Batch zusammenzufassen.
