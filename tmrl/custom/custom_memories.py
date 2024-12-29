import random  # Import the random module, used for generating random numbers.  # Importiere das Modul random, das für die Erzeugung von Zufallszahlen verwendet wird.
import numpy as np  # Import the numpy module, which is used for numerical operations, especially with arrays.  # Importiere das numpy-Modul, das für numerische Operationen, insbesondere mit Arrays, verwendet wird.

from tmrl.memory import TorchMemory  # Import the TorchMemory class from the tmrl.memory module.  # Importiere die Klasse TorchMemory aus dem Modul tmrl.memory.


# LOCAL BUFFER COMPRESSION ==============================  # Lokale Pufferkompression ==============================

def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):  # Define a function to get a local buffer sample, focusing on LIDAR data.  # Definiere eine Funktion, um eine lokale Pufferprobe zu erhalten, die sich auf LIDAR-Daten konzentriert.
    """
    Input:  # Eingabe:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)  # prev_act: Aktion, die aus einer vorherigen Beobachtung berechnet wurde und angewendet wurde, um die Beobachtung im Übergang zu erzeugen (aber die unbeeinflusste Beobachtung in Echtzeit-Umgebungen nicht beeinflusst).
        obs, rew, terminated, truncated, info: outcome of the transition  # obs, rew, terminated, truncated, info: Ergebnis des Übergangs
    this function creates the object that will actually be stored in local buffers for networking  # Diese Funktion erstellt das Objekt, das tatsächlich in lokalen Puffern für das Netzwerk gespeichert wird.
    this is to compress the sample before sending it over the Internet/local network  # Dies dient dazu, die Probe vor dem Senden über das Internet/Netzwerk zu komprimieren.
    buffers of such samples will be given as input to the append() method of the memory  # Buffer solcher Proben werden als Eingabe an die append()-Methode des Speichers übergeben.
    the user must define both this function and the append() method of the memory  # Der Benutzer muss sowohl diese Funktion als auch die append()-Methode des Speichers definieren.
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))  # ACHTUNG: prev_act ist die Aktion, die VOR der Beobachtung kommt (d.h. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-19:])  # speed and most recent LIDAR only  # Geschwindigkeit und nur das jüngste LIDAR
    rew_mod = np.float32(rew)  # Convert reward to 32-bit float.  # Konvertiere die Belohnung in einen 32-Bit-Gleitkommawert.
    terminated_mod = terminated  # Store the termination status.  # Speichere den Beendigungsstatus.
    truncated_mod = truncated  # Store the truncation status.  # Speichere den Kürzungsstatus.
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info  # Return the modified sample.  # Gibt die modifizierte Probe zurück.


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):  # Define another function for local buffer sample focusing on progress.  # Definiere eine weitere Funktion für lokale Pufferproben, die sich auf den Fortschritt konzentriert.
    """
    Input:  # Eingabe:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)  # prev_act: Aktion, die aus einer vorherigen Beobachtung berechnet wurde und angewendet wurde, um die Beobachtung im Übergang zu erzeugen (aber die unbeeinflusste Beobachtung in Echtzeit-Umgebungen nicht beeinflusst).
        obs, rew, terminated, truncated, info: outcome of the transition  # obs, rew, terminated, truncated, info: Ergebnis des Übergangs
    this function creates the object that will actually be stored in local buffers for networking  # Diese Funktion erstellt das Objekt, das tatsächlich in lokalen Puffern für das Netzwerk gespeichert wird.
    this is to compress the sample before sending it over the Internet/local network  # Dies dient dazu, die Probe vor dem Senden über das Internet/Netzwerk zu komprimieren.
    buffers of such samples will be given as input to the append() method of the memory  # Buffer solcher Proben werden als Eingabe an die append()-Methode des Speichers übergeben.
    the user must define both this function and the append() method of the memory  # Der Benutzer muss sowohl diese Funktion als auch die append()-Methode des Speichers definieren.
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))  # ACHTUNG: prev_act ist die Aktion, die VOR der Beobachtung kommt (d.h. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1], obs[2][-19:])  # speed, LIDAR, and most recent data only  # Geschwindigkeit, LIDAR und nur die neuesten Daten
    rew_mod = np.float32(rew)  # Convert reward to 32-bit float.  # Konvertiere die Belohnung in einen 32-Bit-Gleitkommawert.
    terminated_mod = terminated  # Store the termination status.  # Speichere den Beendigungsstatus.
    truncated_mod = truncated  # Store the truncation status.  # Speichere den Kürzungsstatus.
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info  # Return the modified sample.  # Gibt die modifizierte Probe zurück.


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):  # Define a function for local buffer sample for images (likely TM20).  # Definiere eine Funktion für lokale Pufferproben von Bildern (wahrscheinlich TM20).
    """
    Sample compressor for MemoryTMFull  # Probenkompressor für MemoryTMFull
    Input:  # Eingabe:
        prev_act: action computed from a previous observation and applied to yield obs in the transition  # prev_act: Aktion, die aus einer vorherigen Beobachtung berechnet wurde und angewendet wurde, um die Beobachtung im Übergang zu erzeugen
        obs, rew, terminated, truncated, info: outcome of the transition  # obs, rew, terminated, truncated, info: Ergebnis des Übergangs
    this function creates the object that will actually be stored in local buffers for networking  # Diese Funktion erstellt das Objekt, das tatsächlich in lokalen Puffern für das Netzwerk gespeichert wird.
    this is to compress the sample before sending it over the Internet/local network  # Dies dient dazu, die Probe vor dem Senden über das Internet/Netzwerk zu komprimieren.
    buffers of such samples will be given as input to the append() method of the memory  # Buffer solcher Proben werden als Eingabe an die append()-Methode des Speichers übergeben.
    the user must define both this function and the append() method of the memory  # Der Benutzer muss sowohl diese Funktion als auch die append()-Methode des Speichers definieren.
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))  # ACHTUNG: prev_act ist die Aktion, die VOR der Beobachtung kommt (d.h. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    prev_act_mod = prev_act  # Keep the previous action.  # Behalte die vorherige Aktion.
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))  # Extract and process image data, scaling the last element of obs.  # Extrahiere und verarbeite Bilddaten, indem das letzte Element von obs skaliert wird.
    rew_mod = rew  # Keep the reward unchanged.  # Behalte die Belohnung unverändert.
    terminated_mod = terminated  # Store the termination status.  # Speichere den Beendigungsstatus.
    truncated_mod = truncated  # Store the truncation status.  # Speichere den Kürzungsstatus.
    info_mod = info  # Store additional information.  # Speichere zusätzliche Informationen.
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod  # Return the modified sample.  # Gibt die modifizierte Probe zurück.


# FUNCTIONS ====================================================  # FUNKTIONEN ====================================================

def last_true_in_list(li):  # Define a function to find the index of the last true value in a list.  # Definiere eine Funktion, um den Index des letzten "True"-Werts in einer Liste zu finden.
    for i in reversed(range(len(li))):  # Iterate over the list in reverse order.  # Durchlaufe die Liste in umgekehrter Reihenfolge.
        if li[i]:  # Check if the current element is true.  # Überprüfe, ob das aktuelle Element wahr ist.
            return i  # Return the index of the last true value.  # Gibt den Index des letzten "True"-Werts zurück.
    return None  # Return None if no true value is found.  # Gibt None zurück, wenn kein wahrer Wert gefunden wurde.


def replace_hist_before_eoe(hist, eoe_idx_in_hist):  # Define a function to replace history before the End of Episode (EOE).  # Definiere eine Funktion, um die Geschichte vor dem Ende der Episode (EOE) zu ersetzen.
    """
    Pads the history hist before the End Of Episode (EOE) index.  # Paddiert die Geschichte hist vor dem End-of-Episode (EOE)-Index.
    
    Previous entries in hist are padded with copies of the first element occurring after EOE.  # Frühere Einträge in hist werden mit Kopien des ersten Elements nach dem EOE gepolstert.
    """
    last_idx = len(hist) - 1  # Get the last index of the history list.  # Hole den letzten Index der Geschichtsliste.
    assert eoe_idx_in_hist <= last_idx, f"replace_hist_before_eoe: eoe_idx_in_hist:{eoe_idx_in_hist}, last_idx:{last_idx}"  # Ensure the EOE index is within bounds.  # Stelle sicher, dass der EOE-Index innerhalb der Grenzen liegt.
    if 0 <= eoe_idx_in_hist < last_idx:  # Check if the EOE index is valid.  # Überprüfe, ob der EOE-Index gültig ist.
        for i in reversed(range(len(hist))):  # Iterate through the list in reverse order.  # Durchlaufe die Liste in umgekehrter Reihenfolge.
            if i <= eoe_idx_in_hist:  # If the current index is at or before the EOE index.  # Wenn der aktuelle Index vor oder am EOE-Index liegt.
                hist[i] = hist[i + 1]  # Replace the element with the next element.  # Ersetze das Element mit dem nächsten Element.


class GenericTorchMemory(TorchMemory):  # Defines a class that inherits from TorchMemory.  # Definiert eine Klasse, die von TorchMemory erbt.
    def __init__(self,  # Constructor to initialize the class.  # Konstruktor, um die Klasse zu initialisieren.
                 memory_size=1e6,  # Memory size, default is 1 million.  # Speichergröße, Standardwert ist 1 Million.
                 batch_size=1,  # Batch size, default is 1.  # Batch-Größe, Standardwert ist 1.
                 dataset_path="",  # Path to the dataset.  # Pfad zum Datensatz.
                 nb_steps=1,  # Number of steps for processing.  # Anzahl der Schritte für die Verarbeitung.
                 sample_preprocessor: callable = None,  # Optional function for preprocessing samples.  # Optionales Funktionsargument für die Vorverarbeitung von Proben.
                 crc_debug=False,  # Debug flag for CRC checks.  # Debug-Flag für CRC-Prüfungen.
                 device="cpu"):  # Device to use for computation (default is CPU).  # Gerät zur Verwendung für Berechnungen (Standard ist CPU).
        super().__init__(memory_size=memory_size,  # Calls the constructor of the parent class.  # Ruft den Konstruktor der Elternklasse auf.
                         batch_size=batch_size,  # Passes batch size to the parent class.  # Übergibt die Batch-Größe an die Elternklasse.
                         dataset_path=dataset_path,  # Passes dataset path to the parent class.  # Übergibt den Datensatzpfad an die Elternklasse.
                         nb_steps=nb_steps,  # Passes number of steps to the parent class.  # Übergibt die Anzahl der Schritte an die Elternklasse.
                         sample_preprocessor=sample_preprocessor,  # Passes the sample preprocessor to the parent class.  # Übergibt den Probenvorverarbeiter an die Elternklasse.
                         crc_debug=crc_debug,  # Passes the CRC debug flag to the parent class.  # Übergibt das CRC-Debug-Flag an die Elternklasse.
                         device=device)  # Passes the device to the parent class.  # Übergibt das Gerät an die Elternklasse.

    def append_buffer(self, buffer):  # Method to append data to the buffer.  # Methode, um Daten zum Puffer hinzuzufügen.
        
        # parse:  # Parsing the data from the buffer.  # Parst die Daten aus dem Puffer.
        d0 = [b[0] for b in buffer.memory]  # Extract actions from the buffer.  # Extrahiert Aktionen aus dem Puffer.
        d1 = [b[1] for b in buffer.memory]  # Extract observations from the buffer.  # Extrahiert Beobachtungen aus dem Puffer.
        d2 = [b[2] for b in buffer.memory]  # Extract rewards from the buffer.  # Extrahiert Belohnungen aus dem Puffer.
        d3 = [b[3] for b in buffer.memory]  # Extract termination flags from the buffer.  # Extrahiert Beendigungsflaggen aus dem Puffer.
        d4 = [b[4] for b in buffer.memory]  # Extract truncation flags from the buffer.  # Extrahiert Abschneide-Flaggen aus dem Puffer.
        d5 = [b[5] for b in buffer.memory]  # Extract additional info from the buffer.  # Extrahiert zusätzliche Infos aus dem Puffer.
        d6 = [b[3] or b[4] for b in buffer.memory]  # Combine termination and truncation flags into a single list.  # Kombiniert Beendigungs- und Abschneide-Flaggen in einer einzigen Liste.

        # append:  # Appends the parsed data to the internal storage.  # Fügt die geparsten Daten zum internen Speicher hinzu.
        if self.__len__() > 0:  # Checks if there is already data in the memory.  # Überprüft, ob bereits Daten im Speicher vorhanden sind.
            self.data[0] += d0  # Adds new actions to the data.  # Fügt neue Aktionen zu den Daten hinzu.
            self.data[1] += d1  # Adds new observations to the data.  # Fügt neue Beobachtungen zu den Daten hinzu.
            self.data[2] += d2  # Adds new rewards to the data.  # Fügt neue Belohnungen zu den Daten hinzu.
            self.data[3] += d3  # Adds new termination flags to the data.  # Fügt neue Beendigungs-Flaggen zu den Daten hinzu.
            self.data[4] += d4  # Adds new truncation flags to the data.  # Fügt neue Abschneide-Flaggen zu den Daten hinzu.
            self.data[5] += d5  # Adds new additional info to the data.  # Fügt neue zusätzliche Infos zu den Daten hinzu.
            self.data[6] += d6  # Adds new termination/truncation flags to the data.  # Fügt neue Beendigungs-/Abschneide-Flaggen zu den Daten hinzu.
        else:  # If there is no data, initializes the data storage.  # Wenn keine Daten vorhanden sind, wird der Datenspeicher initialisiert.
            self.data.append(d0)  # Initializes actions data.  # Initialisiert die Aktionsdaten.
            self.data.append(d1)  # Initializes observations data.  # Initialisiert die Beobachtungsdaten.
            self.data.append(d2)  # Initializes rewards data.  # Initialisiert die Belohnungsdaten.
            self.data.append(d3)  # Initializes termination flags data.  # Initialisiert die Beendigungs-Flaggen-Daten.
            self.data.append(d4)  # Initializes truncation flags data.  # Initialisiert die Abschneide-Flaggen-Daten.
            self.data.append(d5)  # Initializes additional info data.  # Initialisiert die zusätzlichen Informationsdaten.
            self.data.append(d6)  # Initializes termination/truncation flags data.  # Initialisiert die Beendigungs-/Abschneide-Flaggen-Daten.

        # trim  # Trims the memory to the defined size.  # Kürzt den Speicher auf die definierte Größe.
        to_trim = int(self.__len__() - self.memory_size)  # Calculates how much memory should be trimmed.  # Berechnet, wie viel Speicher gekürzt werden soll.
        if to_trim > 0:  # If there is excess memory, it will be trimmed.  # Wenn zu viel Speicher vorhanden ist, wird er gekürzt.
            self.data[0] = self.data[0][to_trim:]  # Trims actions data.  # Kürzt die Aktionsdaten.
            self.data[1] = self.data[1][to_trim:]  # Trims observations data.  # Kürzt die Beobachtungsdaten.
            self.data[2] = self.data[2][to_trim:]  # Trims rewards data.  # Kürzt die Belohnungsdaten.
            self.data[3] = self.data[3][to_trim:]  # Trims termination flags data.  # Kürzt die Beendigungs-Flaggen-Daten.
            self.data[4] = self.data[4][to_trim:]  # Trims truncation flags data.  # Kürzt die Abschneide-Flaggen-Daten.
            self.data[5] = self.data[5][to_trim:]  # Trims additional info data.  # Kürzt die zusätzlichen Informationsdaten.
            self.data[6] = self.data[6][to_trim:]  # Trims termination/truncation flags data.  # Kürzt die Beendigungs-/Abschneide-Flaggen-Daten.

    def __len__(self):  # Method to get the length of the data.  # Methode, um die Länge der Daten zu erhalten.
        if len(self.data) == 0:  # If no data exists, return 0.  # Wenn keine Daten existieren, wird 0 zurückgegeben.
            return 0  # Returns 0 if data is empty.  # Gibt 0 zurück, wenn keine Daten vorhanden sind.
        res = len(self.data[0]) - 1  # Calculates the length of the data, minus 1.  # Berechnet die Länge der Daten, minus 1.
        if res < 0:  # If the length is negative, return 0.  # Wenn die Länge negativ ist, wird 0 zurückgegeben.
            return 0  # Returns 0 for negative lengths.  # Gibt 0 für negative Längen zurück.
        else:  # Otherwise, return the calculated length.  # Andernfalls wird die berechnete Länge zurückgegeben.
            return res  # Returns the length of the data.  # Gibt die Länge der Daten zurück.

    def get_transition(self, item):  # Method to get a transition from the memory buffer.  # Methode, um eine Übergang aus dem Speich Puffer zu erhalten.
        
        # This is a hack to avoid invalid transitions from terminal to initial  # This handles invalid transitions from terminal states to initial states.  # Dies ist ein Trick, um ungültige Übergänge von Endzuständen zu Anfangszuständen zu vermeiden.
        while self.data[6][item]:  # While the transition is invalid (done flag is set).  # Solange der Übergang ungültig ist (done-Flag gesetzt).
            item = random.randint(a=0, b=self.__len__() - 1)  # Randomly select a valid transition.  # Wählt zufällig einen gültigen Übergang.

        idx_last = item  # Set the current index as last.  # Setzt den aktuellen Index als den letzten.
        idx_now = item + 1  # Set the next index as current.  # Setzt den nächsten Index als den aktuellen.

        last_obs = self.data[1][idx_last]  # Get the last observation.  # Holt die letzte Beobachtung.
        new_act = self.data[0][idx_now]  # Get the action taken in the new state.  # Holt die Aktion, die im neuen Zustand ausgeführt wurde.
        rew = self.data[2][idx_now]  # Get the reward for the new action.  # Holt die Belohnung für die neue Aktion.
        new_obs = self.data[1][idx_now]  # Get the new observation.  # Holt die neue Beobachtung.
        terminated = self.data[3][idx_now]  # Get the termination flag for the new state.  # Holt das Beendigungs-Flag für den neuen Zustand.
        truncated = self.data[4][idx_now]  # Get the truncation flag for the new state.  # Holt das Abschneide-Flag für den neuen Zustand.
        info = self.data[5][idx_now]  # Get additional information for the new state.  # Holt zusätzliche Informationen für den neuen Zustand.

        return last_obs, new_act, rew, new_obs, terminated, truncated, info  # Returns the full transition data.  # Gibt die vollständigen Übergangs-Daten zurück.


class MemoryTM(TorchMemory):  # Define the MemoryTM class that inherits from TorchMemory class.  # Definiert die Klasse MemoryTM, die von der Klasse TorchMemory erbt.
    def __init__(self,  # Constructor of the class to initialize memory parameters.  # Konstruktor der Klasse, um die Speicherparameter zu initialisieren.
                 memory_size=None,  # Memory size parameter.  # Speichergröße-Parameter.
                 batch_size=None,  # Batch size parameter for memory operations.  # Batch-Größe Parameter für Speicheroperationen.
                 dataset_path="",  # Path to dataset (default empty string).  # Pfad zum Datensatz (Standard leere Zeichenkette).
                 imgs_obs=4,  # Number of images observed at once (default 4).  # Anzahl der auf einmal beobachteten Bilder (Standard 4).
                 act_buf_len=1,  # Length of action buffer (default 1).  # Länge des Aktionspuffers (Standard 1).
                 nb_steps=1,  # Number of steps to simulate (default 1).  # Anzahl der Schritte, die simuliert werden sollen (Standard 1).
                 sample_preprocessor: callable = None,  # Optional function for preprocessing samples.  # Optionale Funktion zur Vorverarbeitung von Proben.
                 crc_debug=False,  # CRC debug flag for debugging.  # CRC-Debug-Flag zur Fehlerbehebung.
                 device="cpu"):  # Device (e.g., CPU or GPU) for computations.  # Gerät (z. B. CPU oder GPU) für Berechnungen.
        self.imgs_obs = imgs_obs  # Store the number of observed images.  # Speichert die Anzahl der beobachteten Bilder.
        self.act_buf_len = act_buf_len  # Store the action buffer length.  # Speichert die Länge des Aktionspuffers.
        self.min_samples = max(self.imgs_obs, self.act_buf_len)  # Ensure the minimum samples are at least as large as the max of imgs_obs and act_buf_len.  # Stellt sicher, dass die Mindestanzahl von Proben mindestens so groß ist wie das Maximum von imgs_obs und act_buf_len.
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)  # Calculate the offset for image samples.  # Berechnet den Versatz für Bildproben.
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)  # Calculate the offset for action samples.  # Berechnet den Versatz für Aktionsproben.
        super().__init__(memory_size=memory_size,  # Initialize the parent class (TorchMemory) with memory size.  # Initialisiert die Elternklasse (TorchMemory) mit der Speichergröße.
                         batch_size=batch_size,  # Pass batch size to the parent class.  # Übergibt die Batch-Größe an die Elternklasse.
                         dataset_path=dataset_path,  # Pass dataset path to the parent class.  # Übergibt den Datensatzpfad an die Elternklasse.
                         nb_steps=nb_steps,  # Pass number of steps to the parent class.  # Übergibt die Anzahl der Schritte an die Elternklasse.
                         sample_preprocessor=sample_preprocessor,  # Pass sample preprocessor to the parent class.  # Übergibt den Sample-Preprocessor an die Elternklasse.
                         crc_debug=crc_debug,  # Pass CRC debug flag to the parent class.  # Übergibt das CRC-Debug-Flag an die Elternklasse.
                         device=device)  # Pass the device (CPU or GPU) to the parent class.  # Übergibt das Gerät (CPU oder GPU) an die Elternklasse.

    def append_buffer(self, buffer):  # Define the method to append data to the buffer (not implemented here).  # Definiert die Methode zum Anhängen von Daten an den Puffer (hier nicht implementiert).
        raise NotImplementedError  # Raise an error if not implemented.  # Fehler wird ausgelöst, wenn nicht implementiert.

    def __len__(self):  # Define the method to get the length of the memory.  # Definiert die Methode zur Bestimmung der Länge des Speichers.
        if len(self.data) == 0:  # Check if data is empty.  # Überprüft, ob die Daten leer sind.
            return 0  # Return 0 if no data.  # Gibt 0 zurück, wenn keine Daten vorhanden sind.
        res = len(self.data[0]) - self.min_samples - 1  # Calculate the length of the data minus minimum samples and 1.  # Berechnet die Länge der Daten minus die Mindestanzahl von Proben und 1.
        if res < 0:  # If the result is less than 0.  # Wenn das Ergebnis kleiner als 0 ist.
            return 0  # Return 0.  # Gibt 0 zurück.
        else:  # Otherwise.  # Ansonsten.
            return res  # Return the calculated length.  # Gibt die berechnete Länge zurück.

    def get_transition(self, item):  # Define method to get the transition for an item (not implemented here).  # Definiert eine Methode, um die Übergänge für ein Element zu erhalten (hier nicht implementiert).
        raise NotImplementedError  # Raise an error if not implemented.  # Fehler wird ausgelöst, wenn nicht implementiert.

class MemoryTMLidar(MemoryTM):  # Define the MemoryTMLidar class that inherits from MemoryTM class.  # Definiert die Klasse MemoryTMLidar, die von der Klasse MemoryTM erbt.
    def get_transition(self, item):  # Implement get_transition for Lidar-based memory.  # Implementiert get_transition für Lidar-basierten Speicher.
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """  # Warning about how observations and actions are stored and processed.  # Warnung darüber, wie Beobachtungen und Aktionen gespeichert und verarbeitet werden.
        if self.data[4][item + self.min_samples - 1]:  # Check if the data at the specified index is valid.  # Überprüft, ob die Daten am angegebenen Index gültig sind.
            if item == 0:  # If the first item in the buffer.  # Wenn es das erste Element im Puffer ist.
                item += 1  # Move to the next item.  # Gehe zum nächsten Element.
            elif item == self.__len__() - 1:  # If it's the last item in the buffer.  # Wenn es das letzte Element im Puffer ist.
                item -= 1  # Move to the previous item.  # Gehe zum vorherigen Element.
            elif random.random() < 0.5:  # Otherwise, choose a random item.  # Andernfalls wähle ein zufälliges Element.
                item += 1  # Move to the next item.  # Gehe zum nächsten Element.
            else:  # Otherwise.  # Andernfalls.
                item -= 1  # Move to the previous item.  # Gehe zum vorherigen Element.

        idx_last = item + self.min_samples - 1  # Calculate the index of the last sample.  # Berechnet den Index der letzten Probe.
        idx_now = item + self.min_samples  # Calculate the index of the current sample.  # Berechnet den Index der aktuellen Probe.

        acts = self.load_acts(item)  # Load actions for the item.  # Lade die Aktionen für das Element.
        last_act_buf = acts[:-1]  # Store all actions except the last one.  # Speichert alle Aktionen außer der letzten.
        new_act_buf = acts[1:]  # Store all actions except the first one.  # Speichert alle Aktionen außer der ersten.

        imgs = self.load_imgs(item)  # Load images for the item.  # Lade Bilder für das Element.
        imgs_last_obs = imgs[:-1]  # Store all images except the last one.  # Speichert alle Bilder außer dem letzten.
        imgs_new_obs = imgs[1:]  # Store all images except the first one.  # Speichert alle Bilder außer dem ersten.

        # If a reset transition has influenced the observation, special care must be taken  # Wenn eine Reset-Übergang die Beobachtung beeinflusst hat, muss besondere Vorsicht walten.
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # Get last EOE values from the data.  # Holt die letzten EOE-Werte aus den Daten.
        last_eoe_idx = last_true_in_list(last_eoes)  # Find the last True index in EOE.  # Findet den letzten True-Index in EOE.

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"  # Assert that the last EOE is valid.  # Stellt sicher, dass das letzte EOE gültig ist.

        if last_eoe_idx is not None:  # If EOE exists.  # Wenn EOE existiert.
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)  # Replace history before EOE for new actions.  # Ersetzt die Historie vor EOE für neue Aktionen.
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)  # Replace history before EOE for last actions.  # Ersetzt die Historie vor EOE für letzte Aktionen.
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)  # Replace history before EOE for new images.  # Ersetzt die Historie vor EOE für neue Bilder.
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)  # Replace history before EOE for last images.  # Ersetzt die Historie vor EOE für letzte Bilder.

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)  # Flatten the new observation images.  # Flacht die neuen Beobachtungsbilder ab.
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)  # Flatten the last observation images.  # Flacht die letzten Beobachtungsbilder ab.

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)  # Prepare the last observation.  # Bereitet die letzte Beobachtung vor.
        new_act = self.data[1][idx_now]  # Get the new action.  # Holt die neue Aktion.
        rew = np.float32(self.data[5][idx_now])  # Get the reward for the current sample.  # Holt die Belohnung für die aktuelle Probe.
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)  # Prepare the new observation.  # Bereitet die neue Beobachtung vor.
        terminated = self.data[7][idx_now]  # Check if the episode is terminated.  # Überprüft, ob die Episode beendet ist.
        truncated = self.data[8][idx_now]  # Check if the episode is truncated.  # Überprüft, ob die Episode abgeschnitten ist.
        info = self.data[6][idx_now]  # Get additional information.  # Holt zusätzliche Informationen.
        return last_obs, new_act, rew, new_obs, terminated, truncated, info  # Return the transition.  # Gibt die Übergangsdaten zurück.

    def load_imgs(self, item):  # Method to load images for an item.  # Methode zum Laden von Bildern für ein Element.
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]  # Load images from data.  # Lädt Bilder aus den Daten.
        return np.stack(res)  # Return the stacked images.  # Gibt die gestapelten Bilder zurück.

    def load_acts(self, item):  # Method to load actions for an item.  # Methode zum Laden von Aktionen für ein Element.
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]  # Load actions from data.  # Lädt Aktionen aus den Daten.
        return res  # Return the actions.  # Gibt die Aktionen zurück.

    def append_buffer(self, buffer):  # Method to append buffer data.  # Methode zum Hinzufügen von Pufferdaten.
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """  # Explanation that buffer contains samples and needs to retain CRC info.  # Erklärung, dass der Puffer Proben enthält und CRC-Info behalten muss.

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0  # Get the index of the first new sample.  # Holt den Index der ersten neuen Probe.

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # Generate indexes for new samples.  # Erzeugt Indizes für neue Proben.
        d1 = [b[0] for b in buffer.memory]  # Extract actions from buffer.  # Extrahiert Aktionen aus dem Puffer.
        d2 = [b[1][0] for b in buffer.memory]  # Extract speed data.  # Extrahiert Geschwindigkeitsdaten.
        d3 = [b[1][1] for b in buffer.memory]  # Extract Lidar data.  # Extrahiert Lidar-Daten.
        d4 = [b[3] or b[4] for b in buffer.memory]  # Extract EOE values (terminated or truncated).  # Extrahiert EOE-Werte (beendet oder abgeschnitten).
        d5 = [b[2] for b in buffer.memory]  # Extract rewards.  # Extrahiert Belohnungen.
        d6 = [b[5] for b in buffer.memory]  # Extract additional info.  # Extrahiert zusätzliche Infos.
        d7 = [b[3] for b in buffer.memory]  # Extract termination flag.  # Extrahiert den Beendigungs-Flag.
        d8 = [b[4] for b in buffer.memory]  # Extract truncated flag.  # Extrahiert den Abbruch-Flag.

        if self.__len__() > 0:  # If there is already data.  # Wenn bereits Daten vorhanden sind.
            self.data[0] += d0  # Append new indexes to the data.  # Fügt neue Indizes zu den Daten hinzu.
            self.data[1] += d1  # Append actions to the data.  # Fügt Aktionen zu den Daten hinzu.
            self.data[2] += d2  # Append speeds to the data.  # Fügt Geschwindigkeiten zu den Daten hinzu.
            self.data[3] += d3  # Append Lidar data to the data.  # Fügt Lidar-Daten zu den Daten hinzu.
            self.data[4] += d4  # Append EOE values to the data.  # Fügt EOE-Werte zu den Daten hinzu.
            self.data[5] += d5  # Append rewards to the data.  # Fügt Belohnungen zu den Daten hinzu.
            self.data[6] += d6  # Append info to the data.  # Fügt Infos zu den Daten hinzu.
            self.data[7] += d7  # Append termination flags to the data.  # Fügt Beendigungs-Flags zu den Daten hinzu.
            self.data[8] += d8  # Append truncated flags to the data.  # Fügt Abbruch-Flags zu den Daten hinzu.
        else:  # If there is no data yet.  # Wenn noch keine Daten vorhanden sind.
            self.data.append(d0)  # Initialize data with new indexes.  # Initialisiert die Daten mit neuen Indizes.
            self.data.append(d1)  # Initialize data with new actions.  # Initialisiert die Daten mit neuen Aktionen.
            self.data.append(d2)  # Initialize data with new speeds.  # Initialisiert die Daten mit neuen Geschwindigkeiten.
            self.data.append(d3)  # Initialize data with new Lidar data.  # Initialisiert die Daten mit neuen Lidar-Daten.
            self.data.append(d4)  # Initialize data with new EOE values.  # Initialisiert die Daten mit neuen EOE-Werten.
            self.data.append(d5)  # Initialize data with new rewards.  # Initialisiert die Daten mit neuen Belohnungen.
            self.data.append(d6)  # Initialize data with new info.  # Initialisiert die Daten mit neuen Infos.
            self.data.append(d7)  # Initialize data with new termination flags.  # Initialisiert die Daten mit neuen Beendigungs-Flags.
            self.data.append(d8)  # Initialize data with new truncated flags.  # Initialisiert die Daten mit neuen Abbruch-Flags.

        to_trim = self.__len__() - self.memory_size  # Check if data exceeds memory size.  # Überprüft, ob die Daten die Speichergröße überschreiten.
        if to_trim > 0:  # If there is excess data.  # Wenn es überschüssige Daten gibt.
            self.data[0] = self.data[0][to_trim:]  # Trim excess indexes.  # Kürzt überschüssige Indizes.
            self.data[1] = self.data[1][to_trim:]  # Trim excess actions.  # Kürzt überschüssige Aktionen.
            self.data[2] = self.data[2][to_trim:]  # Trim excess speeds.  # Kürzt überschüssige Geschwindigkeiten.
            self.data[3] = self.data[3][to_trim:]  # Trim excess Lidar data.  # Kürzt überschüssige Lidar-Daten.
            self.data[4] = self.data[4][to_trim:]  # Trim excess EOE values.  # Kürzt überschüssige EOE-Werte.
            self.data[5] = self.data[5][to_trim:]  # Trim excess rewards.  # Kürzt überschüssige Belohnungen.
            self.data[6] = self.data[6][to_trim:]  # Trim excess info.  # Kürzt überschüssige Infos.
            self.data[7] = self.data[7][to_trim:]  # Trim excess termination flags.  # Kürzt überschüssige Beendigungs-Flags.
            self.data[8] = self.data[8][to_trim:]  # Trim excess truncated flags.  # Kürzt überschüssige Abbruch-Flags.

        return self  # Return the updated memory object.  # Gibt das aktualisierte Speicherobjekt zurück.




class MemoryTMLidarProgress(MemoryTM):  # Defines a new class 'MemoryTMLidarProgress' that inherits from 'MemoryTM'.  # Definiert eine neue Klasse 'MemoryTMLidarProgress', die von 'MemoryTM' erbt.
    def get_transition(self, item):  # Defines the method 'get_transition', which takes an 'item' as input.  # Definiert die Methode 'get_transition', die ein 'item' als Eingabe erhält.
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """  # Explanation for the function's usage and precautions.  # Erklärung zur Verwendung und Vorsicht bei der Funktion.
        
        if self.data[4][item + self.min_samples - 1]:  # Checks if a condition in the data at the specified index is True.  # Überprüft, ob eine Bedingung in den Daten an dem angegebenen Index wahr ist.
            if item == 0:  # if first item of the buffer  # Wenn es das erste Element des Puffers ist
                item += 1  # Increment the index.  # Erhöht den Index.
            elif item == self.__len__() - 1:  # if last item of the buffer  # Wenn es das letzte Element des Puffers ist
                item -= 1  # Decrement the index.  # Verringert den Index.
            elif random.random() < 0.5:  # otherwise, sample randomly  # Ansonsten zufällig sampeln
                item += 1  # Increment the index randomly.  # Erhöht den Index zufällig.
            else:
                item -= 1  # Decrement the index randomly.  # Verringert den Index zufällig.

        idx_last = item + self.min_samples - 1  # Sets index for last sample based on current item.  # Setzt den Index für das letzte Sample basierend auf dem aktuellen Element.
        idx_now = item + self.min_samples  # Sets index for the current sample.  # Setzt den Index für das aktuelle Sample.

        acts = self.load_acts(item)  # Loads actions from a method 'load_acts' for the given item.  # Lädt Aktionen aus einer Methode 'load_acts' für das gegebene Element.
        last_act_buf = acts[:-1]  # All actions except the last one.  # Alle Aktionen außer der letzten.
        new_act_buf = acts[1:]  # All actions except the first one.  # Alle Aktionen außer der ersten.

        imgs = self.load_imgs(item)  # Loads images from a method 'load_imgs' for the given item.  # Lädt Bilder aus einer Methode 'load_imgs' für das gegebene Element.
        imgs_last_obs = imgs[:-1]  # All images except the last one.  # Alle Bilder außer dem letzten.
        imgs_new_obs = imgs[1:]  # All images except the first one.  # Alle Bilder außer dem ersten.

        # if a reset transition has influenced the observation, special care must be taken  # Falls eine Zurücksetz-Übergang die Beobachtung beeinflusst hat, muss besondere Vorsicht walten.
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values  # self.min_samples Werte
        last_eoe_idx = last_true_in_list(last_eoes)  # Finds the last occurrence of True in the list.  # Findet das letzte Vorkommen von True in der Liste.

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"  # Asserts that the last eoe index is correct.  # Stellt sicher, dass der letzte eoe-Index korrekt ist.

        if last_eoe_idx is not None:  # If the last eoe index is found  # Wenn der letzte eoe-Index gefunden wurde
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)  # Replaces history in new_act_buf before the eoe.  # Ersetzt die Geschichte im new_act_buf vor dem eoe.
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)  # Replaces history in last_act_buf before the eoe.  # Ersetzt die Geschichte im last_act_buf vor dem eoe.
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)  # Replaces history in imgs_new_obs before the eoe.  # Ersetzt die Geschichte in imgs_new_obs vor dem eoe.
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)  # Replaces history in imgs_last_obs before the eoe.  # Ersetzt die Geschichte in imgs_last_obs vor dem eoe.

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)  # Flattens the array of new observations.  # Flatten das Array der neuen Beobachtungen.
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)  # Flattens the array of last observations.  # Flatten das Array der letzten Beobachtungen.

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], imgs_last_obs, *last_act_buf)  # Forms the last observation tuple.  # Bildet das Tupel der letzten Beobachtung.
        new_act = self.data[1][idx_now]  # Retrieves the new action from the data.  # Holt die neue Aktion aus den Daten.
        rew = np.float32(self.data[5][idx_now])  # Retrieves the reward and converts to a float32.  # Holt die Belohnung und konvertiert sie zu float32.
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], imgs_new_obs, *new_act_buf)  # Forms the new observation tuple.  # Bildet das Tupel der neuen Beobachtung.
        terminated = self.data[8][idx_now]  # Retrieves the termination flag for the current observation.  # Holt das Beendigungsflag für die aktuelle Beobachtung.
        truncated = self.data[9][idx_now]  # Retrieves the truncation flag for the current observation.  # Holt das Abbruchflag für die aktuelle Beobachtung.
        info = self.data[6][idx_now]  # Retrieves additional information for the current observation.  # Holt zusätzliche Informationen für die aktuelle Beobachtung.
        return last_obs, new_act, rew, new_obs, terminated, truncated, info  # Returns the observed and action data.  # Gibt die beobachteten und Aktionsdaten zurück.

    def load_imgs(self, item):  # Defines a method to load images.  # Definiert eine Methode zum Laden von Bildern.
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]  # Loads images from the data.  # Lädt Bilder aus den Daten.
        return np.stack(res)  # Stacks the images into a single array.  # Stapelt die Bilder in ein einziges Array.

    def load_acts(self, item):  # Defines a method to load actions.  # Definiert eine Methode zum Laden von Aktionen.
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]  # Loads actions from the data.  # Lädt Aktionen aus den Daten.
        return res  # Returns the loaded actions.  # Gibt die geladenen Aktionen zurück.

    def append_buffer(self, buffer):  # Defines a method to append data to the buffer.  # Definiert eine Methode zum Anhängen von Daten an den Puffer.
        """
        buffer is a list of samples (act, obs, rew, truncated, terminated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """  # Explanation of the buffer parameter.  # Erklärung des Buffer-Parameters.

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0  # Sets the index for the first data entry.  # Setzt den Index für den ersten Dateneintrag.

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # Indexes for the buffer data.  # Indizes für die Buffer-Daten.
        d1 = [b[0] for b in buffer.memory]  # Actions from the buffer.  # Aktionen aus dem Buffer.
        d2 = [b[1][0] for b in buffer.memory]  # Speed data from the buffer.  # Geschwindigkeitsdaten aus dem Buffer.
        d3 = [b[1][2] for b in buffer.memory]  # Lidar data from the buffer.  # Lidar-Daten aus dem Buffer.
        d4 = [b[3] or b[4] for b in buffer.memory]  # EOE (End of Episode) flags.  # EOE (Ende der Episode) Flags.
        d5 = [b[2] for b in buffer.memory]  # Rewards from the buffer.  # Belohnungen aus dem Buffer.
        d6 = [b[5] for b in buffer.memory]  # Info data from the buffer.  # Info-Daten aus dem Buffer.
        d7 = [b[1][1] for b in buffer.memory]  # Progress data from the buffer.  # Fortschrittsdaten aus dem Buffer.
        d8 = [b[3] for b in buffer.memory]  # Termination flags from the buffer.  # Beendigungsflags aus dem Buffer.
        d9 = [b[4] for b in buffer.memory]  # Truncation flags from the buffer.  # Abbruchflags aus dem Buffer.

        if self.__len__() > 0:  # Checks if the memory is non-empty.  # Überprüft, ob der Speicher nicht leer ist.
            self.data[0] += d0  # Adds new data to the existing data list.  # Fügt neue Daten zur bestehenden Datenliste hinzu.
            self.data[1] += d1  # Adds new actions.  # Fügt neue Aktionen hinzu.
            self.data[2] += d2  # Adds new speeds.  # Fügt neue Geschwindigkeiten hinzu.
            self.data[3] += d3  # Adds new lidar data.  # Fügt neue Lidar-Daten hinzu.
            self.data[4] += d4  # Adds new EOE flags.  # Fügt neue EOE-Flags hinzu.
            self.data[5] += d5  # Adds new rewards.  # Fügt neue Belohnungen hinzu.
            self.data[6] += d6  # Adds new info data.  # Fügt neue Info-Daten hinzu.
            self.data[7] += d7  # Adds new progress data.  # Fügt neue Fortschrittsdaten hinzu.
            self.data[8] += d8  # Adds new termination flags.  # Fügt neue Beendigungsflags hinzu.
            self.data[9] += d9  # Adds new truncation flags.  # Fügt neue Abbruchflags hinzu.
        else:
            self.data.append(d0)  # Appends data if memory is empty.  # Fügt Daten hinzu, wenn der Speicher leer ist.
            self.data.append(d1)  # Appends actions.  # Fügt Aktionen hinzu.
            self.data.append(d2)  # Appends speeds.  # Fügt Geschwindigkeiten hinzu.
            self.data.append(d3)  # Appends lidar data.  # Fügt Lidar-Daten hinzu.
            self.data.append(d4)  # Appends EOE flags.  # Fügt EOE-Flags hinzu.
            self.data.append(d5)  # Appends rewards.  # Fügt Belohnungen hinzu.
            self.data.append(d6)  # Appends info data.  # Fügt Info-Daten hinzu.
            self.data.append(d7)  # Appends progress data.  # Fügt Fortschrittsdaten hinzu.
            self.data.append(d8)  # Appends termination flags.  # Fügt Beendigungsflags hinzu.
            self.data.append(d9)  # Appends truncation flags.  # Fügt Abbruchflags hinzu.

        to_trim = self.__len__() - self.memory_size  # Calculates the excess memory to trim.  # Berechnet den überschüssigen Speicher zum Kürzen.
        if to_trim > 0:  # If there is excess memory to trim  # Wenn es überschüssigen Speicher zum Kürzen gibt
            self.data[0] = self.data[0][to_trim:]  # Trims the data.  # Kürzt die Daten.
            self.data[1] = self.data[1][to_trim:]  # Trims the actions.  # Kürzt die Aktionen.
            self.data[2] = self.data[2][to_trim:]  # Trims the speeds.  # Kürzt die Geschwindigkeiten.
            self.data[3] = self.data[3][to_trim:]  # Trims the lidar data.  # Kürzt die Lidar-Daten.
            self.data[4] = self.data[4][to_trim:]  # Trims the EOE flags.  # Kürzt die EOE-Flags.
            self.data[5] = self.data[5][to_trim:]  # Trims the rewards.  # Kürzt die Belohnungen.
            self.data[6] = self.data[6][to_trim:]  # Trims the info data.  # Kürzt die Info-Daten.
            self.data[7] = self.data[7][to_trim:]  # Trims the progress data.  # Kürzt die Fortschrittsdaten.
            self.data[8] = self.data[8][to_trim:]  # Trims the termination flags.  # Kürzt die Beendigungsflags.
            self.data[9] = self.data[9][to_trim:]  # Trims the truncation flags.  # Kürzt die Abbruchflags.

        return self  # Returns the updated object.  # Gibt das aktualisierte Objekt zurück.


class MemoryTMFull(MemoryTM):  # Defines a new class MemoryTMFull that inherits from the MemoryTM class.  # Definiert eine neue Klasse MemoryTMFull, die von der Klasse MemoryTM erbt.
    def get_transition(self, item):  # Defines the method to get a transition based on the given 'item'.  # Definiert die Methode, um eine Transition basierend auf dem gegebenen 'item' zu erhalten.
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """  # Warning explaining the handling of data, including the structure of observations and actions.  # Warnung, die den Umgang mit Daten erklärt, einschließlich der Struktur von Beobachtungen und Aktionen.
        
        if self.data[4][item + self.min_samples - 1]:  # Checks if the item in the data buffer exists.  # Überprüft, ob das Element im Datenpuffer existiert.
            if item == 0:  # if first item of the buffer  # wenn das erste Element des Puffers
                item += 1  # Increment item by 1 if it's the first item.  # Erhöhe das Element um 1, wenn es das erste Element ist.
            elif item == self.__len__() - 1:  # if last item of the buffer  # wenn das letzte Element des Puffers
                item -= 1  # Decrement item by 1 if it's the last item.  # Verringere das Element um 1, wenn es das letzte Element ist.
            elif random.random() < 0.5:  # otherwise, sample randomly  # andernfalls, zufällig auswählen
                item += 1  # Increment item randomly.  # Erhöhe das Element zufällig.
            else:
                item -= 1  # Decrement item randomly.  # Verringere das Element zufällig.

        idx_last = item + self.min_samples - 1  # Sets the index for the last item in the sample.  # Setzt den Index für das letzte Element im Sample.
        idx_now = item + self.min_samples  # Sets the index for the current item.  # Setzt den Index für das aktuelle Element.

        acts = self.load_acts(item)  # Loads actions based on the given item.  # Lädt Aktionen basierend auf dem gegebenen Element.
        last_act_buf = acts[:-1]  # Selects all actions except the last one.  # Wählt alle Aktionen außer der letzten aus.
        new_act_buf = acts[1:]  # Selects all actions except the first one.  # Wählt alle Aktionen außer der ersten aus.

        imgs = self.load_imgs(item)  # Loads images based on the given item.  # Lädt Bilder basierend auf dem gegebenen Element.
        imgs_last_obs = imgs[:-1]  # Selects all images except the last one.  # Wählt alle Bilder außer dem letzten aus.
        imgs_new_obs = imgs[1:]  # Selects all images except the first one.  # Wählt alle Bilder außer dem ersten aus.

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # Retrieves the last 'min_samples' values.  # Ruft die letzten 'min_samples' Werte ab.
        last_eoe_idx = last_true_in_list(last_eoes)  # Finds the last occurrence of True in the list.  # Findet das letzte Vorkommen von True in der Liste.

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"  # Asserts that the last_eoe_idx is None or True.  # Stellt sicher, dass last_eoe_idx None oder True ist.

        if last_eoe_idx is not None:  # If there was a reset transition affecting the observation  # Wenn eine Reset-Transition die Beobachtung beeinflusst hat
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)  # Replaces the history before the reset.  # Ersetzt die Historie vor dem Reset.
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)  # Replaces the history before the reset.  # Ersetzt die Historie vor dem Reset.
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)  # Replaces image history before the reset.  # Ersetzt die Bildhistorie vor dem Reset.
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)  # Replaces image history before the reset.  # Ersetzt die Bildhistorie vor dem Reset.

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], self.data[8][idx_last], imgs_last_obs, *last_act_buf)  # Creates a tuple of the last observation.  # Erstellt ein Tupel der letzten Beobachtung.
        new_act = self.data[1][idx_now]  # Retrieves the new action.  # Ruft die neue Aktion ab.
        rew = np.float32(self.data[5][idx_now])  # Retrieves the reward and converts it to a float32 type.  # Ruft die Belohnung ab und konvertiert sie in den Typ float32.
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], imgs_new_obs, *new_act_buf)  # Creates a tuple of the new observation.  # Erstellt ein Tupel der neuen Beobachtung.
        terminated = self.data[9][idx_now]  # Retrieves the termination flag.  # Ruft das Beendigungsflag ab.
        truncated = self.data[10][idx_now]  # Retrieves the truncation flag.  # Ruft das Truncationsflag ab.
        info = self.data[6][idx_now]  # Retrieves the info dictionary.  # Ruft das Info-Dictionary ab.
        return last_obs, new_act, rew, new_obs, terminated, truncated, info  # Returns the gathered information.  # Gibt die gesammelten Informationen zurück.

    def load_imgs(self, item):  # Loads images from the buffer starting at the given item index.  # Lädt Bilder aus dem Puffer, beginnend mit dem angegebenen Elementindex.
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]  # Retrieves a slice of images from the data buffer.  # Ruft einen Ausschnitt von Bildern aus dem Datenpuffer ab.
        return np.stack(res).astype(np.float32) / 256.0  # Stacks the images and normalizes them.  # Stapelt die Bilder und normalisiert sie.

    def load_acts(self, item):  # Loads actions from the buffer based on the given item index.  # Lädt Aktionen aus dem Puffer basierend auf dem angegebenen Elementindex.
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]  # Retrieves a slice of actions from the data buffer.  # Ruft einen Ausschnitt von Aktionen aus dem Datenpuffer ab.
        return res  # Returns the actions.  # Gibt die Aktionen zurück.

    def append_buffer(self, buffer):  # Appends the given buffer of samples to the data.  # Fügt den gegebenen Puffer von Samples zu den Daten hinzu.
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """  # Describes the buffer as a list of samples and notes the importance of keeping the info dictionary.  # Beschreibt den Puffer als Liste von Samples und weist darauf hin, dass das Info-Dictionary beibehalten werden muss.

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0  # Determines the starting index for new data.  # Bestimmt den Startindex für neue Daten.

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # Generates indices for the new data.  # Generiert Indizes für die neuen Daten.
        d1 = [b[0] for b in buffer.memory]  # Extracts actions from the buffer.  # Extrahiert Aktionen aus dem Puffer.
        d2 = [b[1][0] for b in buffer.memory]  # Extracts speeds from the buffer.  # Extrahiert Geschwindigkeiten aus dem Puffer.
        d3 = [b[1][3] for b in buffer.memory]  # Extracts images from the buffer.  # Extrahiert Bilder aus dem Puffer.
        d4 = [b[3] or b[4] for b in buffer.memory]  # Extracts EOE flags (end-of-episode) from the buffer.  # Extrahiert EOE-Flags (Ende der Episode) aus dem Puffer.
        d5 = [b[2] for b in buffer.memory]  # Extracts rewards from the buffer.  # Extrahiert Belohnungen aus dem Puffer.
        d6 = [b[5] for b in buffer.memory]  # Extracts info from the buffer.  # Extrahiert Info aus dem Puffer.
        d7 = [b[1][1] for b in buffer.memory]  # Extracts gears from the buffer.  # Extrahiert Gänge aus dem Puffer.
        d8 = [b[1][2] for b in buffer.memory]  # Extracts RPMs from the buffer.  # Extrahiert Umdrehungen pro Minute aus dem Puffer.
        d9 = [b[3] for b in buffer.memory]  # Extracts termination flags from the buffer.  # Extrahiert Beendigungs-Flags aus dem Puffer.
        d10 = [b[4] for b in buffer.memory]  # Extracts truncation flags from the buffer.  # Extrahiert Truncations-Flags aus dem Puffer.
  
if self.__len__() > 0:  # If the buffer is not empty, append the new data.  # Wenn der Puffer nicht leer ist, füge die neuen Daten hinzu.
            self.data[0] += d0  # Adds new data (d0) to the first element of the buffer.  # Fügt die neuen Daten (d0) zum ersten Element des Puffers hinzu.
            self.data[1] += d1  # Adds new data (d1) to the second element of the buffer.  # Fügt die neuen Daten (d1) zum zweiten Element des Puffers hinzu.
            self.data[2] += d2  # Adds new data (d2) to the third element of the buffer.  # Fügt die neuen Daten (d2) zum dritten Element des Puffers hinzu.
            self.data[3] += d3  # Adds new data (d3) to the fourth element of the buffer.  # Fügt die neuen Daten (d3) zum vierten Element des Puffers hinzu.
            self.data[4] += d4  # Adds new data (d4) to the fifth element of the buffer.  # Fügt die neuen Daten (d4) zum fünften Element des Puffers hinzu.
            self.data[5] += d5  # Adds new data (d5) to the sixth element of the buffer.  # Fügt die neuen Daten (d5) zum sechsten Element des Puffers hinzu.
            self.data[6] += d6  # Adds new data (d6) to the seventh element of the buffer.  # Fügt die neuen Daten (d6) zum siebten Element des Puffers hinzu.
            self.data[7] += d7  # Adds new data (d7) to the eighth element of the buffer.  # Fügt die neuen Daten (d7) zum achten Element des Puffers hinzu.
            self.data[8] += d8  # Adds new data (d8) to the ninth element of the buffer.  # Fügt die neuen Daten (d8) zum neunten Element des Puffers hinzu.
            self.data[9] += d9  # Adds new data (d9) to the tenth element of the buffer.  # Fügt die neuen Daten (d9) zum zehnten Element des Puffers hinzu.
            self.data[10] += d10  # Adds new data (d10) to the eleventh element of the buffer.  # Fügt die neuen Daten (d10) zum elften Element des Puffers hinzu.
        else:  # If the buffer is empty, initialize it with the new data.  # Wenn der Puffer leer ist, initialisiere ihn mit den neuen Daten.
            self.data.append(d0)  # Adds the first data element (d0) to the buffer.  # Fügt das erste Datenelement (d0) zum Puffer hinzu.
            self.data.append(d1)  # Adds the second data element (d1) to the buffer.  # Fügt das zweite Datenelement (d1) zum Puffer hinzu.
            self.data.append(d2)  # Adds the third data element (d2) to the buffer.  # Fügt das dritte Datenelement (d2) zum Puffer hinzu.
            self.data.append(d3)  # Adds the fourth data element (d3) to the buffer.  # Fügt das vierte Datenelement (d3) zum Puffer hinzu.
            self.data.append(d4)  # Adds the fifth data element (d4) to the buffer.  # Fügt das fünfte Datenelement (d4) zum Puffer hinzu.
            self.data.append(d5)  # Adds the sixth data element (d5) to the buffer.  # Fügt das sechste Datenelement (d5) zum Puffer hinzu.
            self.data.append(d6)  # Adds the seventh data element (d6) to the buffer.  # Fügt das siebte Datenelement (d6) zum Puffer hinzu.
            self.data.append(d7)  # Adds the eighth data element (d7) to the buffer.  # Fügt das achte Datenelement (d7) zum Puffer hinzu.
            self.data.append(d8)  # Adds the ninth data element (d8) to the buffer.  # Fügt das neunte Datenelement (d8) zum Puffer hinzu.
            self.data.append(d9)  # Adds the tenth data element (d9) to the buffer.  # Fügt das zehnte Datenelement (d9) zum Puffer hinzu.
            self.data.append(d10)  # Adds the eleventh data element (d10) to the buffer.  # Fügt das elfte Datenelement (d10) zum Puffer hinzu.

        to_trim = self.__len__() - self.memory_size  # Determines how many items need to be removed from the buffer.  # Bestimmt, wie viele Elemente aus dem Puffer entfernt werden müssen.
        if to_trim > 0:  # If there are excess items, trim them.  # Wenn es überschüssige Elemente gibt, entferne sie.
            self.data[0] = self.data[0][to_trim:]  # Trims excess items from the first data element.  # Entfernt überschüssige Elemente aus dem ersten Datenelement.
            self.data[1] = self.data[1][to_trim:]  # Trims excess items from the second data element.  # Entfernt überschüssige Elemente aus dem zweiten Datenelement.
            self.data[2] = self.data[2][to_trim:]  # Trims excess items from the third data element.  # Entfernt überschüssige Elemente aus dem dritten Datenelement.
            self.data[3] = self.data[3][to_trim:]  # Trims excess items from the fourth data element.  # Entfernt überschüssige Elemente aus dem vierten Datenelement.
            self.data[4] = self.data[4][to_trim:]  # Trims excess items from the fifth data element.  # Entfernt überschüssige Elemente aus dem fünften Datenelement.
            self.data[5] = self.data[5][to_trim:]  # Trims excess items from the sixth data element.  # Entfernt überschüssige Elemente aus dem sechsten Datenelement.
            self.data[6] = self.data[6][to_trim:]  # Trims excess items from the seventh data element.  # Entfernt überschüssige Elemente aus dem siebten Datenelement.
            self.data[7] = self.data[7][to_trim:]  # Trims excess items from the eighth data element.  # Entfernt überschüssige Elemente aus dem achten Datenelement.
            self.data[8] = self.data[8][to_trim:]  # Trims excess items from the ninth data element.  # Entfernt überschüssige Elemente aus dem neunten Datenelement.
            self.data[9] = self.data[9][to_trim:]  # Trims excess items from the tenth data element.  # Entfernt überschüssige Elemente aus dem zehnten Datenelement.
            self.data[10] = self.data[10][to_trim:]  # Trims excess items from the eleventh data element.  # Entfernt überschüssige Elemente aus dem elften Datenelement.

        return self  # Returns the updated instance of MemoryTMFull.  # Gibt die aktualisierte Instanz von MemoryTMFull zurück.
