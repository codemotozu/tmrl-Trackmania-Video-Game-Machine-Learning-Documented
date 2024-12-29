# standard library imports
import os  # Used for file and directory handling.  # Wird für Datei- und Verzeichnisverwaltung verwendet.
import pickle  # Used to serialize and deserialize Python objects.  # Wird zum Serialisieren und Deserialisieren von Python-Objekten verwendet.

# third-party imports
import numpy as np  # Provides support for numerical operations and arrays.  # Bietet Unterstützung für numerische Operationen und Arrays.
import logging  # Used for logging messages for debugging or other purposes.  # Wird zum Protokollieren von Nachrichten für Debugging oder andere Zwecke verwendet.

class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.  
    Berechnet eine Belohnung aus der Openplanet-API für Trackmania 2020.
    """
    def __init__(self,
                 reward_data_path,  # Path to the trajectory file.  # Pfad zur Trajektoriendatei.
                 nb_obs_forward=10,  # Max distance for forward cuts in trajectory.  # Maximale Entfernung für Vorwärtsabschnitte in der Trajektorie.
                 nb_obs_backward=10,  # Max distance for backward cuts in trajectory.  # Maximale Entfernung für Rückwärtsabschnitte in der Trajektorie.
                 nb_zero_rew_before_failure=10,  # Steps with zero reward before termination.  # Schritte mit null Belohnung vor Beendigung.
                 min_nb_steps_before_failure=int(3.5 * 20),  # Minimum steps before failure allowed.  # Minimale Anzahl an Schritten vor erlaubtem Fehler.
                 max_dist_from_traj=60.0):  # Max distance from trajectory for reward.  # Maximale Entfernung von der Trajektorie für Belohnung.
        """
        Initializes the reward function for Trackmania 2020.  
        Initialisiert die Belohnungsfunktion für Trackmania 2020.
        """
        if not os.path.exists(reward_data_path):  # Checks if reward data exists.  # Überprüft, ob Belohnungsdaten existieren.
            logging.debug(f" reward not found at path:{reward_data_path}")  # Logs message if file not found.  # Protokolliert Nachricht, falls Datei nicht gefunden wird.
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # Sets dummy reward data.  # Setzt Dummy-Belohnungsdaten.
        else:  
            with open(reward_data_path, 'rb') as f:  # Opens trajectory file in read-binary mode.  # Öffnet Trajektoriendatei im Lese-Binärmodus.
                self.data = pickle.load(f)  # Loads trajectory data.  # Lädt Trajektoriendaten.

        self.cur_idx = 0  # Current index in the trajectory.  # Aktueller Index in der Trajektorie.
        self.nb_obs_forward = nb_obs_forward  # Max positions forward for cuts.  # Maximale Positionen für Vorwärtsabschnitte.
        self.nb_obs_backward = nb_obs_backward  # Max positions backward for cuts.  # Maximale Positionen für Rückwärtsabschnitte.
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure  # Zero-reward steps before failure.  # Null-Belohnungs-Schritte vor Fehler.
        self.min_nb_steps_before_failure = min_nb_steps_before_failure  # Minimum steps allowed before failure.  # Minimale Schritte vor Fehler erlaubt.
        self.max_dist_from_traj = max_dist_from_traj  # Max allowed distance from trajectory.  # Maximale erlaubte Entfernung von der Trajektorie.
        self.step_counter = 0  # Counts steps in the episode.  # Zählt Schritte in der Episode.
        self.failure_counter = 0  # Counts failures.  # Zählt Fehler.
        self.datalen = len(self.data)  # Length of trajectory data.  # Länge der Trajektoriendaten.

    def compute_reward(self, pos):
        """
        Computes the reward for the current position.
        Berechnet die Belohnung für die aktuelle Position.
        """
        terminated = False  # Tracks if the episode is terminated.  # Verfolgt, ob die Episode beendet ist.
        self.step_counter += 1  # Increments step counter.  # Erhöht den Schrittzähler.
        min_dist = np.inf  # Initializes minimum distance.  # Initialisiert minimale Entfernung.
        index = self.cur_idx  # Starts search from current index.  # Beginnt Suche ab aktuellem Index.
        temp = self.nb_obs_forward  # Counter for forward cuts.  # Zähler für Vorwärtsabschnitte.
        best_index = 0  # Stores best matching index.  # Speichert den am besten passenden Index.

        while True:
            dist = np.linalg.norm(pos - self.data[index])  # Computes distance to trajectory point.  # Berechnet Entfernung zum Trajektorienpunkt.
            if dist <= min_dist:  # Updates best distance and index if closer point found.  # Aktualisiert beste Entfernung und Index bei näherem Punkt.
                min_dist = dist  
                best_index = index  
                temp = self.nb_obs_forward  
            index += 1  # Moves to the next point.  # Geht zum nächsten Punkt.
            temp -= 1  # Decrements forward counter.  # Verringert Vorwärtszähler.

            if index >= self.datalen or temp <= 0:  # Checks stop conditions.  # Überprüft Stopp-Bedingungen.
                if min_dist > self.max_dist_from_traj:  # If too far, resets to current index.  # Wenn zu weit entfernt, auf aktuellen Index zurücksetzen.
                    best_index = self.cur_idx
                break

        reward = (best_index - self.cur_idx) / 100.0  # Computes reward as progress along trajectory.  # Berechnet Belohnung als Fortschritt entlang der Trajektorie.

        if best_index == self.cur_idx:  # If no progress, rewind.  # Bei keinem Fortschritt, zurückspulen.
            min_dist = np.inf  
            index = self.cur_idx  

            while True:
                dist = np.linalg.norm(pos - self.data[index])  # Checks distances in reverse.  # Überprüft Entfernungen rückwärts.
                if dist <= min_dist:
                    min_dist = dist  
                    best_index = index  
                    temp = self.nb_obs_backward  
                index -= 1  
                temp -= 1  

                if index <= 0 or temp <= 0:  # Stop conditions for reverse search.  # Stopp-Bedingungen für Rückwärtssuche.
                    break

            if self.step_counter > self.min_nb_steps_before_failure:  # Failure if too many steps.  # Fehler bei zu vielen Schritten.
                self.failure_counter += 1  
                if self.failure_counter > self.nb_zero_rew_before_failure:  # Ends episode after repeated failure.  # Beendet Episode nach wiederholtem Fehler.
                    terminated = True

        else:  
            self.failure_counter = 0  # Resets failure counter on progress.  # Setzt Fehlerzähler bei Fortschritt zurück.

        self.cur_idx = best_index  # Updates current index to best match.  # Aktualisiert aktuellen Index auf beste Übereinstimmung.

        return reward, terminated  # Returns computed reward and termination status.  # Gibt berechnete Belohnung und Endstatus zurück.

    def reset(self):
        """
        Resets the reward function for a new episode.
        Setzt die Belohnungsfunktion für eine neue Episode zurück.
        """
        self.cur_idx = 0  # Resets current index.  # Setzt aktuellen Index zurück.
        self.step_counter = 0  # Resets step counter.  # Setzt Schrittzähler zurück.
        self.failure_counter = 0  # Resets failure counter.  # Setzt Fehlerzähler zurück.
