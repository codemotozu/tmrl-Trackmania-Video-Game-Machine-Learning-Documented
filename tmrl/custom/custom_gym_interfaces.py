# rtgym interfaces for Trackmania  # rtgym-Schnittstellen für Trackmania

# standard library imports  # Standardbibliothek-Importe
import logging  # Module for logging messages.  # Modul zum Protokollieren von Nachrichten.
import time  # Module for time-related operations.  # Modul für zeitbezogene Operationen.
from collections import deque  # Module for double-ended queue data structure.  # Modul für die doppelt verkettete Warteschlange.

# third-party imports  # Drittanbieter-Importe
import cv2  # OpenCV library for computer vision tasks.  # OpenCV-Bibliothek für Computer-Vision-Aufgaben.
import gymnasium.spaces as spaces  # Gymnasium library for RL spaces.  # Gymnasium-Bibliothek für RL-Räume.
import numpy as np  # NumPy library for numerical computations.  # NumPy-Bibliothek für numerische Berechnungen.

# third-party imports  # Drittanbieter-Importe
from rtgym import RealTimeGymInterface  # Interface for real-time Gym environments.  # Schnittstelle für Echtzeit-Gym-Umgebungen.

# local imports  # Lokale Importe
import tmrl.config.config_constants as cfg  # Configuration constants for TMRL.  # Konfigurationskonstanten für TMRL.
from tmrl.custom.utils.compute_reward import RewardFunction  # Class to calculate rewards.  # Klasse zum Berechnen von Belohnungen.
from tmrl.custom.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20  # Functions for gamepad controls.  # Funktionen für Gamepad-Steuerungen.
from tmrl.custom.utils.control_mouse import mouse_close_finish_pop_up_tm20  # Function to close pop-ups with a mouse.  # Funktion, um Pop-ups mit der Maus zu schließen.
from tmrl.custom.utils.control_keyboard import apply_control, keyres  # Functions for keyboard controls.  # Funktionen für Tastatursteuerungen.
from tmrl.custom.utils.window import WindowInterface  # Interface to interact with the game window.  # Schnittstelle zur Interaktion mit dem Spiel-Fenster.
from tmrl.custom.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost  # Utilities for Lidar, game client, and saving ghost replays.  # Hilfsprogramme für Lidar, Spiel-Client und Ghost-Wiederholungen.

# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts  # Erlaubt (und belohnt) Abkürzungen von 50 Metern.

# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):  # Interface class for TrackMania 2020 environment.  # Schnittstellenklasse für die TrackMania-2020-Umgebung.
    """
    This is the API needed for the algorithm to control TrackMania 2020  # API zur Steuerung von TrackMania 2020.
    """
    def __init__(self,
                 img_hist_len: int = 4,  # Number of images in the observation history.  # Anzahl der Bilder im Beobachtungsverlauf.
                 gamepad: bool = True,  # Whether to use a virtual gamepad for controls.  # Ob ein virtuelles Gamepad verwendet wird.
                 save_replays: bool = False,  # Whether to save replays of episodes.  # Ob Wiederholungen von Episoden gespeichert werden.
                 grayscale: bool = True,  # Whether images should be grayscale or color.  # Ob Bilder in Graustufen oder Farbe sein sollen.
                 resize_to=(64, 64)):  # Resize output images to these dimensions.  # Ausgabe-Bilder auf diese Abmessungen skalieren.
        """
        Base rtgym interface for TrackMania 2020 (Full environment)  # Grundlegende rtgym-Schnittstelle für TrackMania 2020 (volle Umgebung).

        Args:  # Argumente:
            img_hist_len: int: history of images that are part of observations  # Verlauf der Bilder, die Teil der Beobachtungen sind.
            gamepad: bool: whether to use a virtual gamepad for control  # Ob ein virtuelles Gamepad verwendet wird.
            save_replays: bool: whether to save TrackMania replays on successful episodes  # Ob TrackMania-Wiederholungen bei erfolgreichen Episoden gespeichert werden.
            grayscale: bool: whether to output grayscale images or color images  # Ob Graustufenbilder oder Farbbilder ausgegeben werden sollen.
            resize_to: Tuple[int, int]: resize output images to this (width, height)  # Ausgabe-Bilder auf (Breite, Höhe) skalieren.
        """
        self.last_time = None  # Last time an action was taken.  # Zeitstempel der letzten Aktion.
        self.img_hist_len = img_hist_len  # Number of images in observation history.  # Anzahl der Bilder im Beobachtungsverlauf.
        self.img_hist = None  # Buffer for storing image history.  # Puffer zur Speicherung des Bildverlaufs.
        self.img = None  # Current image frame.  # Aktuelles Bild-Frame.
        self.reward_function = None  # Instance of the reward function.  # Instanz der Belohnungsfunktion.
        self.client = None  # Client for interacting with the game.  # Client zur Interaktion mit dem Spiel.
        self.gamepad = gamepad  # Gamepad control enabled or disabled.  # Gamepad-Steuerung aktiviert oder deaktiviert.
        self.j = None  # Joystick instance for gamepad control.  # Joystick-Instanz für die Gamepad-Steuerung.
        self.window_interface = None  # Interface for the game window.  # Schnittstelle für das Spiel-Fenster.
        self.small_window = None  # Boolean for reduced window size.  # Boolean für verkleinertes Fenster.
        self.save_replays = save_replays  # Save replays flag.  # Flag zum Speichern von Wiederholungen.
        self.grayscale = grayscale  # Output grayscale images flag.  # Flag für Graustufen-Ausgabe.
        self.resize_to = resize_to  # Resize image dimensions.  # Abmessungen für das Skalieren von Bildern.
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']  # Reward for finishing the track.  # Belohnung für das Beenden der Strecke.
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']  # Constant penalty during gameplay.  # Konstante Strafe während des Spiels.

        self.initialized = False  # Initialization flag.  # Initialisierungs-Flag.

    def initialize_common(self):  # Common initialization steps.  # Gemeinsame Initialisierungsschritte.
        if self.gamepad:  # Check if gamepad is enabled.  # Prüfen, ob das Gamepad aktiviert ist.
            import vgamepad as vg  # Import vgamepad module for virtual joystick.  # Modul "vgamepad" für virtuellen Joystick importieren.
            self.j = vg.VX360Gamepad()  # Create a virtual gamepad instance.  # Virtuelle Gamepad-Instanz erstellen.
            logging.debug(" virtual joystick in use")  # Log joystick usage.  # Joystick-Nutzung protokollieren.
        self.window_interface = WindowInterface("Trackmania")  # Create window interface for Trackmania.  # Fenster-Schnittstelle für Trackmania erstellen.
        self.window_interface.move_and_resize()  # Resize the game window.  # Spiel-Fenster skalieren.
        self.last_time = time.time()  # Store the current time.  # Aktuelle Zeit speichern.
        self.img_hist = deque(maxlen=self.img_hist_len)  # Initialize image history buffer.  # Bildverlaufs-Puffer initialisieren.
        self.img = None  # Reset the current image.  # Aktuelles Bild zurücksetzen.
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,  # Initialize the reward function.  # Belohnungsfunktion initialisieren.
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],  # Number of forward observations.  # Anzahl der Vorwärtsbeobachtungen.
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],  # Number of backward observations.  # Anzahl der Rückwärtsbeobachtungen.
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],  # Time until failure is registered.  # Zeit bis ein Fehlschlag registriert wird.
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],  # Minimum steps before failure.  # Mindestanzahl von Schritten vor einem Fehlschlag.
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])  # Maximum distance from trajectory.  # Maximale Abweichung von der Trajektorie.
        self.client = TM2020OpenPlanetClient()  # Initialize client for TrackMania.  # Client für TrackMania initialisieren.

    def initialize(self):  # Full initialization.  # Vollständige Initialisierung.
        self.initialize_common()  # Call common initialization steps.  # Gemeinsame Initialisierungsschritte aufrufen.
        self.small_window = True  # Set small window flag to True.  # Flag für kleines Fenster auf True setzen.
        self.initialized = True  # Mark as initialized.  # Als initialisiert markieren.

    def send_control(self, control):  # Apply controls to the game.  # Steuerungen auf das Spiel anwenden.
        """
        Non-blocking function  # Nicht-blockierende Funktion.
        Applies the action given by the RL policy  # Wendet die durch die RL-Politik gegebene Aktion an.
        If control is None, does nothing (e.g. to record)  # Macht nichts, wenn "control" None ist (z. B. zum Aufzeichnen).
        Args:  # Argumente:
            control: np.array: [forward,backward,right,left]  # Array von Steuerungen: [vorwärts, rückwärts, rechts, links].
        """
        if self.gamepad:  # Check if using gamepad.  # Prüfen, ob das Gamepad verwendet wird.
            if control is not None:  # Apply control if valid.  # Steuerung anwenden, wenn gültig.
                control_gamepad(self.j, control)  # Use gamepad control function.  # Gamepad-Steuerungsfunktion verwenden.
        else:  # If not using gamepad.  # Falls kein Gamepad verwendet wird.
            if control is not None:  # Apply control if valid.  # Steuerung anwenden, wenn gültig.
                actions = []  # Initialize actions list.  # Aktionsliste initialisieren.
                if control[0] > 0:  # Forward control.  # Vorwärtssteuerung.
                    actions.append('f')  # Add forward action.  # Vorwärtsaktion hinzufügen.
                if control[1] > 0:  # Backward control.  # Rückwärtssteuerung.
                    actions.append('b')  # Add backward action.  # Rückwärtsaktion hinzufügen.
                if control[2] > 0.5:  # Turn right control.  # Rechtssteuerung.
                    actions.append('r')  # Add right turn action.  # Aktion für Rechtskurve hinzufügen.
                elif control[2] < -0.5:  # Turn left control.  # Linkssteuerung.
                    actions.append('l')  # Add left turn action.  # Aktion für Linkskurve hinzufügen.
                apply_control(actions)  # Apply the control actions.  # Steuerungsaktionen anwenden.


def grab_data_and_img(self):
    img = self.window_interface.screenshot()[:, :, :3]  # Capture a screenshot, keep only BGR channels (blue, green, red).  # Screenshot machen, nur BGR-Kanäle beibehalten (Blau, Grün, Rot).
    if self.resize_to is not None:  # If resizing is specified, resize the image.  # Wenn Größenänderung angegeben, Bild anpassen.
        img = cv2.resize(img, self.resize_to)
    if self.grayscale:  # If grayscale is enabled, convert the image to grayscale.  # Wenn Graustufen aktiviert, Bild in Graustufen umwandeln.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # If not grayscale, convert to RGB by reversing BGR channels.  # Wenn keine Graustufen, BGR-Kanäle umkehren für RGB.
        img = img[:, :, ::-1]
    data = self.client.retrieve_data()  # Retrieve data from the client.  # Daten vom Client abrufen.
    self.img = img  # Save image for rendering.  # Bild für Rendering speichern.
    return data, img  # Return data and image.  # Daten und Bild zurückgeben.

def reset_race(self):
    if self.gamepad:  # Reset race using gamepad if available.  # Rennen zurücksetzen mit Gamepad, falls vorhanden.
        gamepad_reset(self.j)
    else:  # Otherwise, reset using keyboard.  # Andernfalls mit Tastatur zurücksetzen.
        keyres()

def reset_common(self):
    if not self.initialized:  # Initialize if not done yet.  # Initialisieren, falls noch nicht geschehen.
        self.initialize()
    self.send_control(self.get_default_action())  # Send default control action.  # Standardsteuerungsaktion senden.
    self.reset_race()  # Reset the race.  # Rennen zurücksetzen.
    time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET  # Adjust sleep time.  # Schlafzeit anpassen.
    time.sleep(time_sleep)  # Wait for image refresh.  # Warten, bis Bild aktualisiert ist.

def reset(self, seed=None, options=None):
    """
    Reset the environment and return initial observations.
    """
    self.reset_common()  # Perform common reset tasks.  # Allgemeine Reset-Aufgaben ausführen.
    data, img = self.grab_data_and_img()  # Capture data and image.  # Daten und Bild erfassen.
    speed = np.array([data[0]], dtype='float32')  # Extract speed as a numpy array.  # Geschwindigkeit als Numpy-Array extrahieren.
    gear = np.array([data[9]], dtype='float32')  # Extract gear as a numpy array.  # Gang als Numpy-Array extrahieren.
    rpm = np.array([data[10]], dtype='float32')  # Extract RPM as a numpy array.  # Drehzahl als Numpy-Array extrahieren.
    for _ in range(self.img_hist_len):  # Fill image history with the captured image.  # Bildhistorie mit aufgenommenem Bild füllen.
        self.img_hist.append(img)
    imgs = np.array(list(self.img_hist))  # Convert image history to numpy array.  # Bildhistorie in Numpy-Array umwandeln.
    obs = [speed, gear, rpm, imgs]  # Create observation array.  # Beobachtungsarray erstellen.
    self.reward_function.reset()  # Reset the reward function.  # Belohnungsfunktion zurücksetzen.
    return obs, {}  # Return observations and empty info.  # Beobachtungen und leere Info zurückgeben.

def close_finish_pop_up_tm20(self):
    if self.gamepad:  # Close pop-up using gamepad if available.  # Pop-up mit Gamepad schließen, falls vorhanden.
        gamepad_close_finish_pop_up_tm20(self.j)
    else:  # Otherwise, close with mouse.  # Andernfalls mit Maus schließen.
        mouse_close_finish_pop_up_tm20(small_window=self.small_window)

def wait(self):
    """
    Pause the agent and wait in position.
    """
    self.send_control(self.get_default_action())  # Send default control action.  # Standardsteuerungsaktion senden.
    if self.save_replays:  # Save replays if enabled.  # Replays speichern, falls aktiviert.
        save_ghost()
        time.sleep(1.0)
    self.reset_race()  # Reset the race.  # Rennen zurücksetzen.
    time.sleep(0.5)  # Wait briefly.  # Kurz warten.
    self.close_finish_pop_up_tm20()  # Close the finish pop-up.  # Ziel-Pop-up schließen.

def get_obs_rew_terminated_info(self):
    """
    Return observations, reward, and termination info.
    """
    data, img = self.grab_data_and_img()  # Capture data and image.  # Daten und Bild erfassen.
    speed = np.array([data[0]], dtype='float32')  # Extract speed.  # Geschwindigkeit extrahieren.
    gear = np.array([data[9]], dtype='float32')  # Extract gear.  # Gang extrahieren.
    rpm = np.array([data[10]], dtype='float32')  # Extract RPM.  # Drehzahl extrahieren.
    rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))  # Compute reward.  # Belohnung berechnen.
    self.img_hist.append(img)  # Update image history.  # Bildhistorie aktualisieren.
    imgs = np.array(list(self.img_hist))  # Convert image history to array.  # Bildhistorie in Array umwandeln.
    obs = [speed, gear, rpm, imgs]  # Create observation.  # Beobachtung erstellen.
    end_of_track = bool(data[8])  # Check if track ended.  # Prüfen, ob Strecke zu Ende ist.
    info = {}  # Initialize info dictionary.  # Info-Dictionary initialisieren.
    if end_of_track:  # If track ended, terminate and adjust reward.  # Wenn Strecke zu Ende, Terminierung und Belohnung anpassen.
        terminated = True
        rew += self.finish_reward
    rew += self.constant_penalty  # Add constant penalty to reward.  # Konstante Strafe zur Belohnung hinzufügen.
    rew = np.float32(rew)  # Convert reward to float32.  # Belohnung in float32 umwandeln.
    return obs, rew, terminated, info  # Return results.  # Ergebnisse zurückgeben.

def get_observation_space(self):
    """
    Define the observation space.
    """
    speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))  # Define speed range.  # Geschwindigkeitsbereich definieren.
    gear = spaces.Box(low=0.0, high=6, shape=(1, ))  # Define gear range.  # Gangbereich definieren.
    rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))  # Define RPM range.  # Drehzahlbereich definieren.
    if self.resize_to is not None:  # Check if resizing is specified.  # Prüfen, ob Größenanpassung angegeben ist.
        w, h = self.resize_to
    else:  # Use default dimensions.  # Standardmaße verwenden.
        w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
    if self.grayscale:  # Define grayscale image space.  # Graustufen-Bildbereich definieren.
        img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
    else:  # Define RGB image space.  # RGB-Bildbereich definieren.
        img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
    return spaces.Tuple((speed, gear, rpm, img))  # Return observation space.  # Beobachtungsbereich zurückgeben.

def get_action_space(self):
    """
    Define the action space.
    """
    return spaces.Box(low=-1.0, high=1.0, shape=(3, ))  # Define action range.  # Aktionsbereich definieren.

def get_default_action(self):
    """
    Return the default action.
    """
    return np.array([0.0, 0.0, 0.0], dtype='float32')  # Default action is zero for all controls.  # Standardaktion ist Null für alle Steuerungen.


class TM2020InterfaceLidar(TM2020Interface):  # Defines a new class TM2020InterfaceLidar that inherits from TM2020Interface.  # Definiert eine neue Klasse TM2020InterfaceLidar, die von TM2020Interface erbt.
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):  # Initializes the class with default parameters for image history length, gamepad, and replay saving.  # Initialisiert die Klasse mit Standardparametern für die Bildverlaufslänge, Gamepad und das Speichern von Wiederholungen.
        super().__init__(img_hist_len, gamepad, save_replays)  # Calls the initializer of the parent class TM2020Interface.  # Ruft den Initialisierer der Elternklasse TM2020Interface auf.
        self.window_interface = None  # Initializes the window_interface attribute to None.  # Initialisiert das Attribut window_interface auf None.
        self.lidar = None  # Initializes the lidar attribute to None.  # Initialisiert das Attribut lidar auf None.

    def grab_lidar_speed_and_data(self):  # Defines a method to grab lidar data, speed, and other data.  # Definiert eine Methode zum Abrufen von Lidar-Daten, Geschwindigkeit und anderen Daten.
        img = self.window_interface.screenshot()[:, :, :3]  # Takes a screenshot from the window_interface and keeps the first three color channels (RGB).  # Macht einen Screenshot von window_interface und behält die ersten drei Farbkanäle (RGB).
        data = self.client.retrieve_data()  # Retrieves data from the client.  # Ruft Daten vom Client ab.
        speed = np.array([  # Creates a numpy array for speed using the first element of data.  # Erstellt ein numpy-Array für die Geschwindigkeit unter Verwendung des ersten Elements der Daten.
            data[0],  
        ], dtype='float32')  # Sets the data type to float32.  # Setzt den Datentyp auf float32.
        lidar = self.lidar.lidar_20(img=img, show=False)  # Calls the lidar_20 method of the lidar object to get lidar data.  # Ruft die Methode lidar_20 des lidar-Objekts auf, um Lidar-Daten zu erhalten.
        return lidar, speed, data  # Returns the lidar data, speed, and other data.  # Gibt die Lidar-Daten, Geschwindigkeit und andere Daten zurück.

    def initialize(self):  # Defines the initialize method.  # Definiert die Methode initialize.
        super().initialize_common()  # Calls the common initialize method from the parent class.  # Ruft die allgemeine Initialisierungsmethode der Elternklasse auf.
        self.small_window = False  # Sets the small_window attribute to False.  # Setzt das Attribut small_window auf False.
        self.lidar = Lidar(self.window_interface.screenshot())  # Initializes the lidar object with a screenshot.  # Initialisiert das Lidar-Objekt mit einem Screenshot.
        self.initialized = True  # Marks the object as initialized.  # Markiert das Objekt als initialisiert.

    def reset(self, seed=None, options=None):  # Defines the reset method.  # Definiert die Methode reset.
        """
        obs must be a list of numpy arrays  # Documentation that observations must be a list of numpy arrays.  # Dokumentation, dass Beobachtungen eine Liste von numpy-Arrays sein müssen.
        """
        self.reset_common()  # Calls the common reset method from the parent class.  # Ruft die allgemeine Rücksetz-Methode der Elternklasse auf.
        img, speed, data = self.grab_lidar_speed_and_data()  # Grabs lidar data, speed, and other data.  # Ruft Lidar-Daten, Geschwindigkeit und andere Daten ab.
        for _ in range(self.img_hist_len):  # Loops through the image history length.  # Schleift durch die Bildverlaufslänge.
            self.img_hist.append(img)  # Appends the image to the image history.  # Fügt das Bild dem Bildverlauf hinzu.
        imgs = np.array(list(self.img_hist), dtype='float32')  # Converts the image history to a numpy array.  # Konvertiert den Bildverlauf in ein numpy-Array.
        obs = [speed, imgs]  # Creates an observation consisting of speed and images.  # Erstellt eine Beobachtung bestehend aus Geschwindigkeit und Bildern.
        self.reward_function.reset()  # Resets the reward function.  # Setzt die Belohnungsfunktion zurück.
        return obs, {}  # Returns the observation and an empty dictionary.  # Gibt die Beobachtung und ein leeres Dictionary zurück.

    def get_obs_rew_terminated_info(self):  # Defines a method to get observation, reward, and termination info.  # Definiert eine Methode, um Beobachtungen, Belohnung und Abschlussinformationen zu erhalten.
        """
        returns the observation, the reward, and a terminated signal for end of episode  # Documentation describing the method's return values.  # Dokumentation, die die Rückgabewerte der Methode beschreibt.
        obs must be a list of numpy arrays  # Documentation specifying that the observation must be a list of numpy arrays.  # Dokumentation, dass die Beobachtung eine Liste von numpy-Arrays sein muss.
        """
        img, speed, data = self.grab_lidar_speed_and_data()  # Grabs lidar data, speed, and other data.  # Ruft Lidar-Daten, Geschwindigkeit und andere Daten ab.
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))  # Computes the reward and termination status.  # Berechnet die Belohnung und den Abschlussstatus.
        self.img_hist.append(img)  # Appends the image to the image history.  # Fügt das Bild dem Bildverlauf hinzu.
        imgs = np.array(list(self.img_hist), dtype='float32')  # Converts the image history to a numpy array.  # Konvertiert den Bildverlauf in ein numpy-Array.
        obs = [speed, imgs]  # Creates an observation consisting of speed and images.  # Erstellt eine Beobachtung bestehend aus Geschwindigkeit und Bildern.
        end_of_track = bool(data[8])  # Checks if it's the end of the track.  # Überprüft, ob es das Ende der Strecke ist.
        info = {}  # Initializes an empty dictionary for additional information.  # Initialisiert ein leeres Dictionary für zusätzliche Informationen.
        if end_of_track:  # Checks if the end of the track is reached.  # Überprüft, ob das Ende der Strecke erreicht ist.
            rew += self.finish_reward  # Adds finish reward if it's the end of the track.  # Fügt eine Abschlussbelohnung hinzu, wenn es das Ende der Strecke ist.
            terminated = True  # Sets terminated to True if the track ends.  # Setzt terminated auf True, wenn die Strecke endet.
        rew += self.constant_penalty  # Adds a constant penalty to the reward.  # Fügt eine konstante Strafe zur Belohnung hinzu.
        rew = np.float32(rew)  # Converts the reward to a float32 type.  # Konvertiert die Belohnung in den Datentyp float32.
        return obs, rew, terminated, info  # Returns the observation, reward, termination status, and additional info.  # Gibt die Beobachtung, Belohnung, Abschlussstatus und zusätzliche Informationen zurück.

    def get_observation_space(self):  # Defines a method to get the observation space.  # Definiert eine Methode, um den Beobachtungsraum zu erhalten.
        """
        must be a Tuple  # Documentation that the observation space must be a tuple.  # Dokumentation, dass der Beobachtungsraum ein Tuple sein muss.
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))  # Defines a box space for speed between 0 and 1000.  # Definiert einen Boxraum für die Geschwindigkeit zwischen 0 und 1000.
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(  # Defines a box space for images with an infinite upper limit.
            self.img_hist_len,  # The number of images in history.
            19,  # The number of features in each image (e.g., lidar scan points).
        ))  # End of the images box space definition.
        return spaces.Tuple((speed, imgs))  # Returns a tuple containing the speed and images space.  # Gibt ein Tuple zurück, das den Geschwindigkeits- und Bildraum enthält.

class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):  # Defining a new class that inherits from TM2020InterfaceLidar.  # Definieren einer neuen Klasse, die von TM2020InterfaceLidar erbt.

    def reset(self, seed=None, options=None):  # Method to reset the environment, taking optional seed and options.  # Methode zum Zurücksetzen der Umgebung, mit optionalem "seed" und "options".
        """
        obs must be a list of numpy arrays  # Comment explaining that 'obs' must be a list of numpy arrays.  # Kommentar, der erklärt, dass 'obs' eine Liste von numpy-Arrays sein muss.
        """
        self.reset_common()  # Calls a common reset function defined in the parent class.  # Ruft eine allgemeine Zurücksetzfunktion der übergeordneten Klasse auf.
        img, speed, data = self.grab_lidar_speed_and_data()  # Grabs lidar image, speed, and data.  # Ruft Lidar-Bild, Geschwindigkeit und Daten ab.
        for _ in range(self.img_hist_len):  # Loops through the image history length.  # Schleife durch die Länge der Bildhistorie.
            self.img_hist.append(img)  # Appends the grabbed image to the history.  # Fügt das abgerufene Bild der Historie hinzu.
        imgs = np.array(list(self.img_hist), dtype='float32')  # Converts the image history to a numpy array.  # Wandelt die Bildhistorie in ein numpy-Array um.
        progress = np.array([0], dtype='float32')  # Creates an array with a progress value of 0.  # Erstellt ein Array mit einem Fortschrittswert von 0.
        obs = [speed, progress, imgs]  # Creates the observation as a list of speed, progress, and images.  # Erstellt die Beobachtung als Liste von Geschwindigkeit, Fortschritt und Bildern.
        self.reward_function.reset()  # Resets the reward function.  # Setzt die Belohnungsfunktion zurück.
        return obs, {}  # Returns the observation and an empty dictionary.  # Gibt die Beobachtung und ein leeres Dictionary zurück.

    def get_obs_rew_terminated_info(self):  # Method to return observation, reward, and termination info.  # Methode, die Beobachtung, Belohnung und Beendigungsinformationen zurückgibt.
        """
        returns the observation, the reward, and a terminated signal for end of episode  # Comment explaining the purpose of the method.  # Kommentar, der den Zweck der Methode erklärt.
        obs must be a list of numpy arrays  # Comment explaining that 'obs' must be a list of numpy arrays.  # Kommentar, der erklärt, dass 'obs' eine Liste von numpy-Arrays sein muss.
        """
        img, speed, data = self.grab_lidar_speed_and_data()  # Grabs lidar image, speed, and data.  # Ruft Lidar-Bild, Geschwindigkeit und Daten ab.
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))  # Computes reward and termination based on data.  # Berechnet Belohnung und Beendigung basierend auf den Daten.
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')  # Calculates progress as a ratio of current index to data length.  # Berechnet den Fortschritt als Verhältnis des aktuellen Index zur Datenlänge.
        self.img_hist.append(img)  # Appends the new image to the history.  # Fügt das neue Bild der Historie hinzu.
        imgs = np.array(list(self.img_hist), dtype='float32')  # Converts the image history to a numpy array.  # Wandelt die Bildhistorie in ein numpy-Array um.
        obs = [speed, progress, imgs]  # Creates the observation as a list of speed, progress, and images.  # Erstellt die Beobachtung als Liste von Geschwindigkeit, Fortschritt und Bildern.
        end_of_track = bool(data[8])  # Checks if the end of the track is reached.  # Überprüft, ob das Ende der Strecke erreicht wurde.
        info = {}  # Initializes an empty dictionary for additional information.  # Initialisiert ein leeres Dictionary für zusätzliche Informationen.
        if end_of_track:  # If end of track is reached.  # Wenn das Ende der Strecke erreicht ist.
            rew += self.finish_reward  # Adds the finish reward if the track is completed.  # Fügt die Abschlussbelohnung hinzu, wenn die Strecke abgeschlossen ist.
            terminated = True  # Marks the episode as terminated.  # Markiert die Episode als beendet.
        rew += self.constant_penalty  # Applies a constant penalty to the reward.  # Wendet eine konstante Strafe auf die Belohnung an.
        rew = np.float32(rew)  # Converts the reward to a float32 type.  # Wandelt die Belohnung in den Typ float32 um.
        return obs, rew, terminated, info  # Returns the observation, reward, termination signal, and additional info.  # Gibt die Beobachtung, Belohnung, Beendigungsignal und zusätzliche Informationen zurück.

    def get_observation_space(self):  # Method to define the observation space.  # Methode zur Definition des Beobachtungsraums.
        """
        must be a Tuple  # Comment stating the return type should be a tuple.  # Kommentar, der angibt, dass der Rückgabewert ein Tuple sein muss.
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))  # Defines the speed observation space as a box between 0 and 1000.  # Definiert den Beobachtungsraum der Geschwindigkeit als Box zwischen 0 und 1000.
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))  # Defines the progress observation space as a box between 0 and 1.  # Definiert den Beobachtungsraum des Fortschritts als Box zwischen 0 und 1.
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(  # Defines the image observation space as a box with a specific shape.
            self.img_hist_len,  # The number of images in the history.  # Die Anzahl der Bilder in der Historie.
            19,  # The number of lidar measurements in each image.  # Die Anzahl der Lidar-Messungen in jedem Bild.
        ))  # lidars  # Comment describing the lidar observation space.  # Kommentar, der den Lidar-Beobachtungsraum beschreibt.
        return spaces.Tuple((speed, progress, imgs))  # Returns the observation space as a tuple of speed, progress, and images.  # Gibt den Beobachtungsraum als Tuple von Geschwindigkeit, Fortschritt und Bildern zurück.


if __name__ == "__main__":  # This checks if the script is run directly (not imported).  # Überprüft, ob das Skript direkt (nicht importiert) ausgeführt wird.
    pass  # Placeholder indicating no action is taken if this script is run directly.  # Platzhalter, der anzeigt, dass keine Aktion durchgeführt wird, wenn dieses Skript direkt ausgeführt wird.

