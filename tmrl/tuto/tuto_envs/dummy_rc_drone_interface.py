"""
RealTimeGym interface used for the TMRL library tutorial.

This environment simulates a dummy RC drone evolving in a bounded 2D world.
It features random delays in control and observation capture.
"""
from threading import Thread  # Imports the Thread class from the threading module to create a new thread for rendering.  # Importiert die Thread-Klasse aus dem Modul threading, um einen neuen Thread für das Rendering zu erstellen.

import cv2  # Imports the OpenCV library for image processing and rendering.  # Importiert die OpenCV-Bibliothek für die Bildbearbeitung und das Rendering.

import numpy as np  # Imports the NumPy library for numerical operations.  # Importiert die NumPy-Bibliothek für numerische Operationen.

import gymnasium.spaces as spaces  # Imports the 'spaces' module from the 'gymnasium' library for defining action and observation spaces.  # Importiert das Modul 'spaces' aus der 'gymnasium'-Bibliothek zur Definition von Aktions- und Beobachtungsräumen.

from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone  # Imports relevant classes and configurations from the rtgym package.  # Importiert relevante Klassen und Konfigurationen aus dem rtgym-Paket.

class DummyRCDroneInterface(RealTimeGymInterface):  # Defines the DummyRCDroneInterface class inheriting from RealTimeGymInterface.  # Definiert die Klasse DummyRCDroneInterface, die von RealTimeGymInterface erbt.

    def __init__(self):  # Initializes the class.  # Initialisiert die Klasse.
        self.rc_drone = None  # Initializes the RC drone as None.  # Initialisiert die RC-Drohne als None.
        self.target = np.array([0.0, 0.0], dtype=np.float32)  # Initializes the target location as a NumPy array with [0.0, 0.0].  # Initialisiert die Zielposition als NumPy-Array mit [0.0, 0.0].
        self.initialized = False  # Sets the initialization flag to False.  # Setzt das Initialisierungs-Flag auf False.
        self.blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 255  # Creates a blank white image (500x500 pixels).  # Erstellt ein leeres weißes Bild (500x500 Pixel).
        self.rendering_thread = Thread(target=self._rendering_thread, args=(), kwargs={}, daemon=True)  # Creates a new thread to handle rendering.  # Erstellt einen neuen Thread, der sich um das Rendering kümmert.

    def _rendering_thread(self):  # Defines the thread function for rendering.  # Definiert die Thread-Funktion für das Rendering.
        from time import sleep  # Imports the sleep function for controlling time.  # Importiert die sleep-Funktion zur Steuerung der Zeit.
        while True:  # Starts an infinite loop for continuous rendering.  # Startet eine unendliche Schleife für kontinuierliches Rendering.
            sleep(0.1)  # Waits for 0.1 seconds between render updates.  # Wartet 0,1 Sekunden zwischen den Render-Aktualisierungen.
            self.render()  # Calls the render method to display the updated image.  # Ruft die Render-Methode auf, um das aktualisierte Bild anzuzeigen.

    def get_observation_space(self):  # Defines the observation space for the environment.  # Definiert den Beobachtungsraum für die Umgebung.
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # Defines the x-coordinate space for the drone.  # Definiert den x-Koordinatenraum für die Drohne.
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # Defines the y-coordinate space for the drone.  # Definiert den y-Koordinatenraum für die Drohne.
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))  # Defines the x-coordinate space for the target.  # Definiert den x-Koordinatenraum für das Ziel.
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))  # Defines the y-coordinate space for the target.  # Definiert den y-Koordinatenraum für das Ziel.
        return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))  # Returns the combined observation space as a tuple.  # Gibt den kombinierten Beobachtungsraum als Tupel zurück.

    def get_action_space(self):  # Defines the action space for controlling the drone.  # Definiert den Aktionsraum zur Steuerung der Drohne.
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))  # Defines a 2D action space for controlling the velocity of the drone.  # Definiert einen 2D-Aktionsraum zur Steuerung der Geschwindigkeit der Drohne.

    def get_default_action(self):  # Returns the default action for the drone (no movement).  # Gibt die Standardaktion für die Drohne zurück (keine Bewegung).
        return np.array([0.0, 0.0], dtype='float32')  # Default action is no velocity (0.0, 0.0).  # Standardaktion ist keine Geschwindigkeit (0.0, 0.0).

    def send_control(self, control):  # Sends control signals to the drone.  # Sendet Steuerbefehle an die Drohne.
        vel_x = control[0]  # Extracts the x velocity from the control array.  # Extrahiert die x-Geschwindigkeit aus dem Steuerbefehl.
        vel_y = control[1]  # Extracts the y velocity from the control array.  # Extrahiert die y-Geschwindigkeit aus dem Steuerbefehl.
        self.rc_drone.send_control(vel_x, vel_y)  # Sends the velocities to the drone for movement.  # Sendet die Geschwindigkeiten an die Drohne für die Bewegung.

    def reset(self, seed=None, options=None):  # Resets the environment and drone state.  # Setzt die Umgebung und den Drohnenstatus zurück.
        if not self.initialized:  # Checks if the environment is initialized.  # Überprüft, ob die Umgebung initialisiert ist.
            self.rc_drone = DummyRCDrone()  # Initializes the dummy RC drone.  # Initialisiert die Dummy-RC-Drohne.
            self.rendering_thread.start()  # Starts the rendering thread.  # Startet den Rendering-Thread.
            self.initialized = True  # Sets the initialized flag to True.  # Setzt das Initialisierungs-Flag auf True.
        pos_x, pos_y = self.rc_drone.get_observation()  # Gets the drone's current position.  # Holt die aktuelle Position der Drohne.
        self.target[0] = np.random.uniform(-0.5, 0.5)  # Sets a random x target within the range (-0.5, 0.5).  # Setzt ein zufälliges Ziel für x im Bereich (-0,5, 0,5).
        self.target[1] = np.random.uniform(-0.5, 0.5)  # Sets a random y target within the range (-0.5, 0.5).  # Setzt ein zufälliges Ziel für y im Bereich (-0,5, 0,5).
        return [np.array([pos_x], dtype='float32'),  # Returns the observation as a list of NumPy arrays.  # Gibt die Beobachtung als Liste von NumPy-Arrays zurück.
                np.array([pos_y], dtype='float32'),
                np.array([self.target[0]], dtype='float32'),
                np.array([self.target[1]], dtype='float32')], {}

    def get_obs_rew_terminated_info(self):  # Returns the observation, reward, termination status, and additional information.  # Gibt die Beobachtung, Belohnung, den Status der Beendigung und zusätzliche Informationen zurück.
        pos_x, pos_y = self.rc_drone.get_observation()  # Gets the drone's current position.  # Holt die aktuelle Position der Drohne.
        tar_x = self.target[0]  # Gets the x-coordinate of the target.  # Holt die x-Koordinate des Ziels.
        tar_y = self.target[1]  # Gets the y-coordinate of the target.  # Holt die y-Koordinate des Ziels.
        obs = [np.array([pos_x], dtype='float32'),  # Prepares the observation to return.  # Bereitet die Beobachtung zur Rückgabe vor.
               np.array([pos_y], dtype='float32'),
               np.array([tar_x], dtype='float32'),
               np.array([tar_y], dtype='float32')]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)  # Calculates the reward as the negative distance to the target.  # Berechnet die Belohnung als negative Entfernung zum Ziel.
        terminated = rew > -0.01  # Checks if the task is terminated (when the drone is close to the target).  # Überprüft, ob die Aufgabe beendet ist (wenn die Drohne dem Ziel nahe ist).
        info = {}  # Empty dictionary for additional info.  # Leeres Wörterbuch für zusätzliche Informationen.
        return obs, rew, terminated, info  # Returns observation, reward, termination status, and additional info.  # Gibt Beobachtung, Belohnung, Beendigungsstatus und zusätzliche Informationen zurück.

    def wait(self):  # Placeholder method for waiting.  # Platzhalter-Methode zum Warten.
        pass  # Does nothing.  # Tut nichts.

    def render(self):  # Renders the environment for visual representation.  # Rendert die Umgebung für eine visuelle Darstellung.
        image = self.blank_image.copy()  # Creates a copy of the blank image for rendering.  # Erstellt eine Kopie des leeren Bildes für das Rendering.
        pos_x, pos_y = self.rc_drone.get_observation()  # Gets the drone's current position.  # Holt die aktuelle Position der Drohne.
        image = cv2.circle(img=image,  # Draws a circle at the drone's position.  # Zeichnet einen Kreis an der Position der Drohne.
                           center=(int(pos_x * 200) + 250, int(pos_y * 200) + 250),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=1)
        image = cv2.circle(img=image,  # Draws a circle at the target position.  # Zeichnet einen Kreis an der Zielposition.
                           center=(int(self.target[0] * 200) + 250, int(self.target[1] * 200) + 250),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("Dummy RC drone", image)  # Displays the rendered image in a window.  # Zeigt das gerenderte Bild in einem Fenster an.
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Checks if the user presses the 'q' key to quit.  # Überprüft, ob der Benutzer die 'q'-Taste drückt, um zu beenden.
            return  # Exits the render loop.  # Beendet die Render-Schleife.

# rtgym configuration dictionary:  # Configuration settings for the rtgym environment.  # Konfigurationseinstellungen für die rtgym-Umgebung.

DUMMY_RC_DRONE_CONFIG = DEFAULT_CONFIG_DICT.copy()  # Creates a copy of the default configuration dictionary.  # Erstellt eine Kopie des Standard-Konfigurations-Wörterbuchs.
DUMMY_RC_DRONE_CONFIG["interface"] = DummyRCDroneInterface  # Sets the interface to DummyRCDroneInterface.  # Setzt die Schnittstelle auf DummyRCDroneInterface.
DUMMY_RC_DRONE_CONFIG["time_step_duration"] = 0.05  # Sets the duration of each time step to 0.05 seconds.  # Setzt die Dauer jedes Zeitschritts auf 0,05 Sekunden.
DUMMY_RC_DRONE_CONFIG["start_obs_capture"] = 0.05  # Sets the observation capture start time to 0.05 seconds.  # Setzt die Startzeit der Beobachtungsaufnahme auf 0,05 Sekunden.
DUMMY_RC_DRONE_CONFIG["time_step_timeout_factor"] = 1.0  # Sets the time step timeout factor to 1.0.  # Setzt den Timeout-Faktor des Zeitschritts auf 1,0.
DUMMY_RC_DRONE_CONFIG["ep_max_length"] = 100  # Sets the maximum length of an episode to 100 steps.  # Setzt die maximale Länge einer Episode auf 100 Schritte.
DUMMY_RC_DRONE_CONFIG["act_buf_len"] = 4  # Sets the action buffer length to 4.  # Setzt die Länge des Aktionspuffers auf 4.
DUMMY_RC_DRONE_CONFIG["reset_act_buf"] = False  # Sets the action buffer reset flag to False.  # Setzt das Flag zum Zurücksetzen des Aktionspuffers auf False.
DUMMY_RC_DRONE_CONFIG["benchmark"] = True  # Enables benchmarking.  # Aktiviert das Benchmarking.
DUMMY_RC_DRONE_CONFIG["benchmark_polyak"] = 0.2  # Sets the Polyak averaging parameter for benchmarking.  # Setzt den Polyak-Durchschnitts-Parameter für das Benchmarking.
