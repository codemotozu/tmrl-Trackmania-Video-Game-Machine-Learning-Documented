# third-party imports
import gymnasium  # Imports the gymnasium library for reinforcement learning environments.  # Importiert die gymnasium-Bibliothek für Verstärkungslern-Umgebungen.
import cv2  # Imports the OpenCV library for computer vision tasks, such as image processing.  # Importiert die OpenCV-Bibliothek für Computer Vision-Aufgaben, wie z.B. Bildverarbeitung.
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT  # Imports a default configuration dictionary for a real-time environment from rtgym.  # Importiert ein Standard-Konfigurations-Wörterbuch für eine Echtzeit-Umgebung aus rtgym.

# local imports
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar  # Imports custom interfaces for the TrackMania 2020 game environment.  # Importiert benutzerdefinierte Schnittstellen für die TrackMania 2020-Spielumgebung.
from tmrl.custom.utils.window import WindowInterface  # Imports a utility to interact with windows, like resizing and screenshot capturing.  # Importiert ein Hilfsmodul für die Interaktion mit Fenstern, z.B. für das Ändern der Größe und das Aufnehmen von Screenshots.
from tmrl.custom.utils.tools import Lidar  # Imports a Lidar utility for capturing depth data from images.  # Importiert ein Lidar-Hilfsmodul zum Erfassen von Tiefendaten aus Bildern.
import tmrl.config.config_constants as cfg  # Imports configuration constants for the program from the tmrl.config module.  # Importiert Konfigurationskonstanten für das Programm aus dem tmrl.config-Modul.
import logging  # Imports the logging library to record logs and debug information.  # Importiert die Logging-Bibliothek, um Protokolle und Debug-Informationen aufzuzeichnen.

# Function to check the environment with Lidar sensor
def check_env_tm20lidar():  
    window_interface = WindowInterface("Trackmania")  # Creates a WindowInterface object to interact with the Trackmania game window.  # Erstellt ein WindowInterface-Objekt, um mit dem Trackmania-Spiel-Fenster zu interagieren.
    if cfg.SYSTEM != "Windows":  # Checks if the system is not Windows.  # Überprüft, ob das System nicht Windows ist.
        window_interface.move_and_resize()  # Moves and resizes the window if the system is Linux (required on Linux).  # Verschiebt und ändert die Größe des Fensters, wenn das System Linux ist (erforderlich auf Linux).
    lidar = Lidar(window_interface.screenshot())  # Takes a screenshot of the game window and initializes the Lidar object.  # Macht einen Screenshot des Spiel-Fensters und initialisiert das Lidar-Objekt.
    env_config = DEFAULT_CONFIG_DICT.copy()  # Copies the default configuration dictionary for the environment setup.  # Kopiert das Standard-Konfigurations-Wörterbuch für die Umgebungs-Einrichtung.
    env_config["interface"] = TM2020InterfaceLidar  # Sets the environment interface to TM2020InterfaceLidar.  # Setzt die Umgebungs-Schnittstelle auf TM2020InterfaceLidar.
    env_config["wait_on_done"] = True  # Ensures that the environment waits until the task is done.  # Stellt sicher, dass die Umgebung wartet, bis die Aufgabe abgeschlossen ist.
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False}  # Sets specific arguments for the interface (image history length and no gamepad).  # Setzt spezifische Argumente für die Schnittstelle (Bild-Historienlänge und kein Gamepad).
    env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)  # Creates the environment using the gymnasium library with the specified config.  # Erstellt die Umgebung unter Verwendung der gymnasium-Bibliothek mit der angegebenen Konfiguration.
    o, i = env.reset()  # Resets the environment and retrieves the initial observation (o) and info (i).  # Setzt die Umgebung zurück und ruft die anfängliche Beobachtung (o) sowie die Infos (i) ab.
    while True:  # Begins an infinite loop to interact with the environment.  # Beginnt eine unendliche Schleife, um mit der Umgebung zu interagieren.
        o, r, d, t, i = env.step(None)  # Takes a step in the environment (no action passed here).  # Macht einen Schritt in der Umgebung (keine Aktion wird hier übergeben).
        logging.info(f"r:{r}, d:{d}, t:{t}")  # Logs the reward (r), done (d), and time (t) values.  # Protokolliert die Werte für Belohnung (r), abgeschlossen (d) und Zeit (t).
        if d or t:  # Checks if the episode is done or if time has run out.  # Überprüft, ob die Episode abgeschlossen ist oder die Zeit abgelaufen ist.
            o, i = env.reset()  # Resets the environment if done or time is up.  # Setzt die Umgebung zurück, wenn die Episode abgeschlossen ist oder die Zeit abgelaufen ist.
        img = window_interface.screenshot()[:, :, :3]  # Captures a screenshot and removes alpha channel.  # Nimmt einen Screenshot auf und entfernt den Alphakanal.
        lidar.lidar_20(img, True)  # Processes the captured image using Lidar to gather depth information.  # Verarbeitet das aufgenommene Bild mithilfe von Lidar, um Tiefeninformationen zu sammeln.

# Function to show images with a specific scale
def show_imgs(imgs, scale=cfg.IMG_SCALE_CHECK_ENV):  # Defines a function to display images with scaling.  # Definiert eine Funktion, um Bilder mit Skalierung anzuzeigen.
    imshape = imgs.shape  # Gets the shape of the images.  # Ruft die Form der Bilder ab.
    if len(imshape) == 3:  # Checks if the image is grayscale (3D array with no color channel).  # Überprüft, ob das Bild in Graustufen ist (3D-Array ohne Farbkanal).
        nb, h, w = imshape  # Extracts the number of images (nb), height (h), and width (w).  # Extrahiert die Anzahl der Bilder (nb), Höhe (h) und Breite (w).
        concat = imgs.reshape((nb*h, w))  # Reshapes the images into a single vertical stack.  # Formt die Bilder zu einem einzelnen vertikalen Stapel um.
        width = int(concat.shape[1] * scale)  # Scales the width of the concatenated image.  # Skaliert die Breite des zusammengefügten Bildes.
        height = int(concat.shape[0] * scale)  # Skales the height of the concatenated image.  # Skaliert die Höhe des zusammengefügten Bildes.
        cv2.imshow("Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST))  # Displays the resized image.  # Zeigt das neu skalierte Bild an.
        cv2.waitKey(1)  # Waits for a key event for 1 ms.  # Wartet 1 ms auf ein Tastenereignis.
    elif len(imshape) == 4:  # Checks if the image is in color (4D array with a color channel).  # Überprüft, ob das Bild farbig ist (4D-Array mit Farbkanal).
        nb, h, w, c = imshape  # Extracts the number of images (nb), height (h), width (w), and color channels (c).  # Extrahiert die Anzahl der Bilder (nb), Höhe (h), Breite (w) und Farbkanäle (c).
        concat = imgs.reshape((nb*h, w, c))  # Reshapes the images into a single vertical stack with colors.  # Formt die Bilder zu einem einzelnen vertikalen Stapel mit Farben um.
        width = int(concat.shape[1] * scale)  # Scales the width of the concatenated image with colors.  # Skaliert die Breite des zusammengefügten Bildes mit Farben.
        height = int(concat.shape[0] * scale)  # Scales the height of the concatenated image with colors.  # Skaliert die Höhe des zusammengefügten Bildes mit Farben.
        cv2.imshow("Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST))  # Displays the resized color image.  # Zeigt das neu skalierte Farb-Bild an.
        cv2.waitKey(1)  # Waits for a key event for 1 ms.  # Wartet 1 ms auf ein Tastenereignis.

# Function to check the environment with the full interface (not Lidar)
def check_env_tm20full():  
    env_config = DEFAULT_CONFIG_DICT.copy()  # Copies the default configuration dictionary for the environment setup.  # Kopiert das Standard-Konfigurations-Wörterbuch für die Umgebungs-Einrichtung.
    env_config["interface"] = TM2020Interface  # Sets the environment interface to TM2020Interface.  # Setzt die Umgebungs-Schnittstelle auf TM2020Interface.
    env_config["wait_on_done"] = True  # Ensures that the environment waits until the task is done.  # Stellt sicher, dass die Umgebung wartet, bis die Aufgabe abgeschlossen ist.
    env_config["interface_kwargs"] = {"gamepad": False,  # Sets gamepad to False in the interface arguments.  # Setzt das Gamepad auf False in den Schnittstellen-Argumenten.
                                      "grayscale": cfg.GRAYSCALE,  # Sets grayscale mode based on configuration.  # Setzt den Graustufenmodus basierend auf der Konfiguration.
                                      "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)}  # Sets the image resize dimensions based on configuration.  # Setzt die Bildgrößenänderung basierend auf der Konfiguration.
    env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)  # Creates the environment using the gymnasium library with the specified config.  # Erstellt die Umgebung unter Verwendung der gymnasium-Bibliothek mit der angegebenen Konfiguration.
    o, i = env.reset()  # Resets the environment and retrieves the initial observation (o) and info (i).  # Setzt die Umgebung zurück und ruft die anfängliche Beobachtung (o) sowie die Infos (i) ab.
    show_imgs(o[3])  # Displays the images from the environment.  # Zeigt die Bilder aus der Umgebung an.
    logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")  # Logs the initial observation values.  # Protokolliert die anfänglichen Beobachtungswerte.
    while True:  # Begins an infinite loop to interact with the environment.  # Beginnt eine unendliche Schleife, um mit der Umgebung zu interagieren.
        o, r, d, t, i = env.step(None)  # Takes a step in the environment (no action passed here).  # Macht einen Schritt in der Umgebung (keine Aktion wird hier übergeben).
        show_imgs(o[3])  # Displays the images from the environment after the step.  # Zeigt die Bilder aus der Umgebung nach dem Schritt an.
        logging.info(f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")  # Logs the reward, done, time, and observation values.  # Protokolliert die Belohnung, abgeschlossen, Zeit und Beobachtungswerte.
        if d or t:  # Checks if the episode is done or if time has run out.  # Überprüft, ob die Episode abgeschlossen ist oder die Zeit abgelaufen ist.
            o, i = env.reset()  # Resets the environment if done or time is up.  # Setzt die Umgebung zurück, wenn die Episode abgeschlossen ist oder die Zeit abgelaufen ist.
            show_imgs(o[3])  # Displays the images from the environment after the reset.  # Zeigt die Bilder aus der Umgebung nach dem Zurücksetzen an.
            logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")  # Logs the new observation values after reset.  # Protokolliert die neuen Beobachtungswerte nach dem Zurücksetzen.

# Main execution of the script
if __name__ == "__main__":  
    # check_env_tm20lidar()  # Calls the function to check the environment with Lidar.  # Ruft die Funktion auf, um die Umgebung mit Lidar zu überprüfen.
    check_env_tm20full()  # Calls the function to check the environment with the full interface.  # Ruft die Funktion auf, um die Umgebung mit der vollständigen Schnittstelle zu überprüfen.
