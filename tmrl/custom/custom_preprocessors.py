# third-party imports
import numpy as np  # Importing the NumPy library for numerical operations.  # Importiert die NumPy-Bibliothek für numerische Operationen.
import logging  # Importing the logging library for logging messages.  # Importiert die Logging-Bibliothek für das Protokollieren von Nachrichten.
import cv2  # Importing the OpenCV library for computer vision tasks.  # Importiert die OpenCV-Bibliothek für Computer Vision-Aufgaben.

# OBSERVATION PREPROCESSING ==================================

def obs_preprocessor_tm_act_in_obs(obs):  # Function definition for preprocessing observations in a specific environment.  # Funktionsdefinition zur Vorverarbeitung von Beobachtungen in einer spezifischen Umgebung.
    """
    Preprocessor for TM2020 full environment with grayscale images  # Documentation explaining the purpose of this function.  # Dokumentation, die den Zweck dieser Funktion erklärt.
    """
    grayscale_images = obs[3]  # Extract grayscale images from the 4th element of the input observation.  # Extrahiert Graustufenbilder aus dem 4. Element der Eingabe-Beobachtung.
    grayscale_images = grayscale_images.astype(np.float32) / 256.0  # Converts grayscale images to float32 and normalizes them by dividing by 256.  # Konvertiert Graustufenbilder in float32 und normalisiert sie durch Division durch 256.
    obs = (obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10000.0, grayscale_images, *obs[4:])  # Scales specific elements of the observation and keeps others unchanged.  # Skaliert bestimmte Elemente der Beobachtung und lässt andere unverändert.
    return obs  # Returns the processed observation.  # Gibt die verarbeitete Beobachtung zurück.

def obs_preprocessor_tm_lidar_act_in_obs(obs):  # Function definition for preprocessing observations with LIDAR data.  # Funktionsdefinition zur Vorverarbeitung von Beobachtungen mit LIDAR-Daten.
    """
    Preprocessor for the LIDAR environment, flattening LIDARs  # Documentation for the preprocessing of LIDAR observations.  # Dokumentation für die Vorverarbeitung von LIDAR-Beobachtungen.
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), *obs[2:])  # Flattens the LIDAR data in the second element of the observation.  # Flatten das LIDAR-Daten im zweiten Element der Beobachtung.
    return obs  # Returns the modified observation with flattened LIDAR data.  # Gibt die modifizierte Beobachtung mit abgeflachten LIDAR-Daten zurück.

def obs_preprocessor_tm_lidar_progress_act_in_obs(obs):  # Function definition for processing LIDAR data with progress.  # Funktionsdefinition zur Verarbeitung von LIDAR-Daten mit Fortschritt.
    """
    Preprocessor for the LIDAR environment, flattening LIDARs  # Documentation for flattening LIDAR data with progress included.  # Dokumentation zum Abflachen von LIDAR-Daten mit eingeschlossenem Fortschritt.
    """
    obs = (obs[0], obs[1], np.ndarray.flatten(obs[2]), *obs[3:])  # Flattens the LIDAR data in the third element of the observation.  # Flatten das LIDAR-Daten im dritten Element der Beobachtung.
    return obs  # Returns the observation with LIDAR data flattened and progress intact.  # Gibt die Beobachtung mit abgeflachten LIDAR-Daten und intaktem Fortschritt zurück.

# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC

def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated):  # Function for preprocessing samples from replay memory.  # Funktion zur Vorverarbeitung von Stichproben aus dem Wiedergabespeicher.
    return last_obs, act, rew, new_obs, terminated, truncated  # Returns the same sample without modifications.  # Gibt die gleiche Stichprobe ohne Änderungen zurück.
