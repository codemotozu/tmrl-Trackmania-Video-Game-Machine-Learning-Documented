# standard library imports
import pickle  # Module for serializing and deserializing objects.  # Modul zum Serialisieren und Deserialisieren von Objekten.
import time  # Module for time-related functions.  # Modul für zeitbezogene Funktionen.

# third-party imports
import numpy as np  # Library for numerical computations.  # Bibliothek für numerische Berechnungen.

# local imports
import tmrl.config.config_constants as cfg  # Import configuration constants.  # Konfigurationskonstanten importieren.
from tmrl.custom.utils.tools import TM2020OpenPlanetClient  # Import a custom client for TM2020.  # Einen benutzerdefinierten Client für TM2020 importieren.
import logging  # Module for logging messages.  # Modul zum Protokollieren von Nachrichten.

# Constants for paths
PATH_REWARD = cfg.REWARD_PATH  # Path for reward data storage.  # Pfad für die Speicherung von Belohnungsdaten.
DATASET_PATH = cfg.DATASET_PATH  # Path for dataset storage.  # Pfad für die Speicherung des Datensatzes.

def record_reward_dist(path_reward=PATH_REWARD, use_keyboard=False):
    if use_keyboard:
        import keyboard  # Import keyboard module for handling keyboard events.  # Tastaturmodul für die Verarbeitung von Tastaturereignissen importieren.

    positions = []  # List to store captured positions.  # Liste zur Speicherung erfasster Positionen.
    client = TM2020OpenPlanetClient()  # Initialize the TM2020 client.  # TM2020-Client initialisieren.
    path = path_reward  # Set the path for saving the reward data.  # Pfad für das Speichern der Belohnungsdaten festlegen.

    is_recording = False  # Flag to indicate recording state.  # Flag zur Anzeige des Aufnahmezustands.
    while True:  # Infinite loop for recording.  # Endlosschleife für die Aufnahme.
        if not is_recording:
            if not use_keyboard:
                logging.info(f"start recording")  # Log that recording is starting.  # Protokollieren, dass die Aufnahme beginnt.
                is_recording = True  # Set recording state to true.  # Aufnahmezustand auf "True" setzen.
            else:
                if keyboard.is_pressed('e'):  # Check if the 'e' key is pressed.  # Überprüfen, ob die 'e'-Taste gedrückt wurde.
                    logging.info(f"start recording")  # Log that recording is starting.  # Protokollieren, dass die Aufnahme beginnt.
                    is_recording = True  # Set recording state to true.  # Aufnahmezustand auf "True" setzen.

        if is_recording:
            data = client.retrieve_data(sleep_if_empty=0.01)  # Retrieve data with a short wait if empty.  # Daten abrufen mit kurzer Wartezeit, falls leer.
            terminated = bool(data[8])  # Check if the recording should terminate.  # Überprüfen, ob die Aufnahme beendet werden soll.

            if not use_keyboard:
                early_stop = False  # No early stop without keyboard.  # Kein vorzeitiger Stopp ohne Tastatur.
            else:
                early_stop = keyboard.is_pressed('q')  # Stop if 'q' key is pressed.  # Stopp, wenn die 'q'-Taste gedrückt wurde.

            if early_stop or terminated:
                logging.info(f"Computing reward function checkpoints from captured positions...")  # Log processing start.  # Verarbeitungstart protokollieren.
                logging.info(f"Initial number of captured positions: {len(positions)}")  # Log initial number of positions.  # Initiale Anzahl der Positionen protokollieren.
                positions = np.array(positions)  # Convert positions to NumPy array.  # Positionen in ein NumPy-Array umwandeln.

                final_positions = [positions[0]]  # Start with the first position.  # Mit der ersten Position beginnen.
                dist_between_points = 0.1  # Desired distance between points.  # Gewünschter Abstand zwischen den Punkten.
                j = 1  # Index for traversing positions.  # Index zum Durchlaufen der Positionen.
                move_by = dist_between_points  # Remaining distance to create a new point.  # Restabstand für die Erstellung eines neuen Punktes.
                pt1 = final_positions[-1]  # Last point in the final positions.  # Letzter Punkt in den endgültigen Positionen.
                while j < len(positions):  # Loop through all positions.  # Durch alle Positionen schleifen.
                    pt2 = positions[j]  # Next position in the list.  # Nächste Position in der Liste.
                    pt, dst = line(pt1, pt2, move_by)  # Calculate a new point along the line.  # Einen neuen Punkt entlang der Linie berechnen.
                    if pt is not None:  # If a point was created.  # Wenn ein Punkt erstellt wurde.
                        final_positions.append(pt)  # Add the point to the final list.  # Den Punkt zur endgültigen Liste hinzufügen.
                        move_by = dist_between_points  # Reset distance for next point.  # Abstand für den nächsten Punkt zurücksetzen.
                        pt1 = pt  # Update the current point.  # Den aktuellen Punkt aktualisieren.
                    else:  # If no point was created.  # Wenn kein Punkt erstellt wurde.
                        pt1 = pt2  # Move to the next position.  # Zur nächsten Position wechseln.
                        j += 1  # Increment the index.  # Den Index erhöhen.
                        move_by = dst  # Update the remaining distance.  # Den verbleibenden Abstand aktualisieren.

                final_positions = np.array(final_positions)  # Convert final positions to NumPy array.  # Endgültige Positionen in ein NumPy-Array umwandeln.
                logging.info(f"Final number of checkpoints in the reward function: {len(final_positions)}")  # Log the final count.  # Die endgültige Anzahl protokollieren.

                pickle.dump(final_positions, open(path, "wb"))  # Save final positions to a file.  # Endgültige Positionen in einer Datei speichern.
                logging.info(f"All done")  # Log completion.  # Fertigstellung protokollieren.
                return  # Exit the function.  # Die Funktion beenden.
            else:
                positions.append([data[2], data[3], data[4]])  # Add current position to the list.  # Aktuelle Position zur Liste hinzufügen.
        else:
            time.sleep(0.05)  # Wait before checking again.  # Warten, bevor erneut überprüft wird.

def line(pt1, pt2, dist):
    """
    Creates a point between pt1 and pt2, at distance dist from pt1.

    If dist is too large, returns None and the remaining distance (> 0.0).
    Else, returns the point and 0.0 as remaining distance.
    """
    vec = pt2 - pt1  # Vector from pt1 to pt2.  # Vektor von pt1 nach pt2.
    norm = np.linalg.norm(vec)  # Length of the vector.  # Länge des Vektors.
    if norm < dist:
        return None, dist - norm  # Not enough distance to create a point.  # Nicht genug Abstand, um einen Punkt zu erstellen.
    else:
        vec_unit = vec / norm  # Unit vector for direction.  # Einheitsvektor für die Richtung.
        pt = pt1 + vec_unit * dist  # Calculate the new point.  # Den neuen Punkt berechnen.
        return pt, 0.0  # Return the point and no remaining distance.  # Den Punkt und keine verbleibende Entfernung zurückgeben.

if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)  # Execute the recording function.  # Die Aufzeichnungsfunktion ausführen.
