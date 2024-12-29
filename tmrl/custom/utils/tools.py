# standard library imports
import math  # Import the math module for mathematical functions like sin, cos, radians.  # Importiere das math-Modul für mathematische Funktionen wie sin, cos, radian.
import os  # Import the os module for interacting with the operating system, e.g., file paths.  # Importiere das os-Modul für die Interaktion mit dem Betriebssystem, z.B. Dateipfade.
import socket  # Import the socket module for network communication (TCP/UDP).  # Importiere das socket-Modul für die Netzwerkkommunikation (TCP/UDP).
import struct  # Import the struct module for packing and unpacking binary data.  # Importiere das struct-Modul zum Packen und Entpacken von Binärdaten.
import time  # Import the time module for time-related functions (e.g., sleep).  # Importiere das time-Modul für zeitbezogene Funktionen (z.B. sleep).
from pathlib import Path  # Import Path from pathlib for easy manipulation of file paths.  # Importiere Path von pathlib für die einfache Handhabung von Dateipfaden.
from threading import Lock, Thread  # Import Lock and Thread from threading for multithreading support.  # Importiere Lock und Thread von threading für die Unterstützung von Multithreading.

# third-party imports
import cv2  # Import OpenCV for image processing (Computer Vision).  # Importiere OpenCV für Bildverarbeitung (Computer Vision).
import numpy as np  # Import NumPy for numerical operations on arrays.  # Importiere NumPy für numerische Operationen auf Arrays.

# local imports
from tmrl.config.config_constants import LIDAR_BLACK_THRESHOLD  # Import a constant from local config for thresholding in Lidar processing.  # Importiere eine Konstante aus der lokalen Konfiguration für die Schwellenwertbestimmung in der Lidar-Verarbeitung.

class TM2020OpenPlanetClient:  # Define the TM2020OpenPlanetClient class for network communication.  # Definiere die Klasse TM2020OpenPlanetClient für die Netzwerkkommunikation.
    def __init__(self, host='127.0.0.1', port=9000, struct_str='<' + 'f' * 11):  # Initialize the client with default parameters.  # Initialisiere den Client mit Standardparametern.
        self._struct_str = struct_str  # Store the structure string for unpacking data.  # Speichere die Strukturzeichenkette zum Entpacken von Daten.
        self.nb_floats = self._struct_str.count('f')  # Count the number of floats in the structure.  # Zähle die Anzahl der Floats in der Struktur.
        self.nb_uint64 = self._struct_str.count('Q')  # Count the number of unsigned 64-bit integers.  # Zähle die Anzahl der unsigned 64-Bit-Ganzzahlen.
        self._nb_bytes = self.nb_floats * 4 + self.nb_uint64 * 8  # Calculate the number of bytes required for the structure.  # Berechne die Anzahl der Bytes, die für die Struktur erforderlich sind.

        self._host = host  # Store the host address (IP).  # Speichere die Host-Adresse (IP).
        self._port = port  # Store the port number.  # Speichere die Portnummer.

        # Threading attributes:
        self.__lock = Lock()  # Create a lock object for thread safety.  # Erstelle ein Lock-Objekt für Threadsicherheit.
        self.__data = None  # Initialize data as None (to hold received data).  # Initialisiere die Daten als None (zum Halten der empfangenen Daten).
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)  # Start a new client thread.  # Starte einen neuen Client-Thread.
        self.__t_client.start()  # Start the client thread in the background.  # Starte den Client-Thread im Hintergrund.

    def __client_thread(self):  # Define the thread function for listening to the server.  # Definiere die Thread-Funktion für das Lauschen auf den Server.
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        TODO: handle disconnection
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # Create a socket for TCP connection.  # Erstelle ein Socket für die TCP-Verbindung.
            s.connect((self._host, self._port))  # Connect to the specified host and port.  # Verbinde mit dem angegebenen Host und Port.
            data_raw = b''  # Initialize an empty byte string to store raw data.  # Initialisiere einen leeren Byte-String zum Speichern roher Daten.
            while True:  # main loop to continuously receive data.  # Hauptschleife zum kontinuierlichen Empfangen von Daten.
                while len(data_raw) < self._nb_bytes:  # Wait until the data size is sufficient.  # Warte, bis die Datenmenge ausreichend ist.
                    data_raw += s.recv(1024)  # Receive 1024 bytes of data at a time.  # Empfange 1024 Bytes Daten auf einmal.
                div = len(data_raw) // self._nb_bytes  # Determine how many complete packets of data we have.  # Bestimme, wie viele vollständige Datenpakete wir haben.
                data_used = data_raw[(div - 1) * self._nb_bytes:div * self._nb_bytes]  # Extract the most recent packet.  # Extrahiere das zuletzt empfangene Paket.
                data_raw = data_raw[div * self._nb_bytes:]  # Remove the used data from the raw buffer.  # Entferne die verwendeten Daten aus dem Rohpuffer.
                self.__lock.acquire()  # Acquire the lock to modify shared data safely.  # Erhalte das Lock, um auf die gemeinsamen Daten sicher zuzugreifen.
                self.__data = data_used  # Store the latest received data.  # Speichere die zuletzt empfangenen Daten.
                self.__lock.release()  # Release the lock after modifying the shared data.  # Gib das Lock nach der Modifikation der gemeinsamen Daten frei.

    def retrieve_data(self, sleep_if_empty=0.01, timeout=10.0):  # Function to retrieve the most recent data.  # Funktion zum Abrufen der zuletzt empfangenen Daten.
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        This blocks if nothing has been received so far
        """
        c = True  # Initialize a flag for the loop.  # Initialisiere ein Flag für die Schleife.
        t_start = None  # Initialize the start time for timeout.  # Initialisiere die Startzeit für das Timeout.
        while c:  # Loop until data is successfully retrieved.  # Schleife, bis Daten erfolgreich abgerufen wurden.
            self.__lock.acquire()  # Acquire the lock to read shared data safely.  # Erhalte das Lock, um auf gemeinsame Daten sicher zuzugreifen.
            if self.__data is not None:  # Check if data is available.  # Überprüfe, ob Daten verfügbar sind.
                data = struct.unpack(self._struct_str, self.__data)  # Unpack the raw data using the specified structure string.  # Entpacke die rohen Daten mit der angegebenen Strukturzeichenkette.
                c = False  # Set the flag to stop the loop.  # Setze das Flag, um die Schleife zu stoppen.
                self.__data = None  # Reset the shared data after retrieval.  # Setze die gemeinsamen Daten nach dem Abrufen auf None.
            self.__lock.release()  # Release the lock after reading the data.  # Gib das Lock nach dem Lesen der Daten frei.
            if c:  # If no data was found, check if timeout occurred.  # Wenn keine Daten gefunden wurden, überprüfe, ob das Timeout erreicht wurde.
                if t_start is None:  # If start time is not set, initialize it.  # Wenn die Startzeit nicht gesetzt ist, initialisiere sie.
                    t_start = time.time()  # Set the start time to the current time.  # Setze die Startzeit auf die aktuelle Zeit.
                t_now = time.time()  # Get the current time.  # Hole die aktuelle Zeit.
                assert t_now - t_start < timeout, f"OpenPlanet stopped sending data since more than {timeout}s."  # Assert that the timeout is not exceeded.  # Stelle sicher, dass das Timeout nicht überschritten wird.
                time.sleep(sleep_if_empty)  # Sleep for the specified duration before trying again.  # Schlafe für die angegebene Dauer, bevor erneut versucht wird.
        return data  # Return the retrieved data.  # Gib die abgerufenen Daten zurück.

def save_ghost(host='127.0.0.1', port=10000):  # Define a function to save the current ghost data.  # Definiere eine Funktion, um die aktuellen Ghost-Daten zu speichern.
    """
    Saves the current ghost

    Args:
        host (str): IP address of the ghost-saving server
        port (int): Port of the ghost-saving server
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # Create a socket for communication.  # Erstelle ein Socket für die Kommunikation.
        s.connect((host, port))  # Connect to the specified host and port.  # Verbinde mit dem angegebenen Host und Port.

def armin(tab):  # Define a helper function to find the first non-zero element in an array.  # Definiere eine Hilfsfunktion, um das erste Nicht-Null-Element in einem Array zu finden.
    nz = np.nonzero(tab)[0]  # Find the indices of non-zero elements.  # Finde die Indizes der Nicht-Null-Elemente.
    if len(nz) != 0:  # If there are non-zero elements.  # Wenn Nicht-Null-Elemente vorhanden sind.
        return nz[0].item()  # Return the first non-zero element.  # Gib das erste Nicht-Null-Element zurück.
    else:  # If no non-zero elements are found.  # Wenn keine Nicht-Null-Elemente gefunden wurden.
        return len(tab) - 1  # Return the last index.  # Gib den letzten Index zurück.

class Lidar:  # Define the Lidar class for processing lidar image data.  # Definiere die Lidar-Klasse für die Verarbeitung von Lidar-Bilddaten.
    def __init__(self, im):  # Initialize the Lidar object with an image.  # Initialisiere das Lidar-Objekt mit einem Bild.
        self._set_axis_lidar(im)  # Set the axis for lidar processing.  # Setze die Achse für die Lidar-Verarbeitung.
        self.black_threshold = LIDAR_BLACK_THRESHOLD  # Set the threshold for detecting black pixels.  # Setze den Schwellenwert für das Erkennen von schwarzen Pixeln.

    def _set_axis_lidar(self, im):  # Define the function to set up the Lidar axes based on the image.  # Definiere die Funktion zum Einrichten der Lidar-Achsen basierend auf dem Bild.
        h, w, _ = im.shape  # Get the height and width of the image.  # Hole die Höhe und Breite des Bildes.
        self.h = h  # Set the height.  # Setze die Höhe.
        self.w = w  # Setze die Breite.
        self.road_point = (44*h//49, w//2)  # Set the reference point for the road.  # Setze den Referenzpunkt für die Straße.
        min_dist = 20  # Minimum distance for lidar detection.  # Mindestdistanz für die Lidar-Erkennung.
        list_ax_x = []  # Initialize a list for X-axis coordinates.  # Initialisiere eine Liste für X-Achsen-Koordinaten.
        list_ax_y = []  # Initialize a list for Y-axis coordinates.  # Initialisiere eine Liste für Y-Achsen-Koordinaten.
        for angle in range(90, 280, 10):  # Loop through angles from 90° to 280° in steps of 10°.  # Schleife durch Winkel von 90° bis 280° in Schritten von 10°.
            axis_x = []  # Initialize a list for X coordinates of the axis.  # Initialisiere eine Liste für X-Koordinaten der Achse.
            axis_y = []  # Initialize a list for Y coordinates of the axis.  # Initialisiere eine Liste für Y-Koordinaten der Achse.
            x = self.road_point[0]  # Get the X coordinate of the road point.  # Hole die X-Koordinate des Straßenpunkts.
            y = self.road_point[1]  # Get the Y coordinate of the road point.  # Hole die Y-Koordinate des Straßenpunkts.
            dx = math.cos(math.radians(angle))  # Calculate the X direction cosine for the angle.  # Berechne den Kosinus der X-Richtung für den Winkel.
            dy = math.sin(math.radians(angle))  # Calculate the Y direction sine for the angle.  # Berechne den Sinus der Y-Richtung für den Winkel.
            lenght = False  # Initialize the length flag.  # Initialisiere das Längen-Flag.
            dist = min_dist  # Start distance.  # Starte mit der Mindestdistanz.
            while not lenght:  # Loop until the length is sufficient.  # Schleife, bis die Länge ausreichend ist.
                newx = int(x + dist * dx)  # Calculate new X coordinate.  # Berechne die neue X-Koordinate.
                newy = int(y + dist * dy)  # Calculate new Y coordinate.  # Berechne die neue Y-Koordinate.
                if newx <= 0 or newy <= 0 or newy >= w - 1:  # Check if the coordinates are out of bounds.  # Überprüfe, ob die Koordinaten außerhalb des Rahmens liegen.
                    lenght = True  # Set the length flag to True.  # Setze das Längen-Flag auf True.
                    list_ax_x.append(np.array(axis_x))  # Append the X axis list.  # Hänge die X-Achsen-Liste an.
                    list_ax_y.append(np.array(axis_y))  # Append the Y axis list.  # Hänge die Y-Achsen-Liste an.
                else:  # If the coordinates are within bounds.  # Wenn die Koordinaten innerhalb des Rahmens liegen.
                    axis_x.append(newx)  # Append the new X coordinate.  # Hänge die neue X-Koordinate an.
                    axis_y.append(newy)  # Append the new Y coordinate.  # Hänge die neue Y-Koordinate an.
                dist = dist + 1  # Increase the distance for the next point.  # Erhöhe die Distanz für den nächsten Punkt.
        self.list_axis_x = list_ax_x  # Store the X-axis coordinates.  # Speichere die X-Achsen-Koordinaten.
        self.list_axis_y = list_ax_y  # Store the Y-axis coordinates.  # Speichere die Y-Achsen-Koordinaten.

    def lidar_20(self, img, show=False):  # Define the function to process lidar data for an image.  # Definiere die Funktion zur Verarbeitung von Lidar-Daten für ein Bild.
        h, w, _ = img.shape  # Get the height and width of the image.  # Hole die Höhe und Breite des Bildes.
        if h != self.h or w != self.w:  # If the image dimensions don't match, re-setup the axes.  # Wenn die Bildmaße nicht übereinstimmen, setze die Achsen neu.
            self._set_axis_lidar(img)  # Re-setup the lidar axes.  # Setze die Lidar-Achsen neu.
        distances = []  # Initialize a list to store distances.  # Initialisiere eine Liste zum Speichern von Entfernungen.
        if show:  # If the 'show' flag is set to True, display the image.  # Wenn das Flag 'show' auf True gesetzt ist, zeige das Bild an.
            color = (255, 0, 0)  # Set the color for drawing lines.  # Setze die Farbe zum Zeichnen von Linien.
            thickness = 4  # Set the thickness of the lines.  # Setze die Dicke der Linien.
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert the image color format.  # Konvertiere das Bildfarbformat.
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y):  # Loop through all axis coordinates.  # Schleife durch alle Achsenkoordinaten.
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))  # Find the first index where the pixel value is below the threshold.  # Finde den ersten Index, an dem der Pixelwert unter dem Schwellenwert liegt.
            if show:  # If the 'show' flag is True, draw a line on the image.  # Wenn das 'show'-Flag auf True gesetzt ist, zeichne eine Linie auf das Bild.
                img = cv2.line(img, (self.road_point[1], self.road_point[0]), (axis_y[index], axis_x[index]), color, thickness)  # Draw the line.  # Zeichne die Linie.
            index = np.float32(index)  # Convert the index to a float.  # Konvertiere den Index zu einem Float.
            distances.append(index)  # Append the distance to the list.  # Hänge die Distanz an die Liste an.
        res = np.array(distances, dtype=np.float32)  # Convert the distances list to a NumPy array.  # Konvertiere die Distanzliste in ein NumPy-Array.
        if show:  # If 'show' flag is True, display the image with lines.  # Wenn das 'show'-Flag auf True gesetzt ist, zeige das Bild mit Linien an.
            cv2.imshow("Environment", img)  # Show the processed image.  # Zeige das bearbeitete Bild.
            cv2.waitKey(1)  # Wait for a key press.  # Warte auf einen Tastendruck.
        return res  # Return the distances as a NumPy array.  # Gib die Entfernungen als NumPy-Array zurück.

if __name__ == "__main__":  # If this file is run as a script, this block will be executed.  # Wenn diese Datei als Skript ausgeführt wird, wird dieser Block ausgeführt.
    pass  # No action is taken in this block.  # Es wird keine Aktion in diesem Block ausgeführt.
