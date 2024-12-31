# standard library imports
from typing import Mapping, Sequence  # Importing the Mapping and Sequence types from typing to define the types for collections.  # Importieren der Typen Mapping und Sequence aus typing, um die Typen für Sammlungen zu definieren.

# third-party imports
import gymnasium  # Importing the gymnasium library, a popular toolkit for building reinforcement learning environments.  # Importieren der gymnasium-Bibliothek, ein beliebtes Toolkit zum Erstellen von Reinforcement-Learning-Umgebungen.
import numpy as np  # Importing the numpy library for numerical operations, especially with arrays.  # Importieren der numpy-Bibliothek für numerische Operationen, insbesondere mit Arrays.

class AffineObservationWrapper(gymnasium.ObservationWrapper):  # Creating a custom wrapper class that inherits from gymnasium's ObservationWrapper class.  # Erstellen einer benutzerdefinierten Wrapper-Klasse, die von der ObservationWrapper-Klasse von gymnasium erbt.
    def __init__(self, env, shift, scale):  # Constructor that initializes the environment, shift, and scale.  # Konstruktor, der die Umgebung, den Shift und den Scale initialisiert.
        super().__init__(env)  # Calling the constructor of the parent class (ObservationWrapper).  # Aufrufen des Konstruktors der übergeordneten Klasse (ObservationWrapper).
        assert isinstance(env.observation_space, gymnasium.spaces.Box)  # Ensures that the observation space is a Box space type.  # Stellt sicher, dass der Beobachtungsraum vom Typ Box ist.
        self.shift = shift  # Store the shift parameter for later use.  # Speichert den Shift-Parameter für die spätere Verwendung.
        self.scale = scale  # Store the scale parameter for later use.  # Speichert den Scale-Parameter für die spätere Verwendung.
        self.observation_space = gymnasium.spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high), dtype=env.observation_space.dtype)  # Set the new observation space based on the modified low and high values.  # Setzt den neuen Beobachtungsraum basierend auf den modifizierten low- und high-Werten.

    def observation(self, obs):  # Method to apply the affine transformation (shift and scale) to the observation.  # Methode zur Anwendung der affinen Transformation (Shift und Scale) auf die Beobachtung.
        return (obs + self.shift) * self.scale  # Modify the observation using the shift and scale parameters.  # Modifiziert die Beobachtung unter Verwendung der Shift- und Scale-Parameter.

class Float64ToFloat32(gymnasium.ObservationWrapper):  # A wrapper that converts np.float64 arrays in the observations to np.float32 arrays.  # Ein Wrapper, der np.float64-Arrays in den Beobachtungen in np.float32-Arrays konvertiert.

    # TODO: change observation/action spaces to correct dtype
    def observation(self, observation):  # Method to convert all float64 and float32 data types in the observation to np.float32.  # Methode zur Konvertierung aller float64- und float32-Datentypen in der Beobachtung in np.float32.
        observation = deepmap({np.ndarray: float64_to_float32,  # Apply float64_to_float32 to numpy arrays in the observation.  # Wendet float64_to_float32 auf numpy-Arrays in der Beobachtung an.
                               float: float_to_float32,  # Apply float_to_float32 to regular float values in the observation.  # Wendet float_to_float32 auf reguläre float-Werte in der Beobachtung an.
                               np.float32: float_to_float32,  # Apply float_to_float32 to existing np.float32 values in the observation.  # Wendet float_to_float32 auf bestehende np.float32-Werte in der Beobachtung an.
                               np.float64: float_to_float32}, observation)  # Apply float_to_float32 to np.float64 values in the observation.  # Wendet float_to_float32 auf np.float64-Werte in der Beobachtung an.
        return observation  # Return the modified observation.  # Gibt die modifizierte Beobachtung zurück.

    def step(self, action):  # Custom step function to apply to the action.  # Benutzerdefinierte Schritt-Funktion für die Aktion.
        s, r, d, t, info = super().step(action)  # Call the parent class's step method and return the results.  # Ruft die Schritt-Methode der übergeordneten Klasse auf und gibt die Ergebnisse zurück.
        return s, r, d, t, info  # Return the state, reward, done, time, and info from the step.  # Gibt den Zustand, die Belohnung, done, Zeit und Info vom Schritt zurück.

# === Utilities ========================================================================================================

def deepmap(f, m):  # A utility function to apply a function to each element of a dictionary or list.  # Eine Hilfsfunktion, um eine Funktion auf jedes Element eines Wörterbuchs oder einer Liste anzuwenden.
    """Apply functions to the leaves of a dictionary or list, depending on type of the leaf value."""  # Apply a function to elements depending on their type.  # Wendet eine Funktion auf Elemente abhängig von deren Typ an.
    for cls in f:  # Iterate over the functions provided in the dictionary f.  # Iteriert über die Funktionen im Wörterbuch f.
        if isinstance(m, cls):  # If the element matches the type, apply the corresponding function.  # Wenn das Element dem Typ entspricht, wende die entsprechende Funktion an.
            return f[cls](m)  # Apply the function to the element and return the result.  # Wendet die Funktion auf das Element an und gibt das Ergebnis zurück.
    if isinstance(m, Sequence):  # If the element is a Sequence (like a list or tuple), apply deepmap to each element.  # Wenn das Element eine Sequenz (wie eine Liste oder ein Tupel) ist, wende deepmap auf jedes Element an.
        return type(m)(deepmap(f, x) for x in m)  # Apply deepmap recursively to all items in the sequence.  # Wendet deepmap rekursiv auf alle Elemente der Sequenz an.
    elif isinstance(m, Mapping):  # If the element is a Mapping (like a dictionary), apply deepmap to all values.  # Wenn das Element eine Mapping-Struktur (wie ein Wörterbuch) ist, wende deepmap auf alle Werte an.
        return type(m)((k, deepmap(f, m[k])) for k in m)  # Apply deepmap recursively to all values in the mapping.  # Wendet deepmap rekursiv auf alle Werte im Mapping an.
    else:  # If the element is neither a sequence nor a mapping, raise an error.  # Wenn das Element weder eine Sequenz noch ein Mapping ist, wird ein Fehler ausgelöst.
        raise AttributeError(f"m is a {type(m)}, not a Sequence nor a Mapping: {m}")  # Raise an error if the type is unexpected.  # Löst einen Fehler aus, wenn der Typ unerwartet ist.

def float64_to_float32(x):  # Function to convert np.float64 to np.float32.  # Funktion zur Konvertierung von np.float64 in np.float32.
    return np.asarray([x, ], np.float32) if x.dtype == np.float64 else x  # Converts to np.float32 if the input is np.float64, else returns the input unchanged.  # Konvertiert in np.float32, wenn der Eingabewert np.float64 ist, andernfalls wird der Eingabewert unverändert zurückgegeben.

def float_to_float32(x):  # Function to convert float to np.float32.  # Funktion zur Konvertierung von float in np.float32.
    return np.asarray([x, ], np.float32)  # Converts any float input into np.float32.  # Wandelt jeden float-Eingabewert in np.float32 um.
