# standard library imports
from dataclasses import InitVar, dataclass  # Importing InitVar and dataclass from the dataclasses module for class construction.  # Importiere InitVar und dataclass aus dem Modul dataclasses für die Klassenerstellung.

# third-party imports
import gymnasium  # Importing the gymnasium library to work with gym environments.  # Importiere die Gymnasium-Bibliothek, um mit Gym-Umgebungen zu arbeiten.

# local imports
from tmrl.wrappers import (AffineObservationWrapper, Float64ToFloat32)  # Importing custom wrappers for preprocessing and observation conversion.  # Importiere benutzerdefinierte Wrapper für Vorverarbeitung und Beobachtungsumwandlung.

__docformat__ = "google"  # Setting the documentation format to Google style.  # Setzt das Dokumentationsformat auf den Google-Stil.

class GenericGymEnv(gymnasium.Wrapper):  # Defining a class that wraps a gymnasium environment.  # Definiert eine Klasse, die eine Gymnasium-Umgebung umhüllt.
    def __init__(self, id: str = "Pendulum-v0", gym_kwargs=None, obs_scale: float = 0., to_float32=False):  # Constructor for the GenericGymEnv class. Initializes the environment with optional parameters.  # Konstruktor der GenericGymEnv-Klasse. Initialisiert die Umgebung mit optionalen Parametern.
        """
        Use this wrapper when using the framework with arbitrary environments.  # This docstring explains the purpose of the class.  # Diese docstring erklärt den Zweck der Klasse.

        Args:  # Explains the arguments accepted by the class.  # Erklärt die Argumente, die von der Klasse akzeptiert werden.
            id (str): gymnasium id  # The ID of the gym environment (default is "Pendulum-v0").  # Die ID der Gym-Umgebung (Standard ist "Pendulum-v0").
            gym_kwargs (dict): keyword arguments of the gymnasium environment (i.e. between -1.0 and 1.0 when the actual action space is something else)  # Additional gymnasium environment configurations.  # Zusätzliche Konfigurationen der Gym-Umgebung.
            obs_scale (float): change this if wanting to rescale actions by a scalar  # Rescale actions by a scalar if needed.  # Skaliere Aktionen bei Bedarf mit einem Skalar.
            to_float32 (bool): set this to True if wanting observations to be converted to numpy.float32  # If True, convert observations to numpy.float32.  # Wenn True, konvertiere Beobachtungen in numpy.float32.
        """
        if gym_kwargs is None:  # Checks if gym_kwargs is not provided.  # Überprüft, ob gym_kwargs nicht angegeben sind.
            gym_kwargs = {}  # Sets gym_kwargs to an empty dictionary if not provided.  # Setzt gym_kwargs auf ein leeres Wörterbuch, wenn nicht angegeben.
        env = gymnasium.make(id, **gym_kwargs, disable_env_checker=True)  # Creates a gym environment using the given ID and configurations.  # Erstellt eine Gym-Umgebung mit der gegebenen ID und den Konfigurationen.
        if obs_scale:  # Checks if observation scaling is needed.  # Überprüft, ob eine Beobachtungsskala erforderlich ist.
            env = AffineObservationWrapper(env, 0, obs_scale)  # Applies the AffineObservationWrapper to rescale the observations.  # Wendet den AffineObservationWrapper an, um die Beobachtungen neu zu skalieren.
        if to_float32:  # Checks if conversion to float32 is needed.  # Überprüft, ob eine Konvertierung zu float32 erforderlich ist.
            env = Float64ToFloat32(env)  # Converts the environment's observations to numpy.float32 if necessary.  # Konvertiert die Beobachtungen der Umgebung in numpy.float32, falls erforderlich.
        # assert isinstance(env.action_space, gymnasium.spaces.Box), f"{env.action_space}"  # Asserts that the action space is of type Box, which is typically for continuous actions.  # Stellt sicher, dass der Aktionsraum vom Typ Box ist, der normalerweise für kontinuierliche Aktionen verwendet wird.
        # env = NormalizeActionWrapper(env)  # This line is commented out but would normalize actions if uncommented.  # Diese Zeile ist auskommentiert, würde jedoch Aktionen normalisieren, wenn sie entkommentiert wird.
        super().__init__(env)  # Calls the constructor of the parent class (gymnasium.Wrapper) to initialize the environment.  # Ruft den Konstruktor der Elternklasse (gymnasium.Wrapper) auf, um die Umgebung zu initialisieren.

if __name__ == '__main__':  # Checks if the script is run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    pass  # Does nothing in this case. Placeholder for future code.  # Tut in diesem Fall nichts. Platzhalter für zukünftigen Code.
