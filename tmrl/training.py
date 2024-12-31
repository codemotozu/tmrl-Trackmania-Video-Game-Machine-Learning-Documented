from abc import ABC, abstractmethod  # Importing abstract base class module.  # Modul für abstrakte Basisklassen importieren.

class TrainingAgent(ABC):  # Define an abstract class 'TrainingAgent'.  # Abstrakte Klasse 'TrainingAgent' definieren.
    """
    Training algorithm.

    CAUTION: When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """
    def __init__(self,  # Constructor method for initializing the class.  # Konstruktor zur Initialisierung der Klasse.
                 observation_space,  # Space defining observations.  # Raum, der Beobachtungen definiert.
                 action_space,  # Space defining possible actions.  # Raum, der mögliche Aktionen definiert.
                 device):  # Device for training (e.g., CPU or GPU).  # Gerät für das Training (z. B. CPU oder GPU).
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)  # Beobachtungsraum (zur Vereinfachung bereitgestellt).
            action_space (gymnasium.spaces.Space): action space (here for your convenience)  # Aktionsraum (zur Vereinfachung bereitgestellt).
            device (str): device that should be used for training  # Gerät, das für das Training verwendet werden soll.
        """
        self.observation_space = observation_space  # Assign observation space to instance variable.  # Beobachtungsraum der Instanzvariable zuweisen.
        self.action_space = action_space  # Assign action space to instance variable.  # Aktionsraum der Instanzvariable zuweisen.
        self.device = device  # Assign device to instance variable.  # Gerät der Instanzvariable zuweisen.

    @abstractmethod  # Mark method as abstract (must be implemented by subclasses).  # Methode als abstrakt markieren (muss von Unterklassen implementiert werden).
    def train(self, batch):  # Abstract method for a training step.  # Abstrakte Methode für einen Trainingsschritt.
        """
        Executes a training step.

        Args:
            batch: tuple or batched tensors (previous observation, action, reward, new observation, terminated, truncated)  # Tuple oder batched Tensors (vorherige Beobachtung, Aktion, Belohnung, neue Beobachtung, beendet, abgeschnitten).

        Returns:
            dict: a dictionary containing one entry per metric you wish to log (e.g. for wandb)  # Wörterbuch mit einem Eintrag pro zu protokollierender Metrik (z. B. für wandb).
        """
        raise NotImplementedError  # Raise an error if method is not implemented in subclass.  # Fehler auslösen, wenn die Methode in der Unterklasse nicht implementiert ist.

    @abstractmethod  # Mark method as abstract (must be implemented by subclasses).  # Methode als abstrakt markieren (muss von Unterklassen implementiert werden).
    def get_actor(self):  # Abstract method to get the current actor.  # Abstrakte Methode, um den aktuellen Akteur zu erhalten.
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             ActorModule: current actor to be broadcast  # Der aktuelle Akteur, der gesendet werden soll.
        """
        raise NotImplementedError  # Raise an error if method is not implemented in subclass.  # Fehler auslösen, wenn die Methode in der Unterklasse nicht implementiert ist.
