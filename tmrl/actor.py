from abc import ABC, abstractmethod  # Importing abstract base class (ABC) and abstractmethod for defining abstract classes and methods.  # Importieren der abstrakten Basisklasse (ABC) und abstractmethod für das Definieren abstrakter Klassen und Methoden.
import torch  # Importing PyTorch library for tensor computation and model definition.  # Importieren der PyTorch-Bibliothek für Tensorberechnungen und Modelldefinition.
import pickle  # Importing pickle for serializing and deserializing Python objects.  # Importieren von pickle zum Serialisieren und Deserialisieren von Python-Objekten.

from tmrl.util import collate_torch  # Importing collate_torch from tmrl.util to batch observations into tensor format.  # Importieren von collate_torch aus tmrl.util zum Batching von Beobachtungen in Tensorformat.

__docformat__ = "google"  # Defining the docstring format for documentation.  # Festlegen des Dokumentationsformats für die Dokumentation.

class ActorModule(ABC):  # Defining an abstract base class for the ActorModule.  # Definieren einer abstrakten Basisklasse für das ActorModule.
    """
    Implement this interface for the RolloutWorker(s) to interact with your policy.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
       `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """  # Docstring explaining the purpose and requirements of the ActorModule class.  # Docstring, der den Zweck und die Anforderungen der ActorModule-Klasse erklärt.
    
    def __init__(self, observation_space, action_space):  # Constructor that initializes observation_space and action_space.  # Konstruktor, der observation_space und action_space initialisiert.
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
        """  # Arguments docstring for the constructor.  # Dokumentationsstring für den Konstruktor.
        
        self.observation_space = observation_space  # Setting the observation space for the actor.  # Festlegen des Beobachtungsraums für den Akteur.
        self.action_space = action_space  # Setting the action space for the actor.  # Festlegen des Aktionsraums für den Akteur.
        super().__init__()  # Calling the parent class constructor.  # Aufruf des Konstruktors der Elternklasse.

    def save(self, path):  # Method to save the ActorModule to disk.  # Methode zum Speichern des ActorModule auf der Festplatte.
        """
        Save your `ActorModule` on the hard drive.

        If not implemented, `save` defaults to `pickle.dump(obj=self, ...)`.

        You need to override this method if your `ActorModule` is not picklable.

        .. note::
           Everything needs to be saved into a single binary file.
           `tmrl` reads this file and transfers its content over network.

        Args:
            path (pathlib.Path): a filepath to save your `ActorModule` to
        """  # Docstring explaining how the save method works.  # Dokumentationsstring, der erklärt, wie die save-Methode funktioniert.
        
        with open(path, 'wb') as f:  # Opening the file at the given path in write-binary mode.  # Öffnen der Datei am angegebenen Pfad im Schreib-Binärmodus.
            pickle.dump(obj=self, file=f)  # Saving the ActorModule object to the file using pickle.  # Speichern des ActorModule-Objekts in der Datei mit pickle.

    def load(self, path, device):  # Method to load the ActorModule from disk.  # Methode zum Laden des ActorModule von der Festplatte.
        """
        Load and return an instance of your `ActorModule` from the hard drive.

        This method loads your `ActorModule` from the binary file saved by your implementation of `save`

        If not implemented, `load` defaults to returning this output of pickle.load(...).
        By default, the `device` argument is ignored (but you may want to use it in your implementation).

        You need to override this method if your ActorModule is not picklable.

        .. note::
           You can use this function to load attributes and return self, or you can return a new instance.

        Args:
            path (pathlib.Path): a filepath to load your ActorModule from
            device: device to load relevant attributes to (e.g., "cpu" or "cuda:0")

        Returns:
            ActorModule: An instance of your ActorModule
        """  # Docstring explaining how the load method works.  # Dokumentationsstring, der erklärt, wie die load-Methode funktioniert.
        
        with open(path, 'wb') as f:  # Opening the file at the given path in write-binary mode (which is a mistake, it should be read-binary).  # Öffnen der Datei am angegebenen Pfad im Schreib-Binärmodus (was ein Fehler ist, es sollte Lese-Binärmodus sein).
            res = pickle.load(file=f)  # Loading the ActorModule from the file using pickle.  # Laden des ActorModule aus der Datei mit pickle.
        return res  # Returning the loaded ActorModule.  # Zurückgeben des geladenen ActorModule.

    def to_device(self, device):  # Method to move the ActorModule to the specified device.  # Methode, um das ActorModule auf das angegebene Gerät zu verschieben.
        """
        Set the `ActorModule`'s relevant attributes to the designated device.

        By default, this method is a no-op and returns `self`.

        Args:
            device: the device where to move relevant attributes (e.g., `"cpu"` or `"cuda:0"`)

        Returns:
            an `ActorModule` whose relevant attributes are moved to `device` (can be `self`)
        """  # Docstring explaining how the to_device method works.  # Dokumentationsstring, der erklärt, wie die to_device-Methode funktioniert.
        
        return self  # By default, this method returns `self` without modifying anything.  # Standardmäßig gibt diese Methode `self` zurück, ohne etwas zu ändern.

    @abstractmethod  # Declaring this method as abstract so subclasses must implement it.  # Deklarieren dieser Methode als abstrakt, damit Unterklassen sie implementieren müssen.
    def act(self, obs, test=False):  # Abstract method to calculate an action from an observation.  # Abstrakte Methode, um eine Aktion aus einer Beobachtung zu berechnen.
        """
        Must compute an action from an observation.

        Args:
            obs (object): the observation
            test (bool): True at test time, False otherwise

        Returns:
            numpy.array: the computed action
        """  # Docstring explaining the purpose and arguments of the act method.  # Dokumentationsstring, der den Zweck und die Argumente der act-Methode erklärt.
        
        raise NotImplementedError  # This is an abstract method, so it raises an exception to indicate it must be implemented in a subclass.  # Dies ist eine abstrakte Methode, daher wird eine Ausnahme ausgelöst, um anzuzeigen, dass sie in einer Unterklasse implementiert werden muss.

    def act_(self, obs, test=False):  # Wrapper method to call the abstract act method.  # Wrapper-Methode, um die abstrakte act-Methode aufzurufen.
        return self.act(obs, test=test)  # Calling the act method.  # Aufruf der act-Methode.

class TorchActorModule(ActorModule, torch.nn.Module, ABC):  # Defining a subclass of ActorModule and torch.nn.Module for PyTorch-based implementation.  # Definieren einer Unterklasse von ActorModule und torch.nn.Module für eine PyTorch-basierte Implementierung.
    """
    Partial implementation of `ActorModule` as a `torch.nn.Module`.

    You can implement this instead of `ActorModule` when using PyTorch.
    `TorchActorModule` is a subclass of `torch.nn.Module` and may implement `forward()`.
    Typically, your implementation of `act()` can call `forward()` with gradients turned off.

    When using `TorchActorModule`, the `act` method receives observations collated on `device`,
    with an additional dimension corresponding to the batch size.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments (args or kwargs):
       `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """  # Docstring explaining the usage of the TorchActorModule class.  # Dokumentationsstring, der die Verwendung der TorchActorModule-Klasse erklärt.
    
    def __init__(self, observation_space, action_space, device="cpu"):  # Constructor initializing the class with device configuration.  # Konstruktor, der die Klasse mit Geräte-Konfiguration initialisiert.
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
            device: device where your model should live and where observations for `act` will be collated
        """  # Arguments docstring for the constructor.  # Dokumentationsstring für den Konstruktor.
        
        super().__init__(observation_space, action_space)  # Calling the parent class constructor.  # Aufruf des Konstruktors der Elternklasse.
        self.device = device  # Setting the device attribute.  # Setzen des Geräteattributs.

    def save(self, path):  # Method to save the state of the model using torch.save.  # Methode zum Speichern des Modellzustands mit torch.save.
        torch.save(self.state_dict(), path)  # Saving model parameters to the specified path.  # Speichern der Modellparameter an den angegebenen Pfad.

    def load(self, path, device):  # Method to load the state of the model from the specified path.  # Methode zum Laden des Modellzustands vom angegebenen Pfad.
        self.device = device  # Set the device for the loaded model.  # Das Gerät für das geladene Modell festlegen.
        self.load_state_dict(torch.load(path, map_location=self.device))  # Loading model parameters from the file.  # Laden der Modellparameter aus der Datei.
        return self  # Returning the loaded model instance.  # Zurückgeben der geladenen Modellinstanz.

    def act_(self, obs, test=False):  # Method to process observation and compute action.  # Methode zum Verarbeiten der Beobachtung und Berechnen der Aktion.
        obs = collate_torch([obs], device=self.device)  # Collating the observation into tensor format on the specified device.  # Sammeln der Beobachtung in Tensorformat auf dem angegebenen Gerät.
        with torch.no_grad():  # Disabling gradient calculation to save memory during inference.  # Deaktivieren der Gradientenberechnung, um Speicher während der Inferenz zu sparen.
            action = self.act(obs, test=test)  # Computing the action using the act method.  # Berechnen der Aktion mit der act-Methode.
        return action  # Returning the computed action.  # Zurückgeben der berechneten Aktion.

    def to(self, device):  # Method to move the model to a specific device.  # Methode, um das Modell auf ein bestimmtes Gerät zu verschieben.
        self.device = device  # Set the device for the model.  # Das Gerät für das Modell festlegen.
        return super().to(device=device)  # Calling the parent class's to method to move the model to the device.  # Aufruf der to-Methode der Elternklasse, um das Modell auf das Gerät zu verschieben.

    def to_device(self, device):  # Method to move the model to the specified device.  # Methode, um das Modell auf das angegebene Gerät zu verschieben.
        return self.to(device)  # Moving the model to the device by calling the to method.  # Verschieben des Modells auf das Gerät durch Aufruf der to-Methode.
