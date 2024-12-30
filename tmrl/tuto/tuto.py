import random  # Import the random module to generate random numbers.  # Importiert das Modul random, um Zufallszahlen zu erzeugen.
import numpy as np  # Import the numpy module, often used for numerical computations.  # Importiert das Modul numpy, das häufig für numerische Berechnungen verwendet wird.
import torch  # Import the PyTorch module for deep learning tasks.  # Importiert das PyTorch-Modul für Deep-Learning-Aufgaben.
from torch.optim import Adam  # Import the Adam optimizer from PyTorch.  # Importiert den Adam-Optimizer aus PyTorch.
from copy import deepcopy  # Import the deepcopy function to create deep copies of objects.  # Importiert die deepcopy-Funktion, um tiefe Kopien von Objekten zu erstellen.

from threading import Thread  # Import the Thread class to enable multi-threading.  # Importiert die Thread-Klasse, um Multi-Threading zu ermöglichen.

from tmrl.networking import Server, RolloutWorker, Trainer  # Import networking-related classes for server, worker, and training functionalities.  # Importiert netzwerkbezogene Klassen für Server-, Worker- und Trainingsfunktionen.
from tmrl.util import partial, cached_property  # Import utility functions like partial and cached_property for functional programming.  # Importiert Hilfsfunktionen wie partial und cached_property für funktionale Programmierung.
from tmrl.envs import GenericGymEnv  # Import the GenericGymEnv class for creating a generic Gym environment.  # Importiert die Klasse GenericGymEnv zum Erstellen einer generischen Gym-Umgebung.

from tmrl.actor import TorchActorModule  # Import the TorchActorModule class for defining an actor in reinforcement learning.  # Importiert die Klasse TorchActorModule zum Definieren eines Akteurs im Reinforcement Learning.
from tmrl.util import prod  # Import the prod function for calculating the product of a list.  # Importiert die Funktion prod zum Berechnen des Produkts einer Liste.

import tmrl.config.config_constants as cfg  # Import configuration constants from the tmrl config module.  # Importiert Konfigurationskonstanten aus dem tmrl-Konfigurationsmodul.
from tmrl.training_offline import TorchTrainingOffline  # Import offline training functionalities for reinforcement learning.  # Importiert Offline-Trainingsfunktionen für Reinforcement Learning.
from tmrl.training import TrainingAgent  # Import the TrainingAgent class for training an agent.  # Importiert die Klasse TrainingAgent zum Trainieren eines Agenten.
from tmrl.custom.utils.nn import copy_shared, no_grad  # Import helper functions for copying shared parameters and disabling gradient tracking.  # Importiert Hilfsfunktionen zum Kopieren geteilten Parametern und zum Deaktivieren der Gradientenverfolgung.

from tuto_envs.dummy_rc_drone_interface import DUMMY_RC_DRONE_CONFIG  # Import a dummy RC drone configuration.  # Importiert eine Dummy-RC-Drohnen-Konfiguration.

CRC_DEBUG = False  # Set a flag for CRC debugging.  # Setzt ein Flag für das Debugging des CRC.

# === Networking parameters ============================================================================================

security = None  # Set security parameter to None (no security).  # Setzt den Sicherheitsparameter auf None (keine Sicherheit).
password = cfg.PASSWORD  # Get the password from the configuration.  # Holt das Passwort aus der Konfiguration.

server_ip = "127.0.0.1"  # Define the server IP address (localhost).  # Definiert die Server-IP-Adresse (localhost).
server_port = 6666  # Define the server port.  # Definiert den Server-Port.

# === Server ===========================================================================================================

if __name__ == "__main__":  # Check if the script is being run as the main program.  # Überprüft, ob das Skript als Hauptprogramm ausgeführt wird.
    my_server = Server(security=security, password=password, port=server_port)  # Initialize the server with security and password.  # Initialisiert den Server mit Sicherheit und Passwort.

# === Environment ======================================================================================================

# rtgym interface:

my_config = DUMMY_RC_DRONE_CONFIG  # Set up the configuration for the drone environment.  # Stellt die Konfiguration für die Drohnenumgebung ein.

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})  # Create a partial function for the environment class with specific config.  # Erzeugt eine partielle Funktion für die Umgebungs-Klasse mit einer spezifischen Konfiguration.

# Observation and action space:

dummy_env = env_cls()  # Initialize the environment.  # Initialisiert die Umgebung.
act_space = dummy_env.action_space  # Get the action space of the environment.  # Holt den Aktionsraum der Umgebung.
obs_space = dummy_env.observation_space  # Get the observation space of the environment.  # Holt den Beobachtungsraum der Umgebung.

print(f"action space: {act_space}")  # Print the action space.  # Gibt den Aktionsraum aus.
print(f"observation space: {obs_space}")  # Print the observation space.  # Gibt den Beobachtungsraum aus.

# === Worker ===========================================================================================================

import torch.nn.functional as F  # Import PyTorch's functional module for neural network operations.  # Importiert das funktionale Modul von PyTorch für neuronale Netzwerkoperationen.

# ActorModule:

LOG_STD_MAX = 2  # Maximum value for the log of the standard deviation.  # Maximale Wert für den Logarithmus der Standardabweichung.
LOG_STD_MIN = -20  # Minimum value for the log of the standard deviation.  # Minimale Wert für den Logarithmus der Standardabweichung.

# Define a multi-layer perceptron (MLP) function for constructing neural networks:

def mlp(sizes, activation, output_activation=torch.nn.Identity):  # Function to create a multilayer perceptron (MLP) model.  # Funktion zum Erstellen eines Multi-Layer-Perceptron (MLP)-Modells.
    layers = []  # Initialize an empty list to hold the layers.  # Initialisiert eine leere Liste, um die Schichten zu halten.
    for j in range(len(sizes) - 1):  # Loop over the sizes of layers.  # Schleife über die Größen der Schichten.
        act = activation if j < len(sizes) - 2 else output_activation  # Select the activation function for each layer.  # Wählt die Aktivierungsfunktion für jede Schicht aus.
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]  # Add the layer with its activation function.  # Fügt die Schicht mit ihrer Aktivierungsfunktion hinzu.
    return torch.nn.Sequential(*layers)  # Return a sequential container of the layers.  # Gibt einen sequenziellen Container der Schichten zurück.

# Define the MyActorModule class which is an implementation of the actor module in reinforcement learning:

class MyActorModule(TorchActorModule):  # Define a class inheriting from TorchActorModule.  # Definiert eine Klasse, die von TorchActorModule erbt.
    """
    Directly adapted from the Spinup implementation of SAC  # Direkt adaptiert von der Spinup-Implementierung von SAC.
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):  # Initialize the actor module.  # Initialisiert das Aktormodul.
        super().__init__(observation_space, action_space)  # Call the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)  # Calculate the dimensionality of the observation space.  # Berechnet die Dimensionalität des Beobachtungsraums.
        dim_act = action_space.shape[0]  # Get the dimensionality of the action space.  # Holt die Dimensionalität des Aktionsraums.
        act_limit = action_space.high[0]  # Get the action space's upper bound.  # Holt die obere Grenze des Aktionsraums.
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)  # Create the neural network.  # Erstellt das neuronale Netzwerk.
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)  # Define the layer for the mean action.  # Definiert die Schicht für die mittlere Aktion.
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)  # Define the layer for the log standard deviation.  # Definiert die Schicht für den Logarithmus der Standardabweichung.
        self.act_limit = act_limit  # Set the action limit.  # Setzt die Aktionsgrenze.

    def forward(self, obs, test=False, with_logprob=True):  # Define the forward pass of the actor.  # Definiert den Vorwärtspass des Akteurs.
        net_out = self.net(torch.cat(obs, -1))  # Pass observations through the network.  # Führt Beobachtungen durch das Netzwerk.
        mu = self.mu_layer(net_out)  # Get the mean action.  # Holt die mittlere Aktion.
        log_std = self.log_std_layer(net_out)  # Get the log standard deviation.  # Holt den Logarithmus der Standardabweichung.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clamp the log standard deviation to a range.  # Beschränkt den Logarithmus der Standardabweichung auf einen Bereich.
        std = torch.exp(log_std)  # Compute the standard deviation.  # Berechnet die Standardabweichung.
        pi_distribution = torch.distributions.normal.Normal(mu, std)  # Create a normal distribution with mean and std.  # Erstellt eine Normalverteilung mit Mittelwert und Standardabweichung.
        if test:  # If in test mode, use the mean as the action.  # Wenn im Testmodus, wird der Mittelwert als Aktion verwendet.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()  # Sample an action from the distribution.  # Zieht eine Aktion aus der Verteilung.
        if with_logprob:  # If log probability is required, calculate it.  # Wenn Log-Wahrscheinlichkeit erforderlich ist, wird sie berechnet.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Calculate the log probability of the action.  # Berechnet die Log-Wahrscheinlichkeit der Aktion.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # Adjust the log probability.  # Passt die Log-Wahrscheinlichkeit an.
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)  # Apply the tanh function to the action.  # Wendet die tanh-Funktion auf die Aktion an.
        pi_action = self.act_limit * pi_action  # Scale the action by the action limit.  # Skaliert die Aktion mit der Aktionsgrenze.
        pi_action = pi_action.squeeze()  # Remove extra dimensions from the action tensor.  # Entfernt zusätzliche Dimensionen vom Aktions-Tensor.
        return pi_action, logp_pi  # Return the action and log probability.  # Gibt die Aktion und Log-Wahrscheinlichkeit zurück.

    def act(self, obs, test=False):  # Define the act function to get an action from the actor.  # Definiert die Aktionsfunktion, um eine Aktion vom Akteur zu erhalten.
        with torch.no_grad():  # Disable gradient tracking during action computation.  # Deaktiviert die Gradientenverfolgung während der Berechnung der Aktion.
            a, _ = self.forward(obs, test, False)  # Get the action by performing a forward pass.  # Holt die Aktion durch einen Vorwärtspass.
            return a.cpu().numpy()  # Return the action as a NumPy array.  # Gibt die Aktion als NumPy-Array zurück.


actor_module_cls = partial(MyActorModule)  # Create a partial function for creating the actor module.  # Erzeugt eine partielle Funktion zum Erstellen des Akteursmoduls.


# Sample compression

def my_sample_compressor(prev_act, obs, rew, terminated, truncated, info):  # Defines the function to compress samples for network transfer.  # Definiert die Funktion, um Proben für die Netzübertragung zu komprimieren.
    """  # Docstring to describe the purpose of the function.  # Dokumentation, um den Zweck der Funktion zu beschreiben.
    Compresses samples before sending over network.  # Komprimiert Proben vor der Übertragung über das Netzwerk.
    This function creates the sample that will actually be stored in local buffers for networking.  # Diese Funktion erstellt die Probe, die tatsächlich in lokalen Puffern für das Netzwerk gespeichert wird.
    This is to compress the sample before sending it over the Internet/local network.  # Dies dient der Kompression der Probe, bevor sie über das Internet/das lokale Netzwerk gesendet wird.
    Buffers of such samples will be given as input to the append() method of the memory.  # Puffer solcher Proben werden als Eingabe an die append()-Methode des Speichers übergeben.
    When you implement such compressor, you must implement a corresponding decompressor.  # Wenn Sie einen solchen Kompressor implementieren, müssen Sie einen entsprechenden Dekompressor implementieren.
    This decompressor is the append() or get_transition() method of the memory.  # Dieser Dekompressor ist die append()- oder get_transition()-Methode des Speichers.
   
  

    Args:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    Returns:
        prev_act_mod: compressed prev_act
        obs_mod: compressed obs
        rew_mod: compressed rew
        terminated_mod: compressed terminated
        truncated_mod: compressed truncated
        info_mod: compressed info
    """
 # End of docstring.  # Ende der Dokumentation.
    
    prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod = prev_act, obs, rew, terminated, truncated, info  # Assigns input parameters to variables for further modification.  # Weist Eingabeparameter Variablen zu, die weiter modifiziert werden.

    obs_mod = obs_mod[:4]  # here we remove the action buffer from observations  # Hier entfernen wir den Aktionspuffer aus den Beobachtungen.
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod  # Returns the compressed values.  # Gibt die komprimierten Werte zurück.

sample_compressor = my_sample_compressor  # Assigns the function to a variable for later use.  # Weist die Funktion einer Variablen für die spätere Verwendung zu.


# Device
device = "cpu"  # Specifies that the computation will be performed on the CPU.  # Legt fest, dass die Berechnungen auf der CPU durchgeführt werden.

# Networking
max_samples_per_episode = 1000  # Sets the maximum number of samples collected per episode.  # Legt die maximale Anzahl von Stichproben pro Episode fest.

# Model files
my_run_name = "tutorial"  # Name of the current run, used to save model data.  # Name des aktuellen Laufs, der zum Speichern von Modelldaten verwendet wird.
weights_folder = cfg.WEIGHTS_FOLDER  # Path to the folder where model weights are stored.  # Pfad zum Ordner, in dem die Modellgewichte gespeichert sind.

model_path = str(weights_folder / (my_run_name + ".tmod"))  # Defines the path where the model will be saved, using the run name.  # Definiert den Pfad, an dem das Modell gespeichert wird, unter Verwendung des Laufnamens.
model_path_history = str(weights_folder / (my_run_name + "_"))  # Path for saving model history files.  # Pfad zum Speichern von Modellhistorien-Dateien.
model_history = 10  # Defines the number of past models to keep.  # Definiert, wie viele vergangene Modelle aufbewahrt werden sollen.

# Instantiation of the RolloutWorker object:
if __name__ == "__main__":  # Checks if the script is being run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    my_worker = RolloutWorker(  # Creates a new instance of the RolloutWorker class.  # Erstellt eine neue Instanz der RolloutWorker-Klasse.
        env_cls=env_cls,  # Class for the environment used by the worker.  # Klasse für die Umgebung, die vom Worker verwendet wird.
        actor_module_cls=actor_module_cls,  # Class for the actor module in the worker.  # Klasse für das Actor-Modul im Worker.
        sample_compressor=sample_compressor,  # The sample compressor used in the worker.  # Der Sample-Kompressor, der im Worker verwendet wird.
        device=device,  # Device to run the worker (CPU in this case).  # Gerät, auf dem der Worker ausgeführt wird (in diesem Fall CPU).
        server_ip=server_ip,  # IP address of the server for communication.  # IP-Adresse des Servers für die Kommunikation.
        server_port=server_port,  # Port of the server for communication.  # Port des Servers für die Kommunikation.
        password=password,  # Password for server access.  # Passwort für den Serverzugriff.
        max_samples_per_episode=max_samples_per_episode,  # Maximum samples per episode.  # Maximale Stichproben pro Episode.
        model_path=model_path,  # Path to the model for the worker.  # Pfad zum Modell für den Worker.
        model_path_history=model_path_history,  # Path to the model history.  # Pfad zur Modellhistorie.
        model_history=model_history,  # Number of model versions to keep.  # Anzahl der Modellversionen, die behalten werden sollen.
        crc_debug=CRC_DEBUG)  # CRC debug flag for validation.  # CRC-Debug-Flag zur Validierung.

    # my_worker.run(test_episode_interval=10)  # this would block the script here!  # Start the worker, but the script would be blocked.  # Startet den Worker, aber das Skript wird hier blockiert.

# === Trainer ==========================================================================================================

# --- Networking and files ---
weights_folder = cfg.WEIGHTS_FOLDER  # Path to the weights folder.  # Pfad zum Ordner für Gewichte.
checkpoints_folder = cfg.CHECKPOINTS_FOLDER  # Path to the checkpoints folder.  # Pfad zum Ordner für Checkpoints.
my_run_name = "tutorial"  # Run name used for saving training progress.  # Name des Laufs zum Speichern des Trainingsfortschritts.

model_path = str(weights_folder / (my_run_name + "_t.tmod"))  # Path to save the model during training.  # Pfad zum Speichern des Modells während des Trainings.
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))  # Path to save the checkpoint during training.  # Pfad zum Speichern des Checkpoints während des Trainings.

# --- TrainingOffline ---
# Dummy environment:
env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})  # Creates an environment for training with specific configurations.  # Erstellt eine Umgebung für das Training mit spezifischen Konfigurationen.
# env_cls = (observation_space, action_space)  # Placeholder for environment class (not used here).  # Platzhalter für die Umweltklasse (hier nicht verwendet).

# Memory:
from tmrl.memory import TorchMemory  # Import of TorchMemory for handling memory in training.  # Import von TorchMemory zur Verwaltung des Gedächtnisses im Training.

def last_true_in_list(li):  # Function to find the last "True" element in a list.  # Funktion, um das letzte "True"-Element in einer Liste zu finden.
    """
    Returns the index of the last True element in list li, or None.  # Gibt den Index des letzten True-Elements in der Liste li zurück oder None.
    """
    for i in reversed(range(len(li))):  # Loop through the list in reverse order.  # Durchläuft die Liste in umgekehrter Reihenfolge.
        if li[i]:  # If the element is True, return its index.  # Wenn das Element True ist, gibt es dessen Index zurück.
            return i  # Return the index of the last True element.  # Gibt den Index des letzten True-Elements zurück.
    return None  # If no True element is found, return None.  # Wenn kein True-Element gefunden wird, gibt es None zurück.


class MyMemory(TorchMemory):  # Define a class 'MyMemory' that inherits from 'TorchMemory' class.  # Definiere eine Klasse 'MyMemory', die von der Klasse 'TorchMemory' erbt.
    def __init__(self,  # Define the initialization method for the class.  # Definiere die Initialisierungsmethode für die Klasse.
                 act_buf_len=None,  # Optional parameter for the length of the action buffer. Default is None.  # Optionaler Parameter für die Länge des Aktionspuffers. Standard ist None.
                 device=None,  # Optional parameter for the device to run the operations (e.g., CPU, GPU). Default is None.  # Optionaler Parameter für das Gerät, auf dem die Operationen ausgeführt werden (z.B. CPU, GPU). Standard ist None.
                 nb_steps=None,  # Optional parameter for the number of steps. Default is None.  # Optionaler Parameter für die Anzahl der Schritte. Standard ist None.
                 sample_preprocessor: callable = None,  # Optional callable function for preprocessing samples. Default is None.  # Optional eine callable Funktion zum Vorverarbeiten von Stichproben. Standard ist None.
                 memory_size=1000000,  # Default memory size set to 1 million.  # Standardmäßige Speichergröße ist auf 1 Million gesetzt.
                 batch_size=32,  # Default batch size is 32.  # Standard-Batch-Größe ist 32.
                 dataset_path=""):  # Default path to the dataset, set to an empty string.  # Standardpfad zum Datensatz, auf einen leeren String gesetzt.
        
        self.act_buf_len = act_buf_len  # Set the action buffer length attribute.  # Setze das Attribut für die Länge des Aktionspuffers.

        super().__init__(device=device,  # Call the initializer of the parent class 'TorchMemory', passing the device parameter.  # Rufe den Initialisierer der Elternklasse 'TorchMemory' auf und übergebe den Gerät-Parameter.
                         nb_steps=nb_steps,  # Pass the number of steps parameter to the parent class initializer.  # Übergebe den Parameter für die Anzahl der Schritte an den Initialisierer der Elternklasse.
                         sample_preprocessor=sample_preprocessor,  # Pass the sample preprocessor function to the parent class initializer.  # Übergebe die Funktion zum Vorverarbeiten von Stichproben an den Initialisierer der Elternklasse.
                         memory_size=memory_size,  # Pass the memory size to the parent class initializer.  # Übergebe die Speichergröße an den Initialisierer der Elternklasse.
                         batch_size=batch_size,  # Pass the batch size to the parent class initializer.  # Übergebe die Batch-Größe an den Initialisierer der Elternklasse.
                         dataset_path=dataset_path,  # Pass the dataset path to the parent class initializer.  # Übergebe den Datensatzpfad an den Initialisierer der Elternklasse.
                         crc_debug=CRC_DEBUG)  # Pass the CRC_DEBUG parameter to the parent class initializer.  # Übergebe den Parameter CRC_DEBUG an den Initialisierer der Elternklasse.

    def append_buffer(self, buffer):  # Method to append data to the buffer.  # Methode zum Hinzufügen von Daten zum Puffer.
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod) samples
        """  # Explanation of the buffer content.  # Erklärung des Inhalts des Puffers.

        # decompose compressed samples into their relevant components:  # Decompose compressed samples into individual variables.  # Zerlege komprimierte Proben in einzelne Variablen.
        
        list_action = [b[0] for b in buffer.memory]  # Extract actions from buffer memory.  # Extrahiere Aktionen aus dem Puffer-Speicher.
        list_x_position = [b[1][0] for b in buffer.memory]  # Extract X positions from buffer memory.  # Extrahiere X-Positionen aus dem Puffer-Speicher.
        list_y_position = [b[1][1] for b in buffer.memory]  # Extract Y positions from buffer memory.  # Extrahiere Y-Positionen aus dem Puffer-Speicher.
        list_x_target = [b[1][2] for b in buffer.memory]  # Extract X targets from buffer memory.  # Extrahiere X-Ziele aus dem Puffer-Speicher.
        list_y_target = [b[1][3] for b in buffer.memory]  # Extract Y targets from buffer memory.  # Extrahiere Y-Ziele aus dem Puffer-Speicher.
        list_reward = [b[2] for b in buffer.memory]  # Extract rewards from buffer memory.  # Extrahiere Belohnungen aus dem Puffer-Speicher.
        list_terminated = [b[3] for b in buffer.memory]  # Extract termination flags from buffer memory.  # Extrahiere Beendigungsmarken aus dem Puffer-Speicher.
        list_truncated = [b[4] for b in buffer.memory]  # Extract truncation flags from buffer memory.  # Extrahiere Abschneide-Markierungen aus dem Puffer-Speicher.
        list_info = [b[5] for b in buffer.memory]  # Extract additional information from buffer memory.  # Extrahiere zusätzliche Informationen aus dem Puffer-Speicher.
        list_done = [b[3] or b[4] for b in buffer.memory]  # Determine if the episode is done by checking termination or truncation flags.  # Bestimme, ob die Episode beendet ist, indem die Beendigungs- oder Abschneide-Markierungen überprüft werden.

        # append to self.data in some arbitrary way:  # Append the extracted lists to the data attribute.  # Füge die extrahierten Listen der Daten-Eigenschaft hinzu.

        if self.__len__() > 0:  # If the memory already has data, append to it.  # Wenn der Speicher bereits Daten enthält, füge neue hinzu.
            self.data[0] += list_action  # Append actions to the data.  # Füge Aktionen zu den Daten hinzu.
            self.data[1] += list_x_position  # Append X positions to the data.  # Füge X-Positionen zu den Daten hinzu.
            self.data[2] += list_y_position  # Append Y positions to the data.  # Füge Y-Positionen zu den Daten hinzu.
            self.data[3] += list_x_target  # Append X targets to the data.  # Füge X-Ziele zu den Daten hinzu.
            self.data[4] += list_y_target  # Append Y targets to the data.  # Füge Y-Ziele zu den Daten hinzu.
            self.data[5] += list_reward  # Append rewards to the data.  # Füge Belohnungen zu den Daten hinzu.
            self.data[6] += list_terminated  # Append termination flags to the data.  # Füge Beendigungsmarken zu den Daten hinzu.
            self.data[7] += list_info  # Append additional information to the data.  # Füge zusätzliche Informationen zu den Daten hinzu.
            self.data[8] += list_truncated  # Append truncation flags to the data.  # Füge Abschneide-Markierungen zu den Daten hinzu.
            self.data[9] += list_done  # Append done flags to the data.  # Füge erledigte Markierungen zu den Daten hinzu.
        else:  # If the memory is empty, initialize it with the new data.  # Wenn der Speicher leer ist, initialisiere ihn mit den neuen Daten.
            self.data.append(list_action)  # Initialize the memory with actions.  # Initialisiere den Speicher mit Aktionen.
            self.data.append(list_x_position)  # Initialize the memory with X positions.  # Initialisiere den Speicher mit X-Positionen.
            self.data.append(list_y_position)  # Initialize the memory with Y positions.  # Initialisiere den Speicher mit Y-Positionen.
            self.data.append(list_x_target)  # Initialize the memory with X targets.  # Initialisiere den Speicher mit X-Zielen.
            self.data.append(list_y_target)  # Initialize the memory with Y targets.  # Initialisiere den Speicher mit Y-Zielen.
            self.data.append(list_reward)  # Initialize the memory with rewards.  # Initialisiere den Speicher mit Belohnungen.
            self.data.append(list_terminated)  # Initialize the memory with termination flags.  # Initialisiere den Speicher mit Beendigungsmarken.
            self.data.append(list_info)  # Initialize the memory with additional information.  # Initialisiere den Speicher mit zusätzlichen Informationen.
            self.data.append(list_truncated)  # Initialize the memory with truncation flags.  # Initialisiere den Speicher mit Abschneide-Markierungen.
            self.data.append(list_done)  # Initialize the memory with done flags.  # Initialisiere den Speicher mit erledigten Markierungen.

        # trim self.data in some arbitrary way when self.__len__() > self.memory_size:  # Trim the data when it exceeds the memory size.  # Kürze die Daten, wenn ihre Länge größer als die Speicherkapazität ist.
        
        to_trim = self.__len__() - self.memory_size  # Calculate how much data needs to be trimmed.  # Berechne, wie viele Daten gekürzt werden müssen.
        if to_trim > 0:  # If trimming is necessary, perform it.  # Wenn eine Kürzung notwendig ist, führe sie durch.
            self.data[0] = self.data[0][to_trim:]  # Trim the actions list.  # Kürze die Aktionsliste.
            self.data[1] = self.data[1][to_trim:]  # Trim the X positions list.  # Kürze die X-Positionen-Liste.
            self.data[2] = self.data[2][to_trim:]  # Trim the Y positions list.  # Kürze die Y-Positionen-Liste.
            self.data[3] = self.data[3][to_trim:]  # Trim the X targets list.  # Kürze die X-Ziele-Liste.
            self.data[4] = self.data[4][to_trim:]  # Trim the Y targets list.  # Kürze die Y-Ziele-Liste.
            self.data[5] = self.data[5][to_trim:]  # Trim the rewards list.  # Kürze die Belohnungen-Liste.
            self.data[6] = self.data[6][to_trim:]  # Trim the termination flags list.  # Kürze die Beendigungsmarken-Liste.
            self.data[7] = self.data[7][to_trim:]  # Trim the additional information list.  # Kürze die zusätzliche Informationen-Liste.
            self.data[8] = self.data[8][to_trim:]  # Trim the truncation flags list.  # Kürze die Abschneide-Markierungen-Liste.
            self.data[9] = self.data[9][to_trim:]  # Trim the done flags list.  # Kürze die erledigten Markierungen-Liste.

def __len__(self):  # Returns the length of the dataset or buffer.  # Gibt die Länge des Datensatzes oder Puffers zurück.
    if len(self.data) == 0:  # Checks if data is empty.  # Überprüft, ob die Daten leer sind.
        return 0  # self.data is empty  # self.data ist leer.
    result = len(self.data[0]) - self.act_buf_len - 1  # Calculates number of samples that can be reconstructed.  # Berechnet die Anzahl der Proben, die rekonstruiert werden können.
    if result < 0:  # Checks if there are not enough samples to reconstruct the action buffer.  # Überprüft, ob nicht genügend Proben vorhanden sind, um den Aktionspuffer zu rekonstruieren.
        return 0  # not enough samples to reconstruct the action buffer  # nicht genügend Proben, um den Aktionspuffer zu rekonstruieren.
    else:  # Otherwise, return the result.  # Andernfalls wird das Ergebnis zurückgegeben.
        return result  # we can reconstruct that many samples  # wir können so viele Proben rekonstruieren.

def get_transition(self, item):  # Function to get a specific transition.  # Funktion, um eine spezifische Übergang zu erhalten.
    """
    Args:
        item: int: indice of the transition that the Trainer wants to sample  # item: int: Index des Übergangs, den der Trainer abfragen möchte.
    Returns:
        full transition: (last_obs, new_act, rew, new_obs, terminated, truncated, info)  # vollständiger Übergang: (letzte Beobachtung, neue Aktion, Belohnung, neue Beobachtung, beendet, abgeschnitten, Info)
    """
    while True:  # Loop to modify item in edge cases.  # Schleife, um das Element in Randfällen zu modifizieren.
        # if item corresponds to a transition from a terminal state to a reset state  # wenn das Element einem Übergang von einem Endzustand zu einem Zurücksetz-Zustand entspricht
        if self.data[9][item + self.act_buf_len - 1]:  # Check if transition is from terminal to reset state.  # Überprüft, ob der Übergang von einem Endzustand zum Zurücksetz-Zustand erfolgt.
            # this wouldn't make sense in RL, so we replace item by a neighbour transition  # Das wäre in RL nicht sinnvoll, daher wird das Element durch einen benachbarten Übergang ersetzt.
            if item == 0:  # if first item of the buffer  # wenn es das erste Element des Puffers ist
                item += 1  # move to the next item  # zum nächsten Element wechseln
            elif item == self.__len__() - 1:  # if last item of the buffer  # wenn es das letzte Element des Puffers ist
                item -= 1  # move to the previous item  # zum vorherigen Element wechseln
            elif random.random() < 0.5:  # otherwise, sample randomly  # andernfalls zufällig auswählen
                item += 1  # move to the next item  # zum nächsten Element wechseln
            else:  # if random value is greater than 0.5  # wenn der zufällige Wert größer als 0,5 ist
                item -= 1  # move to the previous item  # zum vorherigen Element wechseln

        idx_last = item + self.act_buf_len - 1  # index of previous observation  # Index der vorherigen Beobachtung
        idx_now = item + self.act_buf_len  # index of new observation  # Index der neuen Beobachtung

        # rebuild the action buffer of both observations:  # Rekonstruiere den Aktionspuffer beider Beobachtungen.
        actions = self.data[0][item:(item + self.act_buf_len + 1)]  # Get actions from the data.  # Hole Aktionen aus den Daten.
        last_act_buf = actions[:-1]  # action buffer of previous observation  # Aktionspuffer der vorherigen Beobachtung
        new_act_buf = actions[1:]  # action buffer of new observation  # Aktionspuffer der neuen Beobachtung

        # correct the action buffer when it goes over a reset transition:  # Korrigiere den Aktionspuffer, wenn er über eine Zurücksetz-Transition geht.
        # (NB: we have eliminated the case where the transition *is* the reset transition)  # (Anmerkung: Wir haben den Fall eliminiert, bei dem der Übergang *die* Zurücksetz-Transition ist)
        eoe = last_true_in_list(self.data[9][item:(item + self.act_buf_len)])  # the last one is not important  # Das letzte Element ist nicht wichtig.
        if eoe is not None:  # If the reset transition is found.  # Wenn die Zurücksetz-Transition gefunden wurde.
            # either one or both action buffers are passing over a reset transition  # entweder geht einer oder beide Aktionspuffer über eine Zurücksetz-Transition.
            if eoe < self.act_buf_len - 1:  # if the last action buffer is affected  # wenn der letzte Aktionspuffer betroffen ist
                # last_act_buf is concerned  # Der last_act_buf ist betroffen.
                if item == 0:  # edge case where previous action has been discarded  # Randfall, bei dem die vorherige Aktion verworfen wurde
                    item = random.randint(1, self.__len__())  # Sample another item  # Ein anderes Element zufällig auswählen
                    continue  # Continue to the next iteration  # Fortfahren mit der nächsten Iteration
                last_act_buf_eoe = eoe  # End of episode for last action buffer  # End of episode für den letzten Aktionspuffer
                # replace everything before last_act_buf_eoe by the previous action  # Ersetze alles vor last_act_buf_eoe durch die vorherige Aktion
                prev_act = self.data[0][item - 1]  # Get previous action  # Hole die vorherige Aktion
                for idx in range(last_act_buf_eoe + 1):  # Iterate over the action buffer  # Iteriere über den Aktionspuffer
                    act_tmp = last_act_buf[idx]  # Store temporary action  # Temporäre Aktion speichern
                    last_act_buf[idx] = prev_act  # Replace with previous action  # Mit der vorherigen Aktion ersetzen
                    prev_act = act_tmp  # Set previous action to current  # Setze die vorherige Aktion auf die aktuelle
            if eoe > 0:  # if the new action buffer is concerned  # wenn der neue Aktionspuffer betroffen ist
                new_act_buf_eoe = eoe - 1  # Set the new action buffer end of episode  # Setze das Ende der Episode des neuen Aktionspuffers
                # replace everything before new_act_buf_eoe by the previous action  # Ersetze alles vor new_act_buf_eoe durch die vorherige Aktion
                prev_act = self.data[0][item]  # Get previous action for new buffer  # Hole die vorherige Aktion für den neuen Puffer
                for idx in range(new_act_buf_eoe + 1):  # Iterate over the new action buffer  # Iteriere über den neuen Aktionspuffer
                    act_tmp = new_act_buf[idx]  # Store temporary action  # Temporäre Aktion speichern
                    new_act_buf[idx] = prev_act  # Replace with previous action  # Mit der vorherigen Aktion ersetzen
                    prev_act = act_tmp  # Set previous action to current  # Setze die vorherige Aktion auf die aktuelle

        # rebuild the previous observation:  # Rekonstruiere die vorherige Beobachtung:
        last_obs = (self.data[1][idx_last],  # x position  # x Position
                    self.data[2][idx_last],  # y position  # y Position
                    self.data[3][idx_last],  # x target  # x Ziel
                    self.data[4][idx_last],  # y target  # y Ziel
                    *last_act_buf)  # action buffer  # Aktionspuffer

        # rebuild the new observation:  # Rekonstruiere die neue Beobachtung:
        new_obs = (self.data[1][idx_now],  # x position  # x Position
                   self.data[2][idx_now],  # y position  # y Position
                   self.data[3][idx_now],  # x target  # x Ziel
                   self.data[4][idx_now],  # y target  # y Ziel
                   *new_act_buf)  # action buffer  # Aktionspuffer

        # other components of the transition:  # Andere Komponenten des Übergangs:
        new_act = self.data[0][idx_now]  # action  # Aktion
        rew = np.float32(self.data[5][idx_now])  # reward  # Belohnung
        terminated = self.data[6][idx_now]  # terminated signal  # Beendet-Signal
        truncated = self.data[8][idx_now]  # truncated signal  # Abgeschnitten-Signal
        info = self.data[7][idx_now]  # info dictionary  # Info-Wörterbuch

        break  # Exit the loop  # Schleife verlassen

    return last_obs, new_act, rew, new_obs, terminated, truncated, info  # Return the full transition  # Gibt den vollständigen Übergang zurück.

memory_cls = partial(MyMemory,  # Partial function to create memory class with specific configuration.  # Partielle Funktion zum Erstellen der Speicherklasse mit spezifischer Konfiguration.
                     act_buf_len=my_config["act_buf_len"])  # Set the action buffer length from configuration  # Setze die Länge des Aktionspuffers aus der Konfiguration.



























# Training agent:

class MyCriticModule(torch.nn.Module):  # Defines a Critic module for Q-value approximation.  # Definiert ein Critic-Modul zur Q-Wert-Näherung.
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):  # Initialization method.  # Initialisierungsmethode.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)  # Calculates the observation dimension.  # Berechnet die Beobachtungsdimension.
        act_dim = action_space.shape[0]  # Gets the action dimension.  # Bestimmt die Aktionsdimension.
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)  # Creates a multi-layer perceptron (MLP) for Q-value estimation.  # Erstellt ein Multi-Layer-Perceptron (MLP) zur Q-Wert-Schätzung.

    def forward(self, obs, act):  # Forward pass through the Critic network.  # Vorwärtsdurchgang durch das Critic-Netzwerk.
        x = torch.cat((*obs, act), -1)  # Concatenates the observation and action tensors.  # Verbindet die Beobachtungs- und Aktions-Tensoren.
        q = self.q(x)  # Computes the Q-value from the concatenated input.  # Berechnet den Q-Wert aus dem verbundenen Eingabewert.
        return torch.squeeze(q, -1)  # Removes the singleton dimension.  # Entfernt die Einzel-Dimension.

class MyActorCriticModule(torch.nn.Module):  # Defines an Actor-Critic module.  # Definiert ein Actor-Critic-Modul.
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):  # Initialization of the Actor-Critic model.  # Initialisierung des Actor-Critic-Modells.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        self.actor = MyActorModule(observation_space, action_space, hidden_sizes, activation)  # Initializes the Actor module.  # Initialisiert das Actor-Modul.
        self.q1 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)  # Initializes the first Critic module.  # Initialisiert das erste Critic-Modul.
        self.q2 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)  # Initializes the second Critic module.  # Initialisiert das zweite Critic-Modul.

import itertools  # Imports itertools for efficient iteration.  # Importiert itertools für effiziente Iterationen.

class MyTrainingAgent(TrainingAgent):  # Defines a custom training agent based on the TrainingAgent class.  # Definiert einen benutzerdefinierten Trainings-Agenten basierend auf der TrainingAgent-Klasse.
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))  # Cached property for the model without gradients.  # Zwischengespeicherte Eigenschaft für das Modell ohne Gradienten.

    def __init__(self, observation_space=None, action_space=None, device=None, model_cls=MyActorCriticModule,  # Initialization of the training agent.  # Initialisierung des Trainings-Agenten.
                 gamma=0.99, polyak=0.995, alpha=0.2, lr_actor=1e-3, lr_critic=1e-3, lr_entropy=1e-3, 
                 learn_entropy_coef=True, target_entropy=None):  # Initialization parameters for training.  # Initialisierungsparameter für das Training.
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)  # Calls the parent constructor.  # Ruft den Elternkonstruktor auf.
        model = model_cls(observation_space, action_space)  # Initializes the model with the specified class.  # Initialisiert das Modell mit der angegebenen Klasse.
        self.model = model.to(device)  # Moves the model to the specified device (e.g., CPU/GPU).  # Verschiebt das Modell auf das angegebene Gerät (z.B. CPU/GPU).
        self.model_target = no_grad(deepcopy(self.model))  # Creates a target model without gradients.  # Erstellt ein Zielmodell ohne Gradienten.
        self.gamma = gamma  # Sets the discount factor for future rewards.  # Setzt den Abzinsungsfaktor für zukünftige Belohnungen.
        self.polyak = polyak  # Sets the Polyak averaging factor for the target critic.  # Setzt den Polyak-Averaging-Faktor für den Ziel-Critic.
        self.alpha = alpha  # Sets the entropy coefficient for the SAC algorithm.  # Setzt den Entropie-Koeffizienten für den SAC-Algorithmus.
        self.lr_actor = lr_actor  # Sets the learning rate for the actor network.  # Setzt die Lernrate für das Actor-Netzwerk.
        self.lr_critic = lr_critic  # Sets the learning rate for the critic network.  # Setzt die Lernrate für das Critic-Netzwerk.
        self.lr_entropy = lr_entropy  # Sets the learning rate for the entropy coefficient.  # Setzt die Lernrate für den Entropie-Koeffizienten.
        self.learn_entropy_coef = learn_entropy_coef  # Determines whether to learn the entropy coefficient.  # Bestimmt, ob der Entropie-Koeffizient gelernt werden soll.
        self.target_entropy = target_entropy  # Sets the target entropy for SAC v2.  # Setzt die Ziel-Entropie für SAC v2.
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())  # Combines the parameters of both Q networks.  # Kombiniert die Parameter beider Q-Netzwerke.
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)  # Optimizer for the actor network.  # Optimierer für das Actor-Netzwerk.
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)  # Optimizer for the critic networks.  # Optimierer für die Critic-Netzwerke.
        if self.target_entropy is None:  # If target entropy is not provided.  # Falls keine Ziel-Entropie angegeben ist.
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)  # Set default target entropy.  # Setzt die Standard-Ziel-Entropie.
        else:
            self.target_entropy = float(self.target_entropy)  # Converts target entropy to float.  # Konvertiert Ziel-Entropie in einen Float.
        if self.learn_entropy_coef:  # If entropy coefficient should be learned.  # Falls der Entropie-Koeffizient gelernt werden soll.
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)  # Initializes log of alpha with requires_grad.  # Initialisiert den Logarithmus von Alpha mit requires_grad.
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)  # Optimizer for entropy coefficient.  # Optimierer für den Entropie-Koeffizienten.
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)  # Sets alpha as a constant value.  # Setzt Alpha als konstanten Wert.

    def get_actor(self):  # Returns the actor model without gradients.  # Gibt das Actor-Modell ohne Gradienten zurück.
        return self.model_nograd.actor

    def train(self, batch):  # Training step for a batch of data.  # Trainingsschritt für einen Batch von Daten.
        o, a, r, o2, d, _ = batch  # Unpacks the batch into observations, actions, rewards, etc.  # Entpackt den Batch in Beobachtungen, Aktionen, Belohnungen, etc.
        pi, logp_pi = self.model.actor(o)  # Gets the action from the actor network.  # Holt die Aktion vom Actor-Netzwerk.
        loss_alpha = None  # Initializes the loss for the entropy coefficient.  # Initialisiert den Verlust für den Entropie-Koeffizienten.
        if self.learn_entropy_coef:  # If entropy coefficient is learned.  # Falls der Entropie-Koeffizient gelernt wird.
            alpha_t = torch.exp(self.log_alpha.detach())  # Computes the actual alpha value.  # Berechnet den tatsächlichen Alpha-Wert.
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()  # Entropy loss.  # Entropie-Verlust.
        else:
            alpha_t = self.alpha_t  # Use fixed alpha.  # Verwendet ein festes Alpha.
        if loss_alpha is not None:  # If there is an entropy loss.  # Falls ein Entropie-Verlust existiert.
            self.alpha_optimizer.zero_grad()  # Resets the gradients of the entropy optimizer.  # Setzt die Gradienten des Entropie-Optimierers zurück.
            loss_alpha.backward()  # Backpropagates the entropy loss.  # Rückwärtsdurchlauf des Entropie-Verlusts.
            self.alpha_optimizer.step()  # Updates the entropy coefficient.  # Aktualisiert den Entropie-Koeffizienten.
        q1 = self.model.q1(o, a)  # Gets the Q-values from the first critic network.  # Holt die Q-Werte vom ersten Critic-Netzwerk.
        q2 = self.model.q2(o, a)  # Gets the Q-values from the second critic network.  # Holt die Q-Werte vom zweiten Critic-Netzwerk.
        with torch.no_grad():  # Disables gradient computation for the target network.  # Deaktiviert die Gradientenberechnung für das Zielnetzwerk.
            a2, logp_a2 = self.model.actor(o2)  # Gets the action for the next observation.  # Holt die Aktion für die nächste Beobachtung.
            q1_pi_targ = self.model_target.q1(o2, a2)  # Computes Q-values from the target critic.  # Berechnet Q-Werte vom Ziel-Critic.
            q2_pi_targ = self.model_target.q2(o2, a2)  # Computes Q-values from the target critic.  # Berechnet Q-Werte vom Ziel-Critic.
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # Takes the minimum of the Q-values from both critics.  # Nimmt das Minimum der Q-Werte aus beiden Critic-Netzwerken.
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)  # Calculates the backup target for the Q-value update.  # Berechnet das Backup-Ziel für die Q-Wert-Aktualisierung.
        loss_q1 = ((q1 - backup)**2).mean()  # Calculates the Q1 loss.  # Berechnet den Q1-Verlust.
        loss_q2 = ((q2 - backup)**2).mean()  # Calculates the Q2 loss.  # Berechnet den Q2-Verlust.
        loss_q = loss_q1 + loss_q2  # Total Q-loss.  # Gesamt-Q-Verlust.
        self.q_optimizer.zero_grad()  # Resets the gradients of the critic optimizer.  # Setzt die Gradienten des Critic-Optimierers zurück.
        loss_q.backward()  # Backpropagates the Q-loss.  # Rückwärtsdurchlauf des Q-Verlusts.
        self.q_optimizer.step()  # Updates the critic parameters.  # Aktualisiert die Critic-Parameter.
        for p in self.q_params:  # Freezes the Q-network parameters.  # Friert die Q-Netzwerk-Parameter ein.
            p.requires_grad = False
        q1_pi = self.model.q1(o, pi)  # Gets the Q-values for the action chosen by the actor.  # Holt die Q-Werte für die vom Actor gewählte Aktion.
        q2_pi = self.model.q2(o, pi)  # Gets the Q-values for the action chosen by the actor.  # Holt die Q-Werte für die vom Actor gewählte Aktion.
        q_pi = torch.min(q1_pi, q2_pi)  # Takes the minimum Q-value for the actor's action.  # Nimmt den Minimum-Q-Wert für die Aktion des Actors.
        loss_pi = (alpha_t * logp_pi - q_pi).mean()  # Calculates the policy loss.  # Berechnet den Policy-Verlust.
        self.pi_optimizer.zero_grad()  # Resets the gradients of the actor optimizer.  # Setzt die Gradienten des Actor-Optimierers zurück.
        loss_pi.backward()  # Backpropagates the policy loss.  # Rückwärtsdurchlauf des Policy-Verlusts.
        self.pi_optimizer.step()  # Updates the actor parameters.  # Aktualisiert die Actor-Parameter.
        for p in self.q_params:  # Unfreezes the Q-network parameters.  # Hebt das Einfrieren der Q-Netzwerk-Parameter auf.
            p.requires_grad = True
        with torch.no_grad():  # Disables gradient computation for the target model update.  # Deaktiviert die Gradientenberechnung für die Aktualisierung des Zielmodells.
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):  # Loops through the model and target model parameters.  # Schleift durch die Modell- und Zielmodell-Parameter.
                p_targ.data.mul_(self.polyak)  # Performs Polyak averaging on the target model.  # Führt Polyak-Averaging auf dem Zielmodell durch.
                p_targ.data.add_((1 - self.polyak) * p.data)  # Updates the target model parameters.  # Aktualisiert die Zielmodell-Parameter.
        ret_dict = dict(  # Creates a dictionary to store the loss values.  # Erstellt ein Wörterbuch zur Speicherung der Verlustwerte.
            loss_actor=loss_pi.detach().item(),  # Stores the actor loss.  # Speichert den Actor-Verlust.
            loss_critic=loss_q.detach().item(),  # Stores the critic loss.  # Speichert den Critic-Verlust.
        )
        if self.learn_entropy_coef:  # If entropy coefficient is learned.  # Falls der Entropie-Koeffizient gelernt wird.
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()  # Adds the entropy coefficient loss.  # Fügt den Entropie-Koeffizientenverlust hinzu.
            ret_dict["entropy_coef"] = alpha_t.item()  # Adds the entropy coefficient value.  # Fügt den Wert des Entropie-Koeffizienten hinzu.
        return ret_dict  # Returns the dictionary with loss values.  # Gibt das Wörterbuch mit den Verlustwerten zurück.




training_agent_cls = partial(MyTrainingAgent,  # Create a partial function for MyTrainingAgent class initialization.  # Erstelle eine partielle Funktion für die Initialisierung der Klasse MyTrainingAgent.
                             model_cls=MyActorCriticModule,  # Specifies the model class to be used.  # Gibt die zu verwendende Modellklasse an.
                             gamma=0.99,  # Discount factor for future rewards.  # Abzinsungsfaktor für zukünftige Belohnungen.
                             polyak=0.995,  # Polyak averaging factor for target network updates.  # Polyak-Averaging-Faktor für Zielnetzwerk-Aktualisierungen.
                             alpha=0.2,  # Learning rate for entropy regularization.  # Lernrate für die Entropie-Regularisierung.
                             lr_actor=1e-3,  # Learning rate for the actor part of the model.  # Lernrate für den Actor-Teil des Modells.
                             lr_critic=1e-3,  # Learning rate for the critic part of the model.  # Lernrate für den Critic-Teil des Modells.
                             lr_entropy=1e-3,  # Learning rate for entropy loss.  # Lernrate für den Entropieverlust.
                             learn_entropy_coef=True,  # Whether to allow entropy coefficient to be learned.  # Ob der Entropie-Koeffizient gelernt werden soll.
                             target_entropy=None)  # Target entropy value, set to None for automatic computation.  # Ziel-Entropiewert, auf None gesetzt für automatische Berechnung.

# Training parameters:

epochs = 10  # maximum number of epochs, usually set this to np.inf  # Maximale Anzahl von Epochen, normalerweise auf np.inf gesetzt.
rounds = 10  # number of rounds per epoch  # Anzahl der Runden pro Epoche.
steps = 1000  # number of training steps per round  # Anzahl der Trainingsschritte pro Runde.
update_buffer_interval = 100  # Interval at which the buffer is updated.  # Intervall, in dem der Puffer aktualisiert wird.
update_model_interval = 100  # Interval at which the model is updated.  # Intervall, in dem das Modell aktualisiert wird.
max_training_steps_per_env_step = 2.0  # Maximum training steps per environment step.  # Maximale Trainingsschritte pro Umweltschritt.
start_training = 400  # Number of steps before starting training.  # Anzahl der Schritte vor Beginn des Trainings.
device = None  # Device to run the model on, None means default device.  # Gerät, auf dem das Modell ausgeführt wird, None bedeutet Standardgerät.

# Trainer instance:

training_cls = partial(  # Create a partial function for training class initialization.  # Erstelle eine partielle Funktion für die Initialisierung der Trainingsklasse.
    TorchTrainingOffline,  # Class for offline training using Torch.  # Klasse für Offline-Training mit Torch.
    env_cls=env_cls,  # Environment class to be used for training.  # Umweltsklasse, die für das Training verwendet wird.
    memory_cls=memory_cls,  # Memory class for storing experiences.  # Memory-Klasse zum Speichern von Erfahrungen.
    training_agent_cls=training_agent_cls,  # The training agent class defined earlier.  # Die oben definierte Trainings-Agent-Klasse.
    epochs=epochs,  # Number of epochs for training.  # Anzahl der Epochen für das Training.
    rounds=rounds,  # Number of rounds per epoch.  # Anzahl der Runden pro Epoche.
    steps=steps,  # Number of steps per round.  # Anzahl der Schritte pro Runde.
    update_buffer_interval=update_buffer_interval,  # Buffer update interval.  # Intervall zur Pufferaktualisierung.
    update_model_interval=update_model_interval,  # Model update interval.  # Intervall zur Modellaktualisierung.
    max_training_steps_per_env_step=max_training_steps_per_env_step,  # Maximum training steps per environment step.  # Maximale Trainingsschritte pro Umweltschritt.
    start_training=start_training,  # Number of steps before starting training.  # Anzahl der Schritte vor Trainingsbeginn.
    device=device)  # Device to run the training on.  # Gerät, auf dem das Training ausgeführt wird.

if __name__ == "__main__":  # Checks if the script is run directly (not imported).  # Überprüft, ob das Skript direkt ausgeführt wird (nicht importiert).
    my_trainer = Trainer(  # Create an instance of the Trainer class.  # Erstelle eine Instanz der Trainer-Klasse.
        training_cls=training_cls,  # Pass the training class to the Trainer.  # Übergebe die Trainingsklasse an den Trainer.
        server_ip=server_ip,  # Server IP address for communication.  # IP-Adresse des Servers für die Kommunikation.
        server_port=server_port,  # Server port for communication.  # Server-Port für die Kommunikation.
        password=password,  # Password for authentication.  # Passwort für die Authentifizierung.
        model_path=model_path,  # Path to save or load the model.  # Pfad zum Speichern oder Laden des Modells.
        checkpoint_path=checkpoints_path)  # Path to save or load checkpoints, None means no checkpoints.  # Pfad zum Speichern oder Laden von Checkpoints, None bedeutet keine Checkpoints.

# Separate threads for running the RolloutWorker and Trainer:

def run_worker(worker):  # Function to run the worker in a separate thread.  # Funktion zum Ausführen des Arbeiters in einem separaten Thread.
    worker.run(test_episode_interval=10)  # Run the worker with a test episode interval of 10.  # Führe den Arbeiter mit einem Test-Episode-Intervall von 10 aus.

def run_trainer(trainer):  # Function to run the trainer in a separate thread.  # Funktion zum Ausführen des Trainers in einem separaten Thread.
    trainer.run()  # Run the trainer.  # Führe den Trainer aus.

if __name__ == "__main__":  # Checks if the script is run directly (not imported).  # Überprüft, ob das Skript direkt ausgeführt wird (nicht importiert).
    daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)  # Create a daemon thread for the worker.  # Erstelle einen Daemon-Thread für den Arbeiter.
    daemon_thread_worker.start()  # Start the worker daemon thread.  # Starte den Daemon-Thread des Arbeiters.

    run_trainer(my_trainer)  # Run the trainer.  # Führe den Trainer aus.

    # the worker daemon thread will be killed here.  # Der Daemon-Thread des Arbeiters wird hier beendet.
