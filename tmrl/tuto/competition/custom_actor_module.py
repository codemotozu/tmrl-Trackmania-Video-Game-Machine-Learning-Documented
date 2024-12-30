# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg  # Import configuration constants from config.json.  # Konfigurationskonstanten aus config.json importieren.
# Useful classes:
import tmrl.config.config_objects as cfg_obj  # Import configuration objects.  # Konfigurationsobjekte importieren.
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial  # Import a utility for partially instantiating classes.  # Dienstprogramm importieren, um Klassen teilweise zu instanziieren.
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server  # Import Trainer, Worker, and Server classes for networking.  # Trainer-, Worker- und Server-Klassen für Netzwerk importieren.

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline  # Import the offline training class for customization.  # Offline-Trainingsklasse zur Anpassung importieren.

# And a couple external libraries:
import numpy as np  # Import NumPy library for numerical computations.  # NumPy-Bibliothek für numerische Berechnungen importieren.
import os  # Import OS module for interacting with the operating system.  # OS-Modul für Interaktion mit dem Betriebssystem importieren.


# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
# You can change these parameters here directly (not recommended),
# or you can change them in the TMRL config.json file (recommended).

# Maximum number of training 'epochs':
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]  # Maximum number of training epochs.  # Maximale Anzahl an Trainings-Epochen.

# Number of rounds per 'epoch':
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]  # Number of rounds in each epoch.  # Anzahl der Runden pro Epoche.

# Number of training steps per round:
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]  # Number of training steps in each round.  # Anzahl der Trainingsschritte pro Runde.

# Minimum number of environment steps collected before training starts:
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]  # Steps to collect before starting training.  # Schritte vor Trainingsbeginn sammeln.

# Maximum training steps / environment steps ratio:
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]  # Training-to-environment step ratio.  # Verhältnis von Trainings- zu Umgebungsschritten.

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]  # Steps between policy updates.  # Schritte zwischen Policy-Updates.

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]  # Steps between buffer updates.  # Schritte zwischen Buffer-Updates.

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'  # Select device for training: GPU or CPU.  # Gerät für das Training auswählen: GPU oder CPU.

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]  # Maximum size of the replay buffer.  # Maximale Größe des Replay-Buffers.

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]  # Size of each training batch.  # Größe jedes Trainingsbatches.

# Wandb credentials:
wandb_run_id = cfg.WANDB_RUN_ID  # Unique identifier for wandb run.  # Eindeutige Kennung für den Wandb-Lauf.
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # Wandb project name.  # Name des Wandb-Projekts.
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # Wandb account or entity name.  # Name des Wandb-Kontos oder der Entität.
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # Wandb API key.  # API-Schlüssel für Wandb.

os.environ['WANDB_API_KEY'] = wandb_key  # Set wandb API key in the environment.  # API-Schlüssel in der Umgebung setzen.

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]  # Maximum steps per episode.  # Maximale Schritte pro Episode.

# Networking parameters:
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # Server IP from trainer's perspective.  # Server-IP aus Sicht des Trainers.
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # Server IP from worker's perspective.  # Server-IP aus Sicht des Arbeiters.
server_port = cfg.PORT  # Port used for server communication.  # Port für Serverkommunikation.
password = cfg.PASSWORD  # Password for securing communication.  # Passwort zur Sicherung der Kommunikation.
security = cfg.SECURITY  # Security protocol for communication.  # Sicherheitsprotokoll für die Kommunikation.


# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM  # Base class for the replay memory.  # Basisklasse für den Replay-Speicher.

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR  # Compression method used for replay memory.  # Komprimierungsmethode für den Replay-Speicher.

# Sample preprocessor for data augmentation:
sample_preprocessor = None  # Preprocessor for augmenting replay samples (None by default).  # Prozessor für Datenaugmentation (standardmäßig None).

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH  # Path to load offline dataset for replay memory.  # Pfad zum Laden eines Offline-Datensatzes für den Replay-Speicher.

# Preprocessor applied by the worker to the observations it collects:
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR  # Preprocessor applied to observations (customizable).  # Vorverarbeitung der Beobachtungen (anpassbar).

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS  # The environment class used (TrackMania Gym).  # Verwendete Umgebungsklasse (TrackMania Gym).

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'  # Device for inference, default is CPU.  # Gerät für Inferenz, standardmäßig CPU.

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # Width of the TrackMania window.  # Breite des TrackMania-Fensters.
window_height = cfg.WINDOW_HEIGHT  # Height of the TrackMania window.  # Höhe des TrackMania-Fensters.

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH  # Width of observation images.  # Breite der Beobachtungsbilder.
img_height = cfg.IMG_HEIGHT  # Height of observation images.  # Höhe der Beobachtungsbilder.

# Whether you are using grayscale (default) or color images:
img_grayscale = cfg.GRAYSCALE  # Whether observations are grayscale or colored.  # Ob die Beobachtungen in Graustufen oder Farbe sind.

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN  # Number of screenshots in observation history.  # Anzahl der Screenshots in der Beobachtungshistorie.

# Number of actions in the action buffer (this is part of observations):
act_buf_len = cfg.ACT_BUF_LEN  # Number of actions in the action buffer.  # Anzahl der Aktionen im Aktionspuffer.

# This is the memory class passed to the Trainer.
memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,  # Total size of the replay memory.  # Gesamtspeichergröße des Replay-Speichers.
                     batch_size=batch_size,  # Batch size used for training.  # Batch-Größe für das Training.
                     sample_preprocessor=sample_preprocessor,  # Preprocessor for replay samples.  # Vorverarbeitung der Replay-Daten.
                     dataset_path=cfg.DATASET_PATH,  # Path to initialize memory with offline dataset.  # Pfad zur Initialisierung des Speichers mit einem Offline-Datensatz.
                     imgs_obs=imgs_buf_len,  # Number of images in each observation.  # Anzahl der Bilder in jeder Beobachtung.
                     act_buf_len=act_buf_len,  # Number of actions in each observation.  # Anzahl der Aktionen in jeder Beobachtung.
                     crc_debug=False)  # Debugging flag (default: False).  # Debug-Flag (Standard: False).


# =====================================================================
# CUSTOM MODEL
# =====================================================================
# Alright, now for the fun part.  # This is where the fun begins.  # Deutsch: Jetzt geht es an den spaßigen Teil.
# Our goal in this competition is to come up with the best trained  # The goal is to create the best-trained model.  # Deutsch: Ziel ist es, das beste trainierte Modell zu erstellen.
# ActorModule for TrackMania 2020, where an 'ActorModule' is a policy.  # The "ActorModule" defines the policy.  # Deutsch: Das "ActorModule" definiert die Policy.
# In this tutorial, we present a deep RL way of tackling this problem:  # Deep RL will be used to solve the problem.  # Deutsch: Deep RL wird verwendet, um das Problem zu lösen.
# we implement our own deep neural network architecture (ActorModule),  # A custom neural network for ActorModule is implemented.  # Deutsch: Ein eigenes neuronales Netz für ActorModule wird implementiert.
# and then we implement our own RL algorithm to train this module.  # A custom RL algorithm is also implemented.  # Deutsch: Ein eigener RL-Algorithmus wird ebenfalls implementiert.

# We will implement SAC and a hybrid CNN/MLP model.  # SAC and a hybrid CNN/MLP model will be used.  # Deutsch: SAC und ein hybrides CNN/MLP-Modell werden verwendet.

# The following constants are from the Spinup implementation of SAC  # Constants from Spinup's SAC implementation.  # Deutsch: Konstanten aus der SAC-Implementierung von Spinup.
# that we simply copy/paste and adapt in this tutorial.  # These constants are adapted in the tutorial.  # Deutsch: Diese Konstanten werden im Tutorial angepasst.
LOG_STD_MAX = 2  # Maximum log standard deviation.  # Deutsch: Maximale logarithmische Standardabweichung.
LOG_STD_MIN = -20  # Minimum log standard deviation.  # Deutsch: Minimale logarithmische Standardabweichung.

# Let us import the ActorModule that we are supposed to implement.  # Importing the ActorModule to be implemented.  # Deutsch: Import des zu implementierenden ActorModule.
# We will use PyTorch in this tutorial.  # PyTorch is the framework being used.  # Deutsch: PyTorch ist das verwendete Framework.
# TMRL readily provides a PyTorch-specific subclass of ActorModule:  # TMRL includes a PyTorch ActorModule subclass.  # Deutsch: TMRL enthält eine PyTorch-Unterklasse von ActorModule.
from tmrl.actor import TorchActorModule  # Importing TorchActorModule.  # Deutsch: Importieren des TorchActorModule.

# Plus a couple useful imports:  # Additional useful imports.  # Deutsch: Zusätzliche nützliche Importe.
import torch  # Importing PyTorch.  # Deutsch: Import von PyTorch.
import torch.nn as nn  # Importing PyTorch's neural network module.  # Deutsch: Import des neuronalen Netzwerk-Moduls von PyTorch.
import torch.nn.functional as F  # Importing functional tools from PyTorch.  # Deutsch: Import von Funktionstools aus PyTorch.
from torch.distributions.normal import Normal  # Importing Normal distribution class.  # Deutsch: Import der Normalverteilungs-Klasse.
from math import floor  # Importing floor function from math module.  # Deutsch: Import der Bodenfunktion aus dem Mathematikmodul.

# In the full version of the TrackMania 2020 environment, the  # Explaining TrackMania's observation space.  # Deutsch: Erklärung des Beobachtungsraums von TrackMania.
# observation-space comprises a history of screenshots. Thus, we need  # Observation includes screenshots, requiring CNNs.  # Deutsch: Beobachtungen umfassen Screenshots, daher sind CNNs erforderlich.
# Computer Vision layers such as a CNN in our model to process these.  # CNN layers are needed for processing screenshots.  # Deutsch: CNN-Schichten werden zur Verarbeitung der Screenshots benötigt.
# The observation space also comprises single floats representing speed,  # Observations include floats for speed, etc.  # Deutsch: Beobachtungen umfassen Werte für Geschwindigkeit usw.
# rpm and gear. We will merge these with the information contained in  # Merge floats with screenshot info via MLP.  # Deutsch: Werte werden mit Screenshot-Infos über ein MLP kombiniert.
# screenshots thanks to an MLP following our CNN layers.  # MLP follows CNN layers.  # Deutsch: Ein MLP folgt auf die CNN-Schichten.

# Here is the MLP:  # Definition of the MLP function.  # Deutsch: Definition der MLP-Funktion.
def mlp(sizes, activation, output_activation=nn.Identity):  # MLP function creates multi-layer perceptrons.  # Deutsch: Die MLP-Funktion erstellt mehrschichtige Perzeptrons.
    """
    A simple MLP (MultiLayer Perceptron).  # Describes the MLP.  # Deutsch: Beschreibt das MLP.

    Args:
        sizes: list of integers representing the hidden size of each layer  # Layer sizes are specified as a list.  # Deutsch: Schichtgrößen werden als Liste angegeben.
        activation: activation function of hidden layers  # Specifies activation function for hidden layers.  # Deutsch: Aktivierungsfunktion für versteckte Schichten.
        output_activation: activation function of the last layer  # Specifies activation for the output layer.  # Deutsch: Aktivierung für die Ausgabeschicht.

    Returns:
        Our MLP in the form of a Pytorch Sequential module  # Returns MLP as PyTorch Sequential model.  # Deutsch: Gibt das MLP als PyTorch-Sequenzmodell zurück.
    """
    layers = []  # Initialize list of layers.  # Deutsch: Initialisierung der Schichtenliste.
    for j in range(len(sizes) - 1):  # Loop through layer pairs.  # Deutsch: Schleife durch die Schichtenpaare.
        act = activation if j < len(sizes) - 2 else output_activation  # Use activation or output_activation.  # Deutsch: Verwendung von Aktivierung oder Ausgabeaktivierung.
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]  # Add linear layer and activation.  # Deutsch: Füge lineare Schicht und Aktivierung hinzu.
    return nn.Sequential(*layers)  # Return Sequential module.  # Deutsch: Rückgabe des Sequenzmoduls.

# The next utility computes the dimensionality of CNN feature maps when flattened together:  # Utility to compute flattened feature dimensions.  # Deutsch: Werkzeug zur Berechnung der abgeflachten Merkmalabmessungen.
def num_flat_features(x):  # Compute flattened size of feature maps.  # Deutsch: Berechnung der abgeflachten Größe der Merkmalkarten.
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored  # Ignore batch dimension.  # Deutsch: Batch-Dimension wird ignoriert.
    num_features = 1  # Initialize number of features.  # Deutsch: Initialisierung der Merkmalsanzahl.
    for s in size:  # Multiply all dimensions.  # Deutsch: Multiplikation aller Dimensionen.
        num_features *= s  # Update feature count.  # Deutsch: Merkmalsanzahl aktualisieren.
    return num_features  # Return total features.  # Deutsch: Gesamtanzahl der Merkmale zurückgeben.

# The next utility computes the dimensionality of the output in a 2D CNN layer:  # Computes output size of 2D CNN layers.  # Deutsch: Berechnung der Ausgabegröße von 2D-CNN-Schichten.
def conv2d_out_dims(conv_layer, h_in, w_in):  # Compute CNN output dimensions.  # Deutsch: Berechnung der CNN-Ausgabeabmessungen.
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)  # Height calculation.  # Deutsch: Berechnung der Höhe.
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)  # Width calculation.  # Deutsch: Berechnung der Breite.
    return h_out, w_out  # Return height and width.  # Deutsch: Höhe und Breite zurückgeben.



# Let us now define a module that will be the main building block of both our actor and critic:
class VanillaCNN(nn.Module):  # Define a neural network class for SAC (Soft Actor-Critic).  # Definieren Sie eine neuronale Netzklasse für SAC (Soft Actor-Critic).
    def __init__(self, q_net):  # Initialize the VanillaCNN class.  # Initialisieren Sie die Klasse VanillaCNN.
        """
        Simple CNN (Convolutional Neural Network) model for SAC (Soft Actor-Critic).
        Einfache CNN (Convolutional Neural Network)-Modell für SAC (Soft Actor-Critic).
        Args:
            q_net (bool): indicates whether this neural net is a critic network
            q_net (bool): gibt an, ob dieses neuronale Netz ein Kritiker-Netzwerk ist.
        """
        super(VanillaCNN, self).__init__()  # Call the constructor of the parent nn.Module.  # Rufen Sie den Konstruktor der Elternklasse nn.Module auf.

        self.q_net = q_net  # Store whether the model is a critic network.  # Speichern, ob das Modell ein Kritiker-Netzwerk ist.

        # Define the CNN layers to process screenshots.  # Definieren Sie die CNN-Schichten zur Verarbeitung von Screenshots.
        self.h_out, self.w_out = img_height, img_width  # Initialize image dimensions.  # Initialisieren Sie die Bilddimensionen.
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)  # First convolutional layer.  # Erste Convolutional-Schicht.
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)  # Update dimensions after conv1.  # Aktualisieren Sie die Dimensionen nach conv1.
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)  # Second convolutional layer.  # Zweite Convolutional-Schicht.
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)  # Update dimensions after conv2.  # Aktualisieren Sie die Dimensionen nach conv2.
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)  # Third convolutional layer.  # Dritte Convolutional-Schicht.
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)  # Update dimensions after conv3.  # Aktualisieren Sie die Dimensionen nach conv3.
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)  # Fourth convolutional layer.  # Vierte Convolutional-Schicht.
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)  # Update dimensions after conv4.  # Aktualisieren Sie die Dimensionen nach conv4.
        self.out_channels = self.conv4.out_channels  # Get the number of output channels from conv4.  # Holen Sie sich die Anzahl der Ausgabekanäle von conv4.

        self.flat_features = self.out_channels * self.h_out * self.w_out  # Calculate the flattened features.  # Berechnen Sie die abgeflachten Merkmale.

        # Define the input size for the MLP.  # Definieren Sie die Eingabegröße für das MLP.
        float_features = 12 if self.q_net else 9  # Extra features depend on whether it's a critic network.  # Zusätzliche Merkmale hängen davon ab, ob es ein Kritiker-Netzwerk ist.
        self.mlp_input_features = self.flat_features + float_features  # Total input features for the MLP.  # Gesamteingabemerkmale für das MLP.

        # Define the MLP layers.  # Definieren Sie die MLP-Schichten.
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]  # Layer sizes depend on whether it's a critic.  # Schichtgrößen hängen davon ab, ob es ein Kritiker ist.
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)  # Create the MLP with ReLU activations.  # Erstellen Sie das MLP mit ReLU-Aktivierungen.

    def forward(self, x):  # Define the forward pass of the neural network.  # Definieren Sie den Vorwärtsdurchlauf des neuronalen Netzwerks.
        """
        Compute the network's output given its input.  # Berechnen Sie die Ausgabe des Netzwerks basierend auf der Eingabe.
        Args:
            x (torch.Tensor): input tensor.  # Eingabetensor.
        Returns:
            torch.Tensor: output of the network.  # Ausgabe des Netzwerks.
        """
        if self.q_net:  # Check if this is a critic network.  # Überprüfen, ob es sich um ein Kritiker-Netzwerk handelt.
            speed, gear, rpm, images, act1, act2, act = x  # Parse inputs including the next action.  # Analysieren Sie Eingaben, einschließlich der nächsten Aktion.
        else:  # If not a critic network, we compute the next action.  # Wenn kein Kritiker-Netzwerk, berechnen wir die nächste Aktion.
            speed, gear, rpm, images, act1, act2 = x  # Parse inputs without the next action.  # Analysieren Sie Eingaben ohne die nächste Aktion.

        # Pass images through the CNN layers.  # Führen Sie Bilder durch die CNN-Schichten.
        x = F.relu(self.conv1(images))  # Apply first convolution and ReLU activation.  # Wenden Sie die erste Faltung und ReLU-Aktivierung an.
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation.  # Wenden Sie die zweite Faltung und ReLU-Aktivierung an.
        x = F.relu(self.conv3(x))  # Apply third convolution and ReLU activation.  # Wenden Sie die dritte Faltung und ReLU-Aktivierung an.
        x = F.relu(self.conv4(x))  # Apply fourth convolution and ReLU activation.  # Wenden Sie die vierte Faltung und ReLU-Aktivierung an.

        flat_features = num_flat_features(x)  # Calculate flattened features from the CNN output.  # Berechnen Sie abgeflachte Merkmale aus der CNN-Ausgabe.
        assert flat_features == self.flat_features, f"Mismatch in expected dimensions."  # Ensure dimensions match expectations.  # Stellen Sie sicher, dass die Dimensionen den Erwartungen entsprechen.
        x = x.view(-1, flat_features)  # Flatten the feature map.  # Abflachen der Merkmalskarte.

        if self.q_net:  # Combine features for critic network.  # Kombinieren Sie Merkmale für das Kritiker-Netzwerk.
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)  # Concatenate features with actions.  # Verkettung von Merkmalen mit Aktionen.
        else:  # Combine features for policy network.  # Kombinieren Sie Merkmale für das Politiknetzwerk.
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)  # Concatenate features without next action.  # Verkettung von Merkmalen ohne nächste Aktion.

        x = self.mlp(x)  # Pass concatenated features through the MLP.  # Geben Sie verkettete Merkmale durch das MLP.
        return x  # Return the output of the network.  # Geben Sie die Ausgabe des Netzwerks zurück.


# We can now implement the TMRL ActorModule interface that we are supposed to submit for this competition.

# During training, TMRL will regularly save our trained ActorModule in the TmrlData/weights folder.
# By default, this would be done using the torch (i.e., pickle) serializer.
# However, while saving and loading your own pickle files is fine,
# it is highly dangerous to load other people's pickle files.
# Therefore, the competition submission does not accept pickle files.
# Instead, we can submit our trained weights in the form of a human-readable JSON file.
# The ActorModule interface defines save() and load() methods that we will override with our own JSON serializer.

import json  # Import the json module for JSON serialization and deserialization.  # Importiere das json-Modul für JSON-Serialisierung und -Deserialisierung.

class TorchJSONEncoder(json.JSONEncoder):  # Define a custom JSON encoder class for torch tensors.  # Definiere eine benutzerdefinierte JSON-Encoder-Klasse für Torch-Tensoren.
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):  # Override the default method to handle torch.Tensor objects.  # Überschreibe die Standardmethode, um mit torch.Tensor-Objekten umzugehen.
        if isinstance(obj, torch.Tensor):  # Check if the object is a torch tensor.  # Überprüfe, ob das Objekt ein Torch-Tensor ist.
            return obj.cpu().detach().numpy().tolist()  # Convert tensor to a list of numbers after detaching from GPU and moving to CPU.  # Konvertiere den Tensor in eine Liste von Zahlen, nachdem er vom GPU getrennt und auf die CPU verschoben wurde.
        return json.JSONEncoder.default(self, obj)  # For other objects, use the default encoding behavior.  # Verwende für andere Objekte das Standard-Encodierungsverhalten.

class TorchJSONDecoder(json.JSONDecoder):  # Define a custom JSON decoder class for torch tensors.  # Definiere eine benutzerdefinierte JSON-Decoder-Klasse für Torch-Tensoren.
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):  # Initialize the decoder, passing arguments to the base class.  # Initialisiere den Decoder und übergebe Argumente an die Basisklasse.
        super().__init__(object_hook=self.object_hook, *args, **kwargs)  # Set the object_hook method to decode tensors.  # Setze die object_hook-Methode, um Tensoren zu decodieren.

    def object_hook(self, dct):  # Define how to convert lists back into torch tensors.  # Definiere, wie Listen wieder in Torch-Tensoren umgewandelt werden.
        for key in dct.keys():  # Loop through each key in the dictionary.  # Schleife durch jeden Schlüssel im Wörterbuch.
            if isinstance(dct[key], list):  # If the value is a list, convert it to a tensor.  # Wenn der Wert eine Liste ist, konvertiere ihn in einen Tensor.
                dct[key] = torch.Tensor(dct[key])  # Convert the list to a torch tensor.  # Konvertiere die Liste in einen Torch-Tensor.
        return dct  # Return the updated dictionary.  # Gib das aktualisierte Wörterbuch zurück.

class MyActorModule(TorchActorModule):  # Define a custom ActorModule for the policy.  # Definiere ein benutzerdefiniertes ActorModule für die Politik.
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """
    def __init__(self, observation_space, action_space):  # Initialize the module with observation and action spaces.  # Initialisiere das Modul mit Beobachtungs- und Aktionsräumen.
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.
        
        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        # We must call the superclass __init__:  # Wir müssen den Konstruktor der Basisklasse aufrufen:
        super().__init__(observation_space, action_space)  # Call the superclass constructor.  # Rufe den Konstruktor der Basisklasse auf.

        # And initialize our attributes:  # Und initialisiere unsere Attribute:
        dim_act = action_space.shape[0]  # dimensionality of actions  # Dimensionalität der Aktionen
        act_limit = action_space.high[0]  # maximum amplitude of actions  # Maximale Amplitude der Aktionen
        # Our hybrid CNN+MLP policy:  # Unsere hybride CNN+MLP-Politik:
        self.net = VanillaCNN(q_net=False)  # Initialize a CNN network for the policy.  # Initialisiere ein CNN-Netzwerk für die Politik.
        # The policy output layer, which samples actions stochastically in a gaussian, with means...:  # Die Ausgabeschicht der Politik, die Aktionen stochastisch in einer Gaußschen Verteilung mit Mittelwerten sticht.
        self.mu_layer = nn.Linear(256, dim_act)  # Layer for the means of the action distribution.  # Schicht für die Mittelwerte der Aktionsverteilung.
        # ... and log standard deviations:  # ... und logarhythmische Standardabweichungen:
        self.log_std_layer = nn.Linear(256, dim_act)  # Layer for the log standard deviations.  # Schicht für die logarhythmischen Standardabweichungen.
        # We will squash this within the action space thanks to a tanh final activation:  # Wir werden dies mit einer tanh-Endaktivierung im Aktionsraum quetschen:
        self.act_limit = act_limit  # Store the action limit.  # Speichere die Aktionsgrenze.

    def save(self, path):  # Method to save the ActorModule state to a file.  # Methode zum Speichern des ActorModule-Zustands in einer Datei.
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:  # Open the specified path in write mode.  # Öffne den angegebenen Pfad im Schreibmodus.
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)  # Serialize the state dict to the file using custom encoder.  # Serialisiere das state_dict in die Datei mit dem benutzerdefinierten Encoder.
        # torch.save(self.state_dict(), path)  # Alternative saving using torch (commented out).  # Alternative Speicherung mit torch (auskommentiert).

    def load(self, path, device):  # Method to load the ActorModule state from a file.  # Methode zum Laden des ActorModule-Zustands aus einer Datei.
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device  # Store the device where the ActorModule will be loaded.  # Speichere das Gerät, auf dem das ActorModule geladen wird.
        with open(path, 'r') as json_file:  # Open the JSON file in read mode.  # Öffne die JSON-Datei im Lesemodus.
            state_dict = json.load(json_file, cls=TorchJSONDecoder)  # Load the state dict from the file using custom decoder.  # Lade das state_dict aus der Datei mit dem benutzerdefinierten Decoder.
        self.load_state_dict(state_dict)  # Load the state dict into the ActorModule.  # Lade das state_dict in das ActorModule.
        self.to_device(device)  # Move the ActorModule to the specified device.  # Verschiebe das ActorModule auf das angegebene Gerät.
        # self.load_state_dict(torch.load(path, map_location=self.device))  # Alternative loading using torch (commented out).  # Alternative Laden mit torch (auskommentiert).
        return self  # Return the loaded ActorModule instance.  # Gib die geladene ActorModule-Instanz zurück.


def forward(self, obs, test=False, compute_logprob=True):  # Defines the forward pass of the actor network.  # Definiert den Vorwärtsdurchgang des Actor-Netzwerks.
    """
    Computes the output action of our policy from the input observation.  # Berechnet die Ausgabemaßnahme unserer Politik aus der Eingabe-Beobachtung.
    
    The whole point of deep RL is to train our policy network (actor) such that it outputs relevant actions.  # Das gesamte Ziel von Deep RL ist es, unser Politik-Netzwerk (Actor) so zu trainieren, dass es relevante Maßnahmen ausgibt.
    Training per-se will also rely on a critic network, but this is not part of the trained policy.  # Das Training selbst wird auch auf ein Kritiker-Netzwerk angewiesen sein, aber dies ist nicht Teil der trainierten Politik.
    Thus, our ActorModule will only implement the actor.  # Daher wird unser ActorModule nur den Actor implementieren.

    Args:  # Parameter des Moduls
        obs: the observation from the Gymnasium environment (when using TorchActorModule this is a torch.Tensor)  # Beobachtung aus der Gymnasium-Umgebung (bei Verwendung von TorchActorModule handelt es sich um ein torch.Tensor).
        test (bool): this is True for test episodes (deployment) and False for training episodes;  # test (bool): Ist dies True für Test-Episoden (Einsatz) und False für Trainings-Episoden;
            in SAC, this enables us to sample randomly during training and deterministically at test-time.  # im SAC ermöglicht es uns, während des Trainings zufällig und während des Tests deterministisch zu sampeln.
        compute_logprob (bool): SAC will set this to True to retrieve log probabilities.  # compute_logprob (bool): SAC setzt dies auf True, um Log-Wahrscheinlichkeiten abzurufen.

    Returns:  # Gibt zurück
        the action sampled from our policy from observation obs  # Die Aktion, die aus unserer Politik anhand der Beobachtung obs abgetastet wird.
        the log probability of this action (this will be used for SAC)  # Die Log-Wahrscheinlichkeit dieser Aktion (dies wird für SAC verwendet).
    """
    net_out = self.net(obs)  # We pass the observation through the network (MLP) to get the output.  # Wir leiten die Beobachtung durch das Netzwerk (MLP), um die Ausgabe zu erhalten.
    
    mu = self.mu_layer(net_out)  # Output the means of the multivariate Gaussian.  # Gibt die Mittelwerte der multivariaten Normalverteilung aus.
    log_std = self.log_std_layer(net_out)  # Output the log of the standard deviations.  # Gibt das Logarithmus der Standardabweichungen aus.
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clamps log_std to be within a minimum and maximum value for stability.  # Begrenzt log_std auf einen Minimal- und Maximalwert für die Stabilität.
    std = torch.exp(log_std)  # Compute the standard deviation from the log of the standard deviation.  # Berechnet die Standardabweichung aus dem Logarithmus der Standardabweichung.
    
    pi_distribution = Normal(mu, std)  # Create the normal distribution with the computed means and standard deviations.  # Erzeugt die Normalverteilung mit den berechneten Mittelwerten und Standardabweichungen.
    
    if test:  # During testing, the action is deterministic (use the mean).  # Im Testmodus ist die Aktion deterministisch (verwende den Mittelwert).
        pi_action = mu  # At test time, the action is just the mean of the distribution.  # Im Testzeitpunkt ist die Aktion nur der Mittelwert der Verteilung.
    else:  # During training, sample an action from the distribution.  # Im Training wird eine Aktion aus der Verteilung abgetastet.
        pi_action = pi_distribution.rsample()  # Sample the action from the distribution during training.  # Stichproben der Aktion aus der Verteilung während des Trainings.

    if compute_logprob:  # If log probabilities are needed for SAC, compute them.  # Wenn Log-Wahrscheinlichkeiten für SAC benötigt werden, berechne sie.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Get the log probability of the action.  # Berechne die Log-Wahrscheinlichkeit der Aktion.
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # Correction term for actions squashed by tanh.  # Korrekturterm für durch tanh komprimierte Aktionen.
    else:
        logp_pi = None  # If not computing log probabilities, set it to None.  # Wenn keine Log-Wahrscheinlichkeiten berechnet werden, setze sie auf None.
    
    pi_action = torch.tanh(pi_action)  # Apply tanh squashing to the action to limit it within the action space.  # Wende tanh an, um die Aktion innerhalb des Aktionsraums zu begrenzen.
    pi_action = self.act_limit * pi_action  # Scale the action according to the action limit.  # Skaliere die Aktion gemäß der Aktionsbegrenzung.
    pi_action = pi_action.squeeze()  # Remove the batch dimension from the action.  # Entferne die Batch-Dimension aus der Aktion.
    
    return pi_action, logp_pi  # Return the action and log probability.  # Gibt die Aktion und Log-Wahrscheinlichkeit zurück.


def act(self, obs, test=False):  # Method to compute an action based on the observation.  # Methode zur Berechnung einer Aktion basierend auf der Beobachtung.
    """
    Computes an action from an observation.  # Berechnet eine Aktion aus einer Beobachtung.
    
    This method is the one all participants must implement.  # Diese Methode muss von allen Teilnehmern implementiert werden.
    It is the policy that TMRL will use in TrackMania to evaluate your submission.  # Es ist die Politik, die TMRL in TrackMania verwenden wird, um deine Einreichung zu bewerten.

    Args:  # Parameter des Moduls
        obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)  # Eingabe-Beobachtung (bei Verwendung von TorchActorModule handelt es sich um ein torch.Tensor).
        test (bool): True at test-time (e.g., during evaluation...), False otherwise  # test (bool): True im Testzeitpunkt (z.B. während der Auswertung...), False sonst.

    Returns:  # Gibt zurück
        act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0  # Die berechnete Aktion, in Form eines numpy-Arrays mit 3 Werten zwischen -1.0 und 1.0.
    """
    with torch.no_grad():  # Ensures no gradients are computed during the action calculation.  # Stellt sicher, dass während der Aktionsberechnung keine Gradienten berechnet werden.
        a, _ = self.forward(obs=obs, test=test, compute_logprob=False)  # Get the action from the forward pass without computing log probabilities.  # Hole die Aktion aus dem Vorwärtsdurchgang ohne Berechnung der Log-Wahrscheinlichkeiten.
        return a.cpu().numpy()  # Convert the action to a numpy array and return it.  # Konvertiere die Aktion in ein numpy-Array und gib sie zurück.


class VanillaCNNQFunction(nn.Module):  # Defining the critic module class for SAC.  # Definition der Kritiker-Klasse für SAC.
    """
    Critic module for SAC.  # Explanation of the critic module for SAC.  # Erklärung des Kritiker-Moduls für SAC.
    """
    def __init__(self, observation_space, action_space):  # Constructor method to initialize the class with observation and action space.  # Konstruktor zur Initialisierung der Klasse mit Beobachtungs- und Aktionsraum.
        super().__init__()  # Calling the parent class's initializer.  # Aufruf des Initialisierers der Elternklasse.
        self.net = VanillaCNN(q_net=True)  # q_net is True to specify it's a critic network.  # q_net ist True, um anzugeben, dass es sich um ein Kritiker-Netzwerk handelt.

    def forward(self, obs, act):  # Forward pass method to estimate the action-value for a given state-action pair.  # Vorwärtspass-Methode zur Schätzung des Aktionswerts für ein gegebenes Zustand-Aktions-Paar.
        """
        Estimates the action-value of the (obs, act) state-action pair.  # Schätzt den Aktionswert des Zustand-Aktions-Paars (obs, act).  
        In RL theory, the action-value is the expected sum of (gamma-discounted) future rewards  # In der RL-Theorie ist der Aktionswert die erwartete Summe der (mit Gamma diskontierten) zukünftigen Belohnungen,
        when observing obs, taking action act, and following the current policy ever after.  # wenn wir obs beobachten, die Aktion act ausführen und der aktuellen Politik folgen.
        
        Args:  # Argumente:
            obs: current observation  # Beobachtung der aktuellen Situation  # aktuelle Beobachtung
            act: tried next action  # Die versuchte nächste Aktion  # versuchte nächste Aktion

        Returns:  # Rückgabewert:
            The action-value of act in situation obs, as estimated by our critic network  # Der Aktionswert der Aktion act in der Situation obs, wie vom Kritiker-Netzwerk geschätzt.  # Der Aktionswert von act in der Situation obs, geschätzt von unserem Kritiker-Netzwerk
        """
        # Since q_net is True, we append our action act to our observation obs.  # Da q_net True ist, fügen wir die Aktion act an die Beobachtung obs an.  
        # Note that obs is a tuple of batched tensors: respectively the history of 4 images, speed, etc.  # Beachten Sie, dass obs ein Tupel von Stapel-Tensoren ist: jeweils die Historie von 4 Bildern, Geschwindigkeit usw. 
        x = (*obs, act)  # Combine observation and action into a single input.  # Kombiniere Beobachtung und Aktion zu einem einzigen Input. 
        q = self.net(x)  # Pass the combined input through the neural network to get the action-value.  # Gib den kombinierten Input durch das neuronale Netzwerk, um den Aktionswert zu erhalten.
        return torch.squeeze(q, -1)  # Remove any extra dimensions from the output to return the action-value.  # Entferne alle zusätzlichen Dimensionen aus der Ausgabe, um den Aktionswert zurückzugeben.


class VanillaCNNActorCritic(nn.Module):  # Defining the actor-critic module for SAC algorithm.  # Definition des Actor-Critic-Moduls für den SAC-Algorithmus.
    """
    Actor-critic module for the SAC algorithm.  # Erklärung des Actor-Critic-Moduls für den SAC-Algorithmus.  
    """
    def __init__(self, observation_space, action_space):  # Constructor method for initializing the actor-critic module.  # Konstruktor zur Initialisierung des Actor-Critic-Moduls.
        super().__init__()  # Calling the parent class initializer.  # Aufruf des Initialisierers der Elternklasse.

        # Policy network (actor):  # Policy-Netzwerk (Actor): 
        self.actor = MyActorModule(observation_space, action_space)  # Creating the actor module for selecting actions based on observations.  # Erstelle das Actor-Modul zum Auswählen von Aktionen basierend auf Beobachtungen.
        # Value networks (critics):  # Wertnetzwerke (Critics): 
        self.q1 = VanillaCNNQFunction(observation_space, action_space)  # First critic network to estimate action-values.  # Erstes Kritiker-Netzwerk zur Schätzung der Aktionswerte.
        self.q2 = VanillaCNNQFunction(observation_space, action_space)  # Second critic network to estimate action-values.  # Zweites Kritiker-Netzwerk zur Schätzung der Aktionswerte.



# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================
# So far, we have implemented our custom model.  # Wir haben bisher unser benutzerdefiniertes Modell implementiert.
# We have also wrapped it in an ActorModule, which we will train and  # Wir haben es auch in ein ActorModule verpackt, das wir trainieren werden
# submit as an entry to the TMRL competition.  # und als Beitrag zum TMRL-Wettbewerb einreichen.
# Our ActorModule will be used in Workers to collect training data.  # Unser ActorModule wird in Workern verwendet, um Trainingsdaten zu sammeln.
# Our VanillaCNNActorCritic will be used in the Trainer for training  # Unser VanillaCNNActorCritic wird im Trainer zum Training verwendet
# this ActorModule. Let us now tackle the training algorithm per-se.  # Dieses ActorModule. Lassen Sie uns nun den Trainingsalgorithmus selbst angehen.
# In TMRL, this is done by implementing a custom TrainingAgent.  # In TMRL wird dies durch Implementierung eines benutzerdefinierten TrainingAgent durchgeführt.

from tmrl.training import TrainingAgent  # Importiert den TrainingAgent aus dem tmrl.training Modul.  # Importiert den TrainingAgent aus dem tmrl.training Modul.

# We will also use a couple utilities, and the Adam optimizer:  # Wir werden auch einige Hilfsfunktionen und den Adam-Optimierer verwenden:
from tmrl.custom.utils.nn import copy_shared, no_grad  # Importiert die Hilfsfunktionen copy_shared und no_grad.  # Importiert die Hilfsfunktionen copy_shared und no_grad.
from tmrl.util import cached_property  # Importiert cached_property für das Caching von Eigenschaften.  # Importiert cached_property für das Caching von Eigenschaften.
from copy import deepcopy  # Importiert deepcopy, um tiefe Kopien von Objekten zu erstellen.  # Importiert deepcopy, um tiefe Kopien von Objekten zu erstellen.
import itertools  # Importiert itertools, um Iterationen zu erleichtern.  # Importiert itertools, um Iterationen zu erleichtern.
from torch.optim import Adam  # Importiert den Adam-Optimierer aus PyTorch.  # Importiert den Adam-Optimierer aus PyTorch.

# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinup implementation.

class SACTrainingAgent(TrainingAgent):  # Defining a new class SACTrainingAgent that inherits from the TrainingAgent class.  # Definiert eine neue Klasse SACTrainingAgent, die von der Klasse TrainingAgent erbt.
    """
    Our custom training algorithm (SAC in this tutorial).  # Our custom training algorithm: Soft Actor-Critic (SAC).  # Unser eigener Trainingsalgorithmus: Soft Actor-Critic (SAC).
    Custom TrainingAgents implement two methods: train(batch) and get_actor().  # Custom TrainingAgents müssen zwei Methoden implementieren: train(batch) und get_actor(). 
    The train method performs a training step.  # Die train-Methode führt einen Trainingsschritt aus. 
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.  # Die get_actor-Methode ruft das ActorModule ab, um es zu speichern und an die RolloutWorkers zu senden. 
    Your implementation must also pass three required arguments to the superclass:  # Ihre Implementierung muss auch drei erforderliche Argumente an die Superklasse übergeben:
    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)  # Beobachtungsraum (hier zu Ihrer Bequemlichkeit). 
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)  # Aktionsraum (hier zu Ihrer Bequemlichkeit). 
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)  # Gerät, das für das Training verwendet werden soll (z.B. `"cpu"` oder `"cuda:0"`).
    """
    
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))  # A no-grad copy of the model used to send Actor weights in get_actor() without affecting gradients.  # Eine "no-grad"-Kopie des Modells, die verwendet wird, um die Actor-Gewichte in get_actor() zu senden, ohne die Gradienten zu beeinflussen.

    def __init__(self,  # Constructor method to initialize the agent with various hyperparameters.  # Konstruktormethode, um den Agenten mit verschiedenen Hyperparametern zu initialisieren.
                 observation_space=None,  # Gymnasium observation space (required argument here for your convenience)  # Gymnasium-Beobachtungsraum (erforderliches Argument für Ihre Bequemlichkeit).
                 action_space=None,  # Gymnasium action space (required argument here for your convenience)  # Gymnasium-Aktionsraum (erforderliches Argument für Ihre Bequemlichkeit).
                 device=None,  # Device for training (e.g., "cpu" or "cuda:0").  # Gerät für das Training (z.B. "cpu" oder "cuda:0").
                 model_cls=VanillaCNNActorCritic,  # Actor-critic module class used for the agent's neural network model.  # Actor-Critic-Modulklasse, die für das neuronale Netzwerkmodell des Agenten verwendet wird.
                 gamma=0.99,  # Discount factor, determining the importance of future rewards.  # Abzinsungsfaktor, der die Bedeutung zukünftiger Belohnungen bestimmt.
                 polyak=0.995,  # Polyak averaging factor for updating the target critic.  # Polyak-Averaging-Faktor zur Aktualisierung des Zielkritikers.
                 alpha=0.2,  # Entropy coefficient controlling the exploration-exploitation trade-off.  # Entropie-Koeffizient, der den Handel zwischen Exploration und Exploitation steuert.
                 lr_actor=1e-3,  # Learning rate for the actor part of the model.  # Lernrate für den Actor-Teil des Modells.
                 lr_critic=1e-3):  # Learning rate for the critic part of the model.  # Lernrate für den Critic-Teil des Modells.

        super().__init__(observation_space=observation_space,  # Call the superclass constructor with observation_space, action_space, and device.  # Aufruf des Konstruktors der Superklasse mit observation_space, action_space und device.
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space, action_space)  # Initialize the model using the given class.  # Initialisiere das Modell unter Verwendung der gegebenen Klasse.
        self.model = model.to(self.device)  # Move the model to the appropriate device (e.g., CPU or GPU).  # Verschiebe das Modell auf das entsprechende Gerät (z.B. CPU oder GPU).
        self.model_target = no_grad(deepcopy(self.model))  # Create a copy of the model without gradients for target updates.  # Erstelle eine Kopie des Modells ohne Gradienten für Zielaktualisierungen.
        self.gamma = gamma  # Set the gamma parameter for discounting future rewards.  # Setze den Gamma-Parameter für die Abzinsung zukünftiger Belohnungen.
        self.polyak = polyak  # Set the Polyak averaging factor.  # Setze den Polyak-Averaging-Faktor.
        self.alpha = alpha  # Set the entropy coefficient.  # Setze den Entropie-Koeffizienten.
        self.lr_actor = lr_actor  # Set the learning rate for the actor.  # Setze die Lernrate für den Actor.
        self.lr_critic = lr_critic  # Set the learning rate for the critic.  # Setze die Lernrate für den Critic.
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())  # Combine the parameters of both Q-networks (for the critic).  # Kombiniere die Parameter beider Q-Netzwerke (für den Critic).
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)  # Initialize the optimizer for the actor.  # Initialisiere den Optimierer für den Actor.
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)  # Initialize the optimizer for the critic.  # Initialisiere den Optimierer für den Critic.
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)  # Convert alpha to a tensor and move it to the correct device.  # Wandle Alpha in ein Tensor um und verschiebe es auf das richtige Gerät.

    def get_actor(self):  # Method to get a copy of the current actor module without gradients.  # Methode, um eine Kopie des aktuellen Actor-Moduls ohne Gradienten zu erhalten.
        """
        Returns a copy of the current ActorModule.  # Gibt eine Kopie des aktuellen ActorModules zurück.
        We return a copy without gradients, as this is for sending to the RolloutWorkers.  # Wir geben eine Kopie ohne Gradienten zurück, da diese an die RolloutWorkers gesendet wird.
        Returns:  # Rückgabewerte:
            actor: ActorModule: updated actor module to forward to the worker(s)  # actor: ActorModule: aktualisiertes Actor-Modul, das an den Worker weitergegeben wird.
        """
        return self.model_nograd.actor  # Return the actor part of the model without gradients.  # Gibt den Actor-Teil des Modells ohne Gradienten zurück.


  def train(self, batch):  # Defines the train function that takes a batch of training samples as input.  # Definiert die Trainingsfunktion, die eine Batch von Trainingsbeispielen als Eingabe nimmt.
    """
    Executes a training iteration from batched training samples (batches of RL transitions).  # Führt eine Trainingsiteration aus, basierend auf Stapeln von Trainingsproben (RL-Übergänge).
    A training sample is of the form (o, a, r, o2, d, t) where:  # Ein Trainingsbeispiel hat die Form (o, a, r, o2, d, t), wobei:
    -> o is the initial observation of the transition  # o ist die Anfangsbeobachtung des Übergangs
    -> a is the selected action during the transition  # a ist die ausgewählte Aktion während des Übergangs
    -> r is the reward of the transition  # r ist die Belohnung des Übergangs
    -> o2 is the final observation of the transition  # o2 ist die Endbeobachtung des Übergangs
    -> d is the "terminated" signal indicating whether o2 is a terminal state  # d ist das "terminated"-Signal, das angibt, ob o2 ein Endzustand ist
    -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit  # t ist das "truncated"-Signal, das angibt, ob die Episode durch eine Zeitbegrenzung gekürzt wurde
    """
    batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)  # Beschreibung der Form der Eingabedaten (Batch).  # Beschreibung der Form der Eingabedaten (Batch).

    o, a, r, o2, d, _ = batch  # Decomposes the batch into its components, ignoring the truncated signal.  # Zerlegt das Batch in seine Komponenten und ignoriert das "truncated"-Signal.
    
    pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)  # Samples an action from the actor and computes its log probability.  # Wählt eine Aktion vom Actor und berechnet ihre Log-Wahrscheinlichkeit.
    
    q1 = self.model.q1(o, a)  # Computes the Q-value for the first critic.  # Berechnet den Q-Wert für den ersten Kritiker.
    q2 = self.model.q2(o, a)  # Computes the Q-value for the second critic.  # Berechnet den Q-Wert für den zweiten Kritiker.
    
    with torch.no_grad():  # Disables gradient computation for the following code block.  # Deaktiviert die Gradientenberechnung für den folgenden Codeblock.
        a2, logp_a2 = self.model.actor(o2)  # Samples the action for the next observation.  # Wählt die Aktion für die nächste Beobachtung.
        q1_pi_targ = self.model_target.q1(o2, a2)  # Computes the Q-value for the target model's first critic.  # Berechnet den Q-Wert des ersten Kritikers des Zielmodells.
        q2_pi_targ = self.model_target.q2(o2, a2)  # Computes the Q-value for the target model's second critic.  # Berechnet den Q-Wert des zweiten Kritikers des Zielmodells.
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # Takes the minimum Q-value from the target critics.  # Nimmt den minimalen Q-Wert der Zielkritiker.
        backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)  # Calculates the backup value for the target.  # Berechnet den Backup-Wert für das Ziel.

    loss_q1 = ((q1 - backup)**2).mean()  # Computes the loss for the first critic by comparing with the backup.  # Berechnet den Verlust für den ersten Kritiker im Vergleich zum Backup.
    loss_q2 = ((q2 - backup)**2).mean()  # Computes the loss for the second critic by comparing with the backup.  # Berechnet den Verlust für den zweiten Kritiker im Vergleich zum Backup.
    loss_q = loss_q1 + loss_q2  # Total critic loss is the sum of the individual critic losses.  # Der gesamte Kritikerverlust ist die Summe der einzelnen Kritikerverluste.

    self.q_optimizer.zero_grad()  # Resets the gradients for the critics' optimizer.  # Setzt die Gradienten des Optimierers der Kritiker zurück.
    loss_q.backward()  # Backpropagates the loss through the critic network.  # Führt das Backpropagation-Verfahren durch den Kritiker-Netzwerk.
    self.q_optimizer.step()  # Performs a step of optimization for the critics.  # Führt einen Optimierungsschritt für die Kritiker aus.

    for p in self.q_params:  # Iterates through the parameters of the critics.  # Iteriert durch die Parameter der Kritiker.
        p.requires_grad = False  # Freezes the critics' parameters (no gradients will be calculated).  # Friert die Parameter der Kritiker ein (es werden keine Gradienten berechnet).

    q1_pi = self.model.q1(o, pi)  # Computes the Q-value for the policy action from the actor.  # Berechnet den Q-Wert für die Politikaktion des Actors.
    q2_pi = self.model.q2(o, pi)  # Computes the Q-value for the policy action from the second critic.  # Berechnet den Q-Wert für die Politikaktion des zweiten Kritikers.
    q_pi = torch.min(q1_pi, q2_pi)  # Takes the minimum Q-value between the two critics for the policy action.  # Nimmt den minimalen Q-Wert zwischen den beiden Kritikern für die Politikaktion.

    loss_pi = (self.alpha_t * logp_pi - q_pi).mean()  # Computes the policy loss by comparing the action-value and the log probability.  # Berechnet den Politikverlust, indem der Aktionswert mit der Log-Wahrscheinlichkeit verglichen wird.

    self.pi_optimizer.zero_grad()  # Resets the gradients for the policy optimizer.  # Setzt die Gradienten des Optimierers für die Politik zurück.
    loss_pi.backward()  # Backpropagates the loss through the policy network.  # Führt das Backpropagation-Verfahren durch das Politik-Netzwerk.
    self.pi_optimizer.step()  # Performs a step of optimization for the policy.  # Führt einen Optimierungsschritt für die Politik aus.

    for p in self.q_params:  # Iterates through the parameters of the critics again.  # Iteriert erneut durch die Parameter der Kritiker.
        p.requires_grad = True  # Re-enables gradient calculation for the critics' parameters.  # Aktiviert die Gradientenberechnung für die Parameter der Kritiker wieder.

    with torch.no_grad():  # Disables gradient computation for the following code block.  # Deaktiviert die Gradientenberechnung für den folgenden Codeblock.
        for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):  # Iterates through the parameters of the model and target model.  # Iteriert durch die Parameter des Modells und des Zielmodells.
            p_targ.data.mul_(self.polyak)  # Applies the Polyak averaging to the target model's parameters.  # Wendet das Polyak-Averaging auf die Parameter des Zielmodells an.
            p_targ.data.add_((1 - self.polyak) * p.data)  # Updates the target model's parameters with a weighted average.  # Aktualisiert die Parameter des Zielmodells mit einem gewichteten Durchschnitt.

    ret_dict = dict(  # Creates a dictionary to log the training metrics.  # Erstellt ein Wörterbuch, um die Trainingsmetriken zu protokollieren.
        loss_actor=loss_pi.detach().item(),  # Adds the actor's loss to the dictionary.  # Fügt den Verlust des Actors zum Wörterbuch hinzu.
        loss_critic=loss_q.detach().item(),  # Adds the critic's loss to the dictionary.  # Fügt den Verlust des Kritikers zum Wörterbuch hinzu.
    )
    return ret_dict  # Returns the dictionary with the training metrics.  # Gibt das Wörterbuch mit den Trainingsmetriken zurück.

# Great! We are almost done.
# Now that our TrainingAgent class is defined, let us partially instantiate it.
# SAC has a few hyperparameters that we will need to tune if we want it to work as expected.
# The following have shown reasonable results in the past, using the full TrackMania environment.
# Note however that training a policy with SAC in this environment is a matter of several days!

training_agent_cls = partial(SACTrainingAgent,  # Creating a partially instantiated SACTrainingAgent class with fixed parameters.  # Erstellen einer teilweise instanziierten SACTrainingAgent-Klasse mit festgelegten Parametern.
                             model_cls=VanillaCNNActorCritic,  # Specifies the model class used by the agent.  # Gibt die Modellklasse an, die vom Agenten verwendet wird.
                             gamma=0.995,  # The discount factor for future rewards.  # Der Diskontierungsfaktor für zukünftige Belohnungen.
                             polyak=0.995,  # Polyak averaging factor for stabilizing training.  # Polyak-Glättungsfaktor zur Stabilisierung des Trainings.
                             alpha=0.01,  # Entropy regularization coefficient for SAC.  # Entropie-Regularisierungskoeffizient für SAC.
                             lr_actor=0.00001,  # Learning rate for the actor (policy network).  # Lernrate für den Actor (Policy-Netzwerk).
                             lr_critic=0.00005)  # Learning rate for the critic (value network).  # Lernrate für den Kritiker (Wertnetzwerk).

# =====================================================================
# TMRL TRAINER
# =====================================================================

training_cls = partial(  # Creating a partially instantiated TrainingOffline class with fixed parameters.  # Erstellen einer teilweise instanziierten TrainingOffline-Klasse mit festgelegten Parametern.
    TrainingOffline,  # Specifies the class of the training process.  # Gibt die Klasse des Trainingsprozesses an.
    env_cls=env_cls,  # Environment class used for training.  # Umweltklasse, die für das Training verwendet wird.
    memory_cls=memory_cls,  # Class for the memory buffer.  # Klasse für den Speicherpuffer.
    training_agent_cls=training_agent_cls,  # The training agent class to be used.  # Die zu verwendende Trainingsagenten-Klasse.
    epochs=epochs,  # Number of epochs for training.  # Anzahl der Epochen für das Training.
    rounds=rounds,  # Number of rounds per epoch.  # Anzahl der Runden pro Epoche.
    steps=steps,  # Number of steps per round.  # Anzahl der Schritte pro Runde.
    update_buffer_interval=update_buffer_interval,  # Interval at which the buffer is updated.  # Intervall, in dem der Puffer aktualisiert wird.
    update_model_interval=update_model_interval,  # Interval at which the model is updated.  # Intervall, in dem das Modell aktualisiert wird.
    max_training_steps_per_env_step=max_training_steps_per_env_step,  # Max training steps per environment step.  # Maximale Trainingsschritte pro Umweltschritt.
    start_training=start_training,  # Boolean to start training process.  # Boolean-Wert zum Starten des Trainingsprozesses.
    device=device_trainer)  # The device used for training (e.g., CPU or GPU).  # Das Gerät, das für das Training verwendet wird (z. B. CPU oder GPU).

# =====================================================================
# RUN YOUR TRAINING PIPELINE
# =====================================================================
# The training pipeline configured in this tutorial runs with the "TM20FULL" environment.  # Die im Tutorial konfigurierte Trainingspipeline läuft mit der "TM20FULL"-Umgebung.
# You can configure the "TM20FULL" environment by following the instruction on GitHub:  # Sie können die "TM20FULL"-Umgebung gemäß den Anweisungen auf GitHub konfigurieren:
# https://github.com/trackmania-rl/tmrl#full-environment  # https://github.com/trackmania-rl/tmrl#full-environment

# In TMRL, a training pipeline is made of  # In TMRL besteht eine Trainingspipeline aus
# - one Trainer (encompassing the training algorithm that we have coded in this tutorial)  # - einem Trainer (umfasst den Trainingsalgorithmus, den wir in diesem Tutorial codiert haben)
# - one to several RolloutWorker(s) (encompassing our ActorModule and the Gymnasium environment of the competition)  # - ein oder mehrere RolloutWorker (umfasst unser ActorModule und die Gymnasium-Umgebung des Wettbewerbs)
# - one central Server (through which RolloutWorker(s) and Trainer communicate)  # - ein zentraler Server (über den RolloutWorker und Trainer kommunizieren)
# Let us instantiate these via an argument that we will pass when calling this script:  # Lassen Sie uns diese über ein Argument instanziieren, das wir beim Aufrufen dieses Skripts übergeben werden:

if __name__ == "__main__":  # Checks if the script is being run directly (not imported).  # Überprüft, ob das Skript direkt ausgeführt wird (nicht importiert).
    from argparse import ArgumentParser  # Imports the ArgumentParser module for handling command-line arguments.  # Importiert das ArgumentParser-Modul zum Verarbeiten von Befehlszeilenargumenten.

    parser = ArgumentParser()  # Creates an ArgumentParser object to handle arguments.  # Erstellt ein ArgumentParser-Objekt zur Handhabung von Argumenten.
    parser.add_argument('--server', action='store_true', help='launches the server')  # Adds an argument for launching the server.  # Fügt ein Argument zum Starten des Servers hinzu.
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')  # Adds an argument for launching the trainer.  # Fügt ein Argument zum Starten des Trainers hinzu.
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')  # Adds an argument for launching a rollout worker.  # Fügt ein Argument zum Starten eines Rollout-Workers hinzu.
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')  # Adds an argument for launching a worker in standalone mode.  # Fügt ein Argument zum Starten eines Rollout-Workers im Standalone-Modus hinzu.
    args = parser.parse_args()  # Parses the command-line arguments.  # Parst die Befehlszeilenargumente.

    if args.trainer:  # Checks if the 'trainer' argument is provided.  # Überprüft, ob das 'trainer'-Argument übergeben wurde.
        my_trainer = Trainer(training_cls=training_cls,  # Creates a Trainer object with the specified training class.  # Erstellt ein Trainer-Objekt mit der angegebenen Trainingsklasse.
                             server_ip=server_ip_for_trainer,  # Specifies the server IP address for the trainer.  # Gibt die Server-IP-Adresse für den Trainer an.
                             server_port=server_port,  # Specifies the server port for communication.  # Gibt den Serverport für die Kommunikation an.
                             password=password,  # Provides the password for authentication.  # Gibt das Passwort für die Authentifizierung an.
                             security=security)  # Specifies security parameters.  # Gibt Sicherheitsparameter an.
        my_trainer.run()  # Runs the training process.  # Startet den Trainingsprozess.

        # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:  # Hinweis: Wenn Sie Trainingsmetriken an wandb protokollieren möchten, ersetzen Sie my_trainer.run() durch:
        # my_trainer.run_with_wandb(entity=wandb_entity,  # Logs training metrics to wandb.  # Protokolliert Trainingsmetriken an wandb.
        #                           project=wandb_project,  # Specifies the wandb project.  # Gibt das wandb-Projekt an.
        #                           run_id=wandb_run_id)  # Specifies the run ID for wandb.  # Gibt die Run-ID für wandb an.

    elif args.worker or args.test:  # Checks if the 'worker' or 'test' argument is provided.  # Überprüft, ob das 'worker'- oder 'test'-Argument übergeben wurde.
        rw = RolloutWorker(env_cls=env_cls,  # Creates a RolloutWorker with the specified environment class.  # Erstellt einen RolloutWorker mit der angegebenen Umweltklasse.
                           actor_module_cls=MyActorModule,  # Specifies the actor module class.  # Gibt die Actor-Modul-Klasse an.
                           sample_compressor=sample_compressor,  # Specifies the sample compressor for efficiency.  # Gibt den Sample-Kompressor für Effizienz an.
                           device=device_worker,  # Specifies the device for the worker.  # Gibt das Gerät für den Worker an.
                           server_ip=server_ip_for_worker,  # Specifies the server IP address for the worker.  # Gibt die Server-IP-Adresse für den Worker an.
                           server_port=server_port,  # Specifies the server port for the worker.  # Gibt den Serverport für den Worker an.
                           password=password,  # Provides the password for authentication.  # Gibt das Passwort für die Authentifizierung an.
                           security=security,  # Specifies security parameters.  # Gibt Sicherheitsparameter an.
                           max_samples_per_episode=max_samples_per_episode,  # Max samples to collect per episode.  # Maximale Anzahl an Samples pro Episode.
                           obs_preprocessor=obs_preprocessor,  # Preprocessor for observations.  # Vorverarbeitung der Beobachtungen.
                           standalone=args.test)  # Specifies if the worker runs in standalone mode.  # Gibt an, ob der Worker im Standalone-Modus läuft.
        rw.run(test_episode_interval=10)  # Runs the worker with a test episode interval of 10.  # Startet den Worker mit einem Test-Episode-Intervall von 10.

    elif args.server:  # Checks if the 'server' argument is provided.  # Überprüft, ob das 'server'-Argument übergeben wurde.
        import time  # Imports the time module for time delays.  # Importiert das Zeitmodul für Zeitverzögerungen.
        serv = Server(port=server_port,  # Creates a Server object with the specified port.  # Erstellt ein Server-Objekt mit dem angegebenen Port.
                      password=password,  # Provides the password for authentication.  # Gibt das Passwort für die Authentifizierung an.
                      security=security)  # Specifies security parameters.  # Gibt Sicherheitsparameter an.
        while True:  # Starts an infinite loop to keep the server running.  # Startet eine unendliche Schleife, um den Server am Laufen zu halten.
            time.sleep(1.0)  # Pauses the loop for 1 second to avoid high CPU usage.  # Pausiert die Schleife für 1 Sekunde, um hohe CPU-Auslastung zu vermeiden.
