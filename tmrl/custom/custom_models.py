# === Trackmania =======================================================================================================

# standard library imports
# Standardbibliothek-Importe

# third-party imports
import numpy as np  # Importing NumPy library for numerical operations.  # Importiert die NumPy-Bibliothek für numerische Operationen.
import torch  # Importing PyTorch for tensor computation and deep learning tasks.  # Importiert PyTorch für Tensorberechnungen und Deep-Learning-Aufgaben.
import torch.nn as nn  # Importing the neural network module from PyTorch.  # Importiert das Neural Network-Modul von PyTorch.
import torch.nn.functional as F  # Importing functional APIs from PyTorch for neural networks.  # Importiert funktionale APIs von PyTorch für neuronale Netzwerke.
from torch.distributions.normal import Normal  # Importing Normal distribution from PyTorch for probabilistic modeling.  # Importiert die Normalverteilung von PyTorch für probabilistische Modellierung.
from math import floor, sqrt  # Importing math functions for mathematical calculations.  # Importiert mathematische Funktionen für Berechnungen.
from torch.nn import Conv2d, Module, ModuleList  # Importing convolutional layers and base classes from PyTorch.  # Importiert Convolutional Layers und Basisklassen von PyTorch.

# local imports
from tmrl.util import prod  # Importing a utility function 'prod' from the local module 'tmrl.util'.  # Importiert die Hilfsfunktion 'prod' aus dem lokalen Modul 'tmrl.util'.
from tmrl.actor import TorchActorModule  # Importing the base class 'TorchActorModule' from 'tmrl.actor'.  # Importiert die Basisklasse 'TorchActorModule' aus 'tmrl.actor'.
import tmrl.config.config_constants as cfg  # Importing configuration constants from the 'config_constants' file in the local 'tmrl.config' module.  # Importiert Konfigurationskonstanten aus der Datei 'config_constants' im lokalen 'tmrl.config'-Modul.

# SUPPORTED ============================================================================================================

# Spinup MLP: =======================================================
# Adapted from the SAC implementation of OpenAI Spinup
# Adaptierung aus der SAC-Implementierung von OpenAI Spinup.

def combined_shape(length, shape=None):  # Function to return combined shape of a tensor based on length and optional shape.  # Funktion, die die kombinierte Form eines Tensors basierend auf Länge und optionaler Form zurückgibt.
    if shape is None:  # Check if shape is not provided.  # Überprüft, ob keine Form angegeben wurde.
        return (length, )  # Return tuple with only length.  # Gibt ein Tupel mit nur der Länge zurück.
    return (length, shape) if np.isscalar(shape) else (length, *shape)  # Returns combined shape based on length and shape, supports scalar shapes.  # Gibt die kombinierte Form basierend auf Länge und Form zurück, unterstützt skalare Formen.

def mlp(sizes, activation, output_activation=nn.Identity):  # Function to create a multi-layer perceptron (MLP).  # Funktion, die ein Multi-Layer Perceptron (MLP) erstellt.
    layers = []  # Initialize empty list for layers.  # Initialisiert eine leere Liste für Schichten.
    for j in range(len(sizes) - 1):  # Loop through sizes and create layers.  # Schleife durch Größen und erstelle Schichten.
        act = activation if j < len(sizes) - 2 else output_activation  # Choose activation function for hidden layers or output layer.  # Wählt die Aktivierungsfunktion für verborgene Schichten oder die Ausgabeschicht aus.
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]  # Add linear layer followed by activation function.  # Fügt eine lineare Schicht gefolgt von einer Aktivierungsfunktion hinzu.
    return nn.Sequential(*layers)  # Return the layers as a sequential model.  # Gibt die Schichten als sequenzielles Modell zurück.

def count_vars(module):  # Function to count the total number of parameters (variables) in a model.  # Funktion zum Zählen der Gesamtzahl der Parameter (Variablen) in einem Modell.
    return sum([np.prod(p.shape) for p in module.parameters()])  # Sum of the product of shapes for each parameter in the module.  # Summe des Produkts der Formen für jeden Parameter im Modul.

LOG_STD_MAX = 2  # Maximum value for the log of standard deviation.  # Maximaler Wert für den Logarithmus der Standardabweichung.
LOG_STD_MIN = -20  # Minimum value for the log of standard deviation.  # Minimaler Wert für den Logarithmus der Standardabweichung.
EPSILON = 1e-7  # A small epsilon value to prevent numerical instability.  # Ein kleiner Epsilon-Wert, um numerische Instabilität zu verhindern.

class SquashedGaussianMLPActor(TorchActorModule):  # Class that implements a Gaussian actor model with squashing function.  # Klasse, die ein Gaußsches Actor-Modell mit Squashing-Funktion implementiert.
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):  # Constructor to initialize the actor with observation space, action space, and layer sizes.  # Konstruktor zur Initialisierung des Actors mit Beobachtungsraum, Aktionsraum und Schichtgrößen.
        super().__init__(observation_space, action_space)  # Call the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        try:  # Try block to handle different types of observation spaces.  # Versuchsblock zur Behandlung verschiedener Arten von Beobachtungsräumen.
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)  # Calculate the total observation space dimension.  # Berechnet die Gesamtbeobachtungsraummodul.
            self.tuple_obs = True  # If observation space is a tuple, set flag to True.  # Wenn der Beobachtungsraum ein Tupel ist, wird das Flag auf True gesetzt.
        except TypeError:  # Handle case where observation space is not a tuple.  # Behandelt den Fall, in dem der Beobachtungsraum kein Tupel ist.
            dim_obs = prod(observation_space.shape)  # Get the product of the dimensions of the observation space.  # Erhält das Produkt der Dimensionen des Beobachtungsraums.
            self.tuple_obs = False  # Set flag to False for non-tuple observation space.  # Setzt das Flag auf False für nicht-Tupel-Beobachtungsräume.
        dim_act = action_space.shape[0]  # Get the number of dimensions of the action space.  # Ermittelt die Anzahl der Dimensionen des Aktionsraums.
        act_limit = action_space.high[0]  # Set the action space limit (maximum value).  # Setzt das Aktionsraumlimit (maximaler Wert).
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)  # Create the MLP for the actor.  # Erstellt das MLP für den Actor.
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)  # Create the layer for the mean of the action distribution.  # Erstellt die Schicht für den Mittelwert der Aktionsverteilung.
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)  # Create the layer for the log of the standard deviation of the action distribution.  # Erstellt die Schicht für den Logarithmus der Standardabweichung der Aktionsverteilung.
        self.act_limit = act_limit  # Store the action limit for scaling the output actions.  # Speichert das Aktionslimit zur Skalierung der Ausgabewerte.

    def forward(self, obs, test=False, with_logprob=True):  # Define the forward pass for the actor model.  # Definiert den Forward-Pass für das Actor-Modell.
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)  # Flatten or concatenate observations based on whether they are tuples.  # Flatten oder konkatenieren der Beobachtungen, je nachdem, ob sie Tupel sind.
        net_out = self.net(x)  # Pass the observation through the MLP network.  # Leitet die Beobachtung durch das MLP-Netzwerk.
        mu = self.mu_layer(net_out)  # Get the mean (mu) of the action distribution.  # Holt den Mittelwert (mu) der Aktionsverteilung.
        log_std = self.log_std_layer(net_out)  # Get the log of the standard deviation (log_std) of the action distribution.  # Holt den Logarithmus der Standardabweichung (log_std) der Aktionsverteilung.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clamp the log_std to ensure it stays within the allowed range.  # Begrenzen des log_std, um sicherzustellen, dass es im zulässigen Bereich bleibt.
        std = torch.exp(log_std)  # Calculate the standard deviation from log_std.  # Berechnet die Standardabweichung aus log_std.

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)  # Define the normal distribution for action sampling.  # Definiert die Normalverteilung für das Abtasten der Aktionen.
        if test:  # If in test mode, use the mean as the action.  # Wenn im Testmodus, verwende den Mittelwert als Aktion.
            pi_action = mu  # Set action to mean (mu).  # Setzt die Aktion auf den Mittelwert (mu).
        else:  # Otherwise, sample from the distribution.  # Andernfalls, sampeln aus der Verteilung.
            pi_action = pi_distribution.rsample()  # Sample an action from the distribution.  # Sampelt eine Aktion aus der Verteilung.

        if with_logprob:  # If log probability is requested, compute it.  # Wenn Log-Wahrscheinlichkeit angefordert wird, berechne sie.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Calculate log probability of the action.  # Berechnet die Log-Wahrscheinlichkeit der Aktion.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # Apply Tanh squashing correction to log probability.  # Wendet die Korrektur des Tanh-Squashings auf die Log-Wahrscheinlichkeit an.
        else:  # If not, set log probability to None.  # Wenn nicht, setze Log-Wahrscheinlichkeit auf None.
            logp_pi = None

        pi_action = torch.tanh(pi_action)  # Apply Tanh to squash the actions between -1 and 1.  # Wendet Tanh an, um die Aktionen zwischen -1 und 1 zu quetschen.
        pi_action = self.act_limit * pi_action  # Scale the actions by the action limit.  # Skaliert die Aktionen durch das Aktionslimit.

        # pi_action = pi_action.squeeze()  # (Optional) Squeeze the action if needed.  # (Optional) Entfernt unnötige Dimensionen von der Aktion.

        return pi_action, logp_pi  # Return the action and log probability.  # Gibt die Aktion und Log-Wahrscheinlichkeit zurück.

    def act(self, obs, test=False):  # Function to select an action based on the observation.  # Funktion zur Auswahl einer Aktion basierend auf der Beobachtung.
        with torch.no_grad():  # Disable gradient computation for inference.  # Deaktiviert die Gradientenberechnung für die Inferenz.
            a, _ = self.forward(obs, test, False)  # Perform forward pass to get the action.  # Führt den Forward-Pass durch, um die Aktion zu erhalten.
            res = a.squeeze().cpu().numpy()  # Convert the action to a NumPy array for further processing.  # Konvertiert die Aktion in ein NumPy-Array für die weitere Verarbeitung.
            if not len(res.shape):  # If the result has no shape (empty), add a new dimension.  # Wenn das Ergebnis keine Form hat (leer), füge eine neue Dimension hinzu.
                res = np.expand_dims(res, 0)  # Expand the dimensions to match the expected shape.  # Erweitert die Dimensionen, um der erwarteten Form zu entsprechen.
            return res  # Return the action as a NumPy array.  # Gibt die Aktion als NumPy-Array zurück.



class MLPQFunction(nn.Module):  # Defines a class for the Q-function using a multi-layer perceptron (MLP).  # Definiert eine Klasse für die Q-Funktion mit einem mehrschichtigen Perzeptron (MLP).
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):  # Initializes the class with observation space, action space, hidden layer sizes, and activation function.  # Initialisiert die Klasse mit dem Beobachtungsraum, dem Aktionsraum, der Größe der versteckten Schichten und der Aktivierungsfunktion.
        super().__init__()  # Calls the parent class's constructor to initialize the module.  # Ruft den Konstruktor der Elternklasse auf, um das Modul zu initialisieren.
        try:  # Tries to calculate the total size of the observation space if it is a tuple.  # Versucht, die Gesamtgröße des Beobachtungsraums zu berechnen, wenn es sich um ein Tupel handelt.
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)  # Calculates the total observation dimension when the space is a tuple.  # Berechnet die Gesamtbeobachtungsdimension, wenn der Raum ein Tupel ist.
            self.tuple_obs = True  # Sets a flag indicating that the observation space is a tuple.  # Setzt ein Flag, das anzeigt, dass der Beobachtungsraum ein Tupel ist.
        except TypeError:  # Handles cases where the observation space is not a tuple.  # Behandelt Fälle, in denen der Beobachtungsraum kein Tupel ist.
            obs_dim = prod(obs_space.shape)  # Calculates the observation dimension when the space is a non-tuple.  # Berechnet die Beobachtungsdimension, wenn der Raum kein Tupel ist.
            self.tuple_obs = False  # Sets a flag indicating that the observation space is not a tuple.  # Setzt ein Flag, das anzeigt, dass der Beobachtungsraum kein Tupel ist.
        act_dim = act_space.shape[0]  # Determines the action dimension from the action space.  # Bestimmt die Aktionsdimension aus dem Aktionsraum.
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)  # Builds the MLP network for the Q-function with input size (observation + action) and output size 1.  # Baut das MLP-Netzwerk für die Q-Funktion mit der Eingabedimension (Beobachtung + Aktion) und der Ausgabedimension 1.
    
    def forward(self, obs, act):  # Defines the forward pass for the Q-function.  # Definiert den Vorwärtsdurchgang für die Q-Funktion.
        x = torch.cat((*obs, act), -1) if self.tuple_obs else torch.cat((torch.flatten(obs, start_dim=1), act), -1)  # Concatenates the observation and action tensors, depending on whether the observation is a tuple or not.  # Verknüpft die Beobachtungs- und Aktions-Tensoren, abhängig davon, ob die Beobachtung ein Tupel ist oder nicht.
        q = self.q(x)  # Passes the concatenated input through the Q-function network.  # Leitet den verknüpften Eingang durch das Q-Funktion-Netzwerk.
        return torch.squeeze(q, -1)  # Removes the last dimension (which is size 1) to ensure the correct output shape.  # Entfernt die letzte Dimension (die Größe 1), um die richtige Ausgabeform sicherzustellen.  # FIXME: understand this
        

class MLPActorCritic(nn.Module):  # Defines a class for an actor-critic model using MLPs.  # Definiert eine Klasse für ein Actor-Critic-Modell mit MLPs.
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):  # Initializes the class with observation space, action space, hidden layer sizes, and activation function.  # Initialisiert die Klasse mit Beobachtungsraum, Aktionsraum, versteckten Schichtgrößen und Aktivierungsfunktion.
        super().__init__()  # Calls the parent class's constructor to initialize the module.  # Ruft den Konstruktor der Elternklasse auf, um das Modul zu initialisieren.
        act_limit = action_space.high[0]  # Retrieves the action limit from the action space.  # Holt sich die Aktionsgrenze aus dem Aktionsraum.
        
        # build policy and value functions  # Bauen Sie die Politik- und Wertfunktionen
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)  # Creates the actor (policy) using an MLP.  # Erstellt den Actor (Politik) mit einem MLP.
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)  # Creates the first Q-function (value function) using an MLP.  # Erstellt die erste Q-Funktion (Wertfunktion) mit einem MLP.
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)  # Creates the second Q-function using an MLP.  # Erstellt die zweite Q-Funktion mit einem MLP.

    def act(self, obs, test=False):  # Defines how the agent acts based on the observation, optionally in test mode.  # Definiert, wie der Agent basierend auf der Beobachtung handelt, optional im Testmodus.
        with torch.no_grad():  # Ensures that no gradients are computed during action selection (for efficiency).  # Stellt sicher, dass keine Gradienten während der Aktionsauswahl berechnet werden (zur Effizienzsteigerung).
            a, _ = self.actor(obs, test, False)  # Computes the action using the actor (policy) model.  # Berechnet die Aktion mit dem Actor (Politik)-Modell.
            res = a.squeeze().cpu().numpy()  # Converts the action to a numpy array, squeezing any single-dimensional entries.  # Konvertiert die Aktion in ein numpy-Array und entfernt ein-dimensionalen Einträge.
            if not len(res.shape):  # If the result is a scalar, reshape it to ensure consistency.  # Wenn das Ergebnis ein Skalar ist, reshaped es, um Konsistenz zu gewährleisten.
                res = np.expand_dims(res, 0)  # Expands the dimensions to ensure the result is always a 1D array.  # Erweitert die Dimensionen, um sicherzustellen, dass das Ergebnis immer ein eindimensionales Array ist.
            return res  # Returns the action as a numpy array.  # Gibt die Aktion als numpy-Array zurück.
    

# REDQ MLP: =====================================================  # REDQ MLP: ===================================================== 

class REDQMLPActorCritic(nn.Module):  # Defines a class for the REDQ algorithm using MLPs for actor and critic.  # Definiert eine Klasse für den REDQ-Algorithmus mit MLPs für Actor und Critic.
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 n=10):  # Initializes the class with observation space, action space, hidden layer sizes, activation function, and number of Q-functions (n).  # Initialisiert die Klasse mit dem Beobachtungsraum, dem Aktionsraum, den versteckten Schichtgrößen, der Aktivierungsfunktion und der Anzahl der Q-Funktionen (n).
        super().__init__()  # Calls the parent class's constructor to initialize the module.  # Ruft den Konstruktor der Elternklasse auf, um das Modul zu initialisieren.
        act_limit = action_space.high[0]  # Retrieves the action limit from the action space.  # Holt sich die Aktionsgrenze aus dem Aktionsraum.
        
        # build policy and value functions  # Bauen Sie die Politik- und Wertfunktionen
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)  # Creates the actor using an MLP.  # Erstellt den Actor mit einem MLP.
        self.n = n  # Sets the number of Q-functions (n).  # Setzt die Anzahl der Q-Funktionen (n).
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])  # Creates a list of n Q-functions (MLPs).  # Erstellt eine Liste von n Q-Funktionen (MLPs).
    
    def act(self, obs, test=False):  # Defines how the agent acts based on the observation, optionally in test mode.  # Definiert, wie der Agent basierend auf der Beobachtung handelt, optional im Testmodus.
        with torch.no_grad():  # Ensures that no gradients are computed during action selection (for efficiency).  # Stellt sicher, dass keine Gradienten während der Aktionsauswahl berechnet werden (zur Effizienzsteigerung).
            a, _ = self.actor(obs, test, False)  # Computes the action using the actor model.  # Berechnet die Aktion mit dem Actor-Modell.
            return a.squeeze().cpu().numpy()  # Converts the action to a numpy array.  # Konvertiert die Aktion in ein numpy-Array.


# CNNs: ================================================================================================================
# EfficientNet =========================================================================================================

# EfficientNetV2 implementation adapted from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
# We use the EfficientNetV2 structure for image features and we merge the TM2020 float features to linear layers
# EffcientNetV2-Implementierung basierend auf dem oben angegebenen GitHub-Link, mit Anpassungen für Bild- und TM2020-Features.

def _make_divisible(v, divisor, min_value=None):  
    """
    This function ensures that all layers have a channel number divisible by the divisor (usually 8).
    Diese Funktion stellt sicher, dass die Kanäle aller Schichten durch den Divisor (meistens 8) teilbar sind.
    """
    if min_value is None:  
        min_value = divisor  # Set minimum value to divisor if not provided.  # Setzt den minimalen Wert auf den Divisor, wenn nicht angegeben.
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  
    # Ensures the value is divisible by the divisor.  # Stellt sicher, dass der Wert durch den Divisor teilbar ist.
    if new_v < 0.9 * v:  
        new_v += divisor  # Adjust if the reduction is too large.  # Passt an, wenn die Reduktion zu groß ist.
    return new_v  # Returns adjusted value.  # Gibt den angepassten Wert zurück.

# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):  
    SiLU = nn.SiLU  # Use PyTorch's built-in SiLU if available.  # Verwendet die eingebaute SiLU, wenn verfügbar.
else:  
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):  
        def forward(self, x):  
            return x * torch.sigmoid(x)  # Swish function: x * sigmoid(x).  # Swish-Funktion: x * sigmoid(x).

class SELayer(nn.Module):  
    def __init__(self, inp, oup, reduction=4):  
        super(SELayer, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling.  # Globale durchschnittliche Pooling.
        self.fc = nn.Sequential(  
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),  
            # First fully connected layer with reduced dimensions.  # Erste vollständig verbundene Schicht mit reduzierten Dimensionen.
            SiLU(),  # SiLU activation.  # SiLU-Aktivierung.
            nn.Linear(_make_divisible(inp // reduction, 8), oup),  
            # Second fully connected layer.  # Zweite vollständig verbundene Schicht.
            nn.Sigmoid()  # Sigmoid activation for scaling.  # Sigmoid-Aktivierung zum Skalieren.
        )

    def forward(self, x):  
        b, c, _, _ = x.size()  # Get batch and channel dimensions.  # Holt Batch- und Kanal-Dimensionen.
        y = self.avg_pool(x).view(b, c)  # Pool and reshape.  # Pooling und Umformen.
        y = self.fc(y).view(b, c, 1, 1)  # Apply FC layers and reshape.  # Wendet FC-Schichten an und formt um.
        return x * y  # Scale input by learned weights.  # Skaliert die Eingabe mit den gelernten Gewichten.

def conv_3x3_bn(inp, oup, stride):  
    return nn.Sequential(  
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  
        # 3x3 convolution with padding.  # 3x3-Kern mit Padding.
        nn.BatchNorm2d(oup),  # Batch normalization.  # Batch-Normalisierung.
        SiLU()  # SiLU activation.  # SiLU-Aktivierung.
    )

def conv_1x1_bn(inp, oup):  
    return nn.Sequential(  
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),  
        # 1x1 convolution.  # 1x1-Kern.
        nn.BatchNorm2d(oup),  # Batch normalization.  # Batch-Normalisierung.
        SiLU()  # SiLU activation.  # SiLU-Aktivierung.
    )

class MBConv(nn.Module):  
    def __init__(self, inp, oup, stride, expand_ratio, use_se):  
        super(MBConv, self).__init__()  
        assert stride in [1, 2]  # Stride must be 1 or 2.  # Stride muss 1 oder 2 sein.

        hidden_dim = round(inp * expand_ratio)  # Compute expanded dimension.  # Berechnet die erweiterte Dimension.
        self.identity = stride == 1 and inp == oup  
        # Check if input can be added to output (identity).  # Prüft, ob Input und Output identisch sind.

        if use_se:  
            self.conv = nn.Sequential(  
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),  
                # Pointwise convolution.  # Punktweise Faltung.
                nn.BatchNorm2d(hidden_dim),  # Batch normalization.  # Batch-Normalisierung.
                SiLU(),  # SiLU activation.  # SiLU-Aktivierung.
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),  
                # Depthwise convolution.  # Tiefenfaltung.
                nn.BatchNorm2d(hidden_dim),  # Batch normalization.  # Batch-Normalisierung.
                SiLU(),  # SiLU activation.  # SiLU-Aktivierung.
                SELayer(inp, hidden_dim),  # Squeeze-and-Excitation layer.  # SE-Schicht.
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  
                # Pointwise linear convolution.  # Punktweise lineare Faltung.
                nn.BatchNorm2d(oup),  # Batch normalization.  # Batch-Normalisierung.
            )
        else:  
            self.conv = nn.Sequential(  
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),  
                # Fused convolution.  # Verschmolzene Faltung.
                nn.BatchNorm2d(hidden_dim),  # Batch normalization.  # Batch-Normalisierung.
                SiLU(),  # SiLU activation.  # SiLU-Aktivierung.
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  
                # Pointwise linear convolution.  # Punktweise lineare Faltung.
                nn.BatchNorm2d(oup),  # Batch normalization.  # Batch-Normalisierung.
            )

    def forward(self, x):  
        if self.identity:  
            return x + self.conv(x)  # Add input to output for identity case.  # Addiert Eingabe und Ausgabe im Identitätsfall.
        else:  
            return self.conv(x)  # Apply convolution stack.  # Wendet den Faltungs-Stack an.






















































class EffNetV2(nn.Module):  # Definition of the EffNetV2 class, inheriting from nn.Module.  # Definition der EffNetV2-Klasse, die von nn.Module erbt.
    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.):  
        # Constructor initializing the model with configurations, input channels, output dimensions, and width multiplier.  
        # Konstruktor zur Initialisierung des Modells mit Konfigurationen, Eingabekanälen, Ausgabedimensionen und Breiten-Multiplikator.
        super(EffNetV2, self).__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        self.cfgs = cfgs  # Stores configuration settings.  # Speichert Konfigurationseinstellungen.

        # building first layer  # Erstellen der ersten Schicht
        input_channel = _make_divisible(24 * width_mult, 8)  
        # Calculates the input channel size adjusted to divisibility by 8.  
        # Berechnet die Eingabekanalgröße, angepasst an die Teilbarkeit durch 8.
        layers = [conv_3x3_bn(nb_channels_in, input_channel, 2)]  
        # Creates the first layer with a 3x3 convolution and batch normalization.  
        # Erstellt die erste Schicht mit einer 3x3 Faltung und Batch-Normalisierung.

        # building inverted residual blocks  # Erstellen invertierter Residualblöcke
        block = MBConv  # Defines the block type to be used.  # Definiert den zu verwendenden Blocktyp.
        for t, c, n, s, use_se in self.cfgs:  
            # Iterates through each configuration in the provided cfgs.  
            # Iteriert durch jede Konfiguration in den bereitgestellten cfgs.
            output_channel = _make_divisible(c * width_mult, 8)  
            # Adjusts the output channel size based on width multiplier and divisibility by 8.  
            # Passt die Ausgabekanalgröße basierend auf dem Breiten-Multiplikator und der Teilbarkeit durch 8 an.
            for i in range(n):  
                # Adds 'n' blocks as specified by the configuration.  
                # Fügt 'n' Blöcke gemäß der Konfiguration hinzu.
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))  
                # Appends a block with stride 's' for the first block, otherwise stride 1.  
                # Fügt einen Block mit Schrittweite 's' für den ersten Block hinzu, sonst Schrittweite 1.
                input_channel = output_channel  # Updates input channel for the next block.  # Aktualisiert den Eingabekanal für den nächsten Block.

        self.features = nn.Sequential(*layers)  
        # Combines all layers into a sequential module.  
        # Kombiniert alle Schichten in einem sequentiellen Modul.

        # building last several layers  # Erstellen der letzten Schichten
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792  
        # Determines the size of the output channel for the final layer.  
        # Bestimmt die Größe des Ausgabekanals für die letzte Schicht.
        self.conv = conv_1x1_bn(input_channel, output_channel)  
        # Adds a 1x1 convolution layer with batch normalization.  
        # Fügt eine 1x1 Faltungsschicht mit Batch-Normalisierung hinzu.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        # Applies global average pooling to reduce spatial dimensions to 1x1.  
        # Wendet globales Durchschnittspooling an, um räumliche Dimensionen auf 1x1 zu reduzieren.
        self.classifier = nn.Linear(output_channel, dim_output)  
        # Final classification layer mapping features to output dimension.  
        # Letzte Klassifikationsschicht, die Merkmale auf die Ausgabedimension abbildet.

        self._initialize_weights()  
        # Initializes model weights.  
        # Initialisiert die Gewichte des Modells.

    def forward(self, x):  
        # Defines the forward pass of the model.  
        # Definiert den Vorwärtsdurchlauf des Modells.
        x = self.features(x)  # Passes input through feature extraction layers.  # Übergibt die Eingabe durch die Feature-Extraktionsschichten.
        x = self.conv(x)  # Applies the 1x1 convolution layer.  # Wendet die 1x1 Faltungsschicht an.
        x = self.avgpool(x)  # Applies average pooling.  # Wendet Durchschnittspooling an.
        x = x.view(x.size(0), -1)  # Flattens the feature map.  # Glättet die Merkmalskarte.
        x = self.classifier(x)  # Passes features through the classifier.  # Übergibt Merkmale durch den Klassifikator.
        return x  # Returns the output.  # Gibt die Ausgabe zurück.

    def _initialize_weights(self):  
        # Initializes weights for layers.  
        # Initialisiert die Gewichte für die Schichten.
        for m in self.modules():  # Iterates through all modules.  # Iteriert durch alle Module.
            if isinstance(m, nn.Conv2d):  
                # Initializes weights for convolution layers using a normal distribution.  
                # Initialisiert Gewichte für Faltungsschichten mit einer Normalverteilung.
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                # Calculates a scaling factor for weight initialization.  
                # Berechnet einen Skalierungsfaktor für die Gewichtsinitialisierung.
                m.weight.data.normal_(0, sqrt(2. / n))  
                # Applies the scaling factor to initialize weights.  
                # Wendet den Skalierungsfaktor zur Initialisierung der Gewichte an.
                if m.bias is not None:  
                    m.bias.data.zero_()  
                    # Sets biases to zero.  
                    # Setzt die Biases auf null.
            elif isinstance(m, nn.BatchNorm2d):  
                m.weight.data.fill_(1)  # Sets batch norm weights to 1.  # Setzt Batch-Norm-Gewichte auf 1.
                m.bias.data.zero_()  # Sets biases to zero.  # Setzt die Biases auf null.
            elif isinstance(m, nn.Linear):  
                m.weight.data.normal_(0, 0.001)  
                # Initializes linear layer weights with small random values.  
                # Initialisiert die Gewichte der linearen Schicht mit kleinen Zufallswerten.
                m.bias.data.zero_()  # Sets biases to zero.  # Setzt die Biases auf null.

def effnetv2_s(**kwargs):  
    # Constructs a small version of EfficientNetV2.  
    # Erstellt eine kleine Version von EfficientNetV2.
    cfgs = [  
        # Configuration: t (expansion), c (channels), n (layers), s (stride), SE (squeeze-and-excitation).  
        # Konfiguration: t (Erweiterung), c (Kanäle), n (Schichten), s (Schrittweite), SE (Squeeze-and-Excitation).
        [1, 24, 2, 1, 0],  
        # t=1, c=24, n=2, s=1, SE=0.  
        # t=1, c=24, n=2, s=1, SE=0.
        [4, 48, 4, 2, 0],  
        # t=4, c=48, n=4, s=2, SE=0.  
        # t=4, c=48, n=4, s=2, SE=0.
        [4, 64, 4, 2, 0],  
        # t=4, c=64, n=4, s=2, SE=0.  
        # t=4, c=64, n=4, s=2, SE=0.
        [4, 128, 6, 2, 1],  
        # t=4, c=128, n=6, s=2, SE=1.  
        # t=4, c=128, n=6, s=2, SE=1.
        [6, 160, 9, 1, 1],  
        # t=6, c=160, n=9, s=1, SE=1.  
        # t=6, c=160, n=9, s=1, SE=1.
        [6, 256, 15, 2, 1],  
        # t=6, c=256, n=15, s=2, SE=1.  
        # t=6, c=256, n=15, s=2, SE=1.
    ]
    return EffNetV2(cfgs, **kwargs)  
    # Returns an instance of EffNetV2 with specified configurations.  
    # Gibt eine Instanz von EffNetV2 mit den angegebenen Konfigurationen zurück.


def effnetv2_m(**kwargs):  
    # Constructs a medium version of EfficientNetV2.  
    # Erstellt eine mittlere Version von EfficientNetV2.
    cfgs = [  
        [1, 24, 3, 1, 0],  
        # t=1, c=24, n=3, s=1, SE=0.  
        # t=1, c=24, n=3, s=1, SE=0.
        [4, 48, 5, 2, 0],  
        # t=4, c=48, n=5, s=2, SE=0.  
        # t=4, c=48, n=5, s=2, SE=0.
        [4, 80, 5, 2, 0],  
        # t=4, c=80, n=5, s=2, SE=0.  
        # t=4, c=80, n=5, s=2, SE=0.
        [4, 160, 7, 2, 1],  
        # t=4, c=160, n=7, s=2, SE=1.  
        # t=4, c=160, n=7, s=2, SE=1.
        [6, 176, 14, 1, 1],  
        # t=6, c=176, n=14, s=1, SE=1.  
        # t=6, c=176, n=14, s=1, SE=1.
        [6, 304, 18, 2, 1],  
        # t=6, c=304, n=18, s=2, SE=1.  
        # t=6, c=304, n=18, s=2, SE=1.
        [6, 512, 5, 1, 1],  
        # t=6, c=512, n=5, s=1, SE=1.  
        # t=6, c=512, n=5, s=1, SE=1.
    ]
    return EffNetV2(cfgs, **kwargs)  
    # Returns an instance of EffNetV2 with medium configurations.  
    # Gibt eine Instanz von EffNetV2 mit mittleren Konfigurationen zurück.


def effnetv2_l(**kwargs):  
    # Constructs a large version of EfficientNetV2.  
    # Erstellt eine große Version von EfficientNetV2.
    cfgs = [  
        [1, 32, 4, 1, 0],  
        # t=1, c=32, n=4, s=1, SE=0.  
        # t=1, c=32, n=4, s=1, SE=0.
        [4, 64, 7, 2, 0],  
        # t=4, c=64, n=7, s=2, SE=0.  
        # t=4, c=64, n=7, s=2, SE=0.
        [4, 96, 7, 2, 0],  
        # t=4, c=96, n=7, s=2, SE=0.  
        # t=4, c=96, n=7, s=2, SE=0.
        [4, 192, 10, 2, 1],  
        # t=4, c=192, n=10, s=2, SE=1.  
        # t=4, c=192, n=10, s=2, SE=1.
        [6, 224, 19, 1, 1],  
        # t=6, c=224, n=19, s=1, SE=1.  
        # t=6, c=224, n=19, s=1, SE=1.
        [6, 384, 25, 2, 1],  
        # t=6, c=384, n=25, s=2, SE=1.  
        # t=6, c=384, n=25, s=2, SE=1.
        [6, 640, 7, 1, 1],  
        # t=6, c=640, n=7, s=1, SE=1.  
        # t=6, c=640, n=7, s=1, SE=1.
    ]
    return EffNetV2(cfgs, **kwargs)  
    # Returns an instance of EffNetV2 with large configurations.  
    # Gibt eine Instanz von EffNetV2 mit großen Konfigurationen zurück.


def effnetv2_xl(**kwargs):  
    # Constructs an extra-large version of EfficientNetV2.  
    # Erstellt eine extra große Version von EfficientNetV2.
    cfgs = [  
        [1, 32, 4, 1, 0],  
        # t=1, c=32, n=4, s=1, SE=0.  
        # t=1, c=32, n=4, s=1, SE=0.
        [4, 64, 8, 2, 0],  
        # t=4, c=64, n=8, s=2, SE=0.  
        # t=4, c=64, n=8, s=2, SE=0.
        [4, 96, 8, 2, 0],  
        # t=4, c=96, n=8, s=2, SE=0.  
        # t=4, c=96, n=8, s=2, SE=0.
        [4, 192, 16, 2, 1],  
        # t=4, c=192, n=16, s=2, SE=1.  
        # t=4, c=192, n=16, s=2, SE=1.
        [6, 256, 24, 1, 1],  
        # t=6, c=256, n=24, s=1, SE=1.  
        # t=6, c=256, n=24, s=1, SE=1.
        [6, 512, 32, 2, 1],  
        # t=6, c=512, n=32, s=2, SE=1.  
        # t=6, c=512, n=32, s=2, SE=1.
        [6, 640, 8, 1, 1],  
        # t=6, c=640, n=8, s=1, SE=1.  
        # t=6, c=640, n=8, s=1, SE=1.
    ]
    return EffNetV2(cfgs, **kwargs)  
    # Returns an instance of EffNetV2 with extra-large configurations.  
    # Gibt eine Instanz von EffNetV2 mit extra großen Konfigurationen zurück.


class SquashedGaussianEffNetActor(TorchActorModule):  # Defines a neural network actor model for reinforcement learning.  # Definiert ein neuronales Netzwerk-Aktormodell für Reinforcement Learning.
    def __init__(self, observation_space, action_space):  # Initializes the actor with observation and action spaces.  # Initialisiert den Actor mit Beobachtungs- und Aktionsräumen.
        super().__init__(observation_space, action_space)  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        dim_act = action_space.shape[0]  # Gets the dimension of the action space.  # Holt die Dimension des Aktionsraums.
        act_limit = action_space.high[0]  # Gets the maximum value of actions in the action space.  # Holt den Maximalwert der Aktionen im Aktionsraum.

        self.cnn = effnetv2_s(nb_channels_in=4, dim_output=247, width_mult=1.).float()  # Defines a CNN backbone (EffNetV2) for image processing.  # Definiert ein CNN-Backbone (EffNetV2) zur Bildverarbeitung.
        self.net = mlp([256, 256], [nn.ReLU, nn.ReLU])  # Defines a fully connected network with two layers and ReLU activation.  # Definiert ein vollständig verbundenes Netzwerk mit zwei Schichten und ReLU-Aktivierung.
        self.mu_layer = nn.Linear(256, dim_act)  # Defines a linear layer to compute mean (mu) of the action distribution.  # Definiert eine lineare Schicht zur Berechnung des Mittelwerts (mu) der Aktionsverteilung.
        self.log_std_layer = nn.Linear(256, dim_act)  # Defines a linear layer to compute the log standard deviation.  # Definiert eine lineare Schicht zur Berechnung der logarithmierten Standardabweichung.
        self.act_limit = act_limit  # Stores the action limit.  # Speichert die Aktionsbegrenzung.

    def forward(self, obs, test=False, with_logprob=True):  # Defines the forward pass of the actor.  # Definiert den Vorwärtsdurchlauf des Aktors.
        imgs_tensor = obs[3].float()  # Extracts image tensor from observation and converts it to float.  # Extrahiert das Bildtensor aus der Beobachtung und wandelt es in Float um.
        float_tensors = (obs[0], obs[1], obs[2], *obs[4:])  # Extracts non-image tensors from observation.  # Extrahiert Nicht-Bild-Tensoren aus der Beobachtung.
        float_tensor = torch.cat(float_tensors, -1).float()  # Concatenates the non-image tensors along the last dimension.  # Verbindet die Nicht-Bild-Tensoren entlang der letzten Dimension.
        cnn_out = self.cnn(imgs_tensor)  # Passes the image tensor through the CNN.  # Führt das Bildtensor durch das CNN.
        mlp_in = torch.cat((cnn_out, float_tensor), -1)  # Concatenates CNN output with other inputs.  # Verbindet die CNN-Ausgabe mit anderen Eingaben.
        net_out = self.net(mlp_in)  # Passes the concatenated tensor through the MLP.  # Führt den zusammengefügten Tensor durch das MLP.
        mu = self.mu_layer(net_out)  # Computes the mean (mu) of the action distribution.  # Berechnet den Mittelwert (mu) der Aktionsverteilung.
        log_std = self.log_std_layer(net_out)  # Computes the log standard deviation.  # Berechnet die logarithmierte Standardabweichung.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clamps log_std to predefined bounds.  # Begrenzt log_std auf vordefinierte Werte.
        std = torch.exp(log_std)  # Computes the standard deviation by exponentiating log_std.  # Berechnet die Standardabweichung durch Exponentiation von log_std.

        pi_distribution = Normal(mu, std)  # Creates a normal distribution with mean mu and std.  # Erstellt eine Normalverteilung mit Mittelwert mu und std.
        if test:  # Checks if in test mode.  # Überprüft, ob der Testmodus aktiviert ist.
            pi_action = mu  # In test mode, the action is the mean.  # Im Testmodus ist die Aktion der Mittelwert.
        else:
            pi_action = pi_distribution.rsample()  # Samples an action using reparameterization.  # Wählt eine Aktion mit Reparametrisierung aus.

        if with_logprob:  # If log probability is required.  # Wenn die Log-Wahrscheinlichkeit benötigt wird.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Computes the log probability of the action.  # Berechnet die Log-Wahrscheinlichkeit der Aktion.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # Applies Tanh correction for log probability.  # Wendet Tanh-Korrektur für die Log-Wahrscheinlichkeit an.
        else:
            logp_pi = None  # Sets log probability to None.  # Setzt die Log-Wahrscheinlichkeit auf None.

        pi_action = torch.tanh(pi_action)  # Applies Tanh to squash action values.  # Wendet Tanh an, um Aktionswerte zu begrenzen.
        pi_action = self.act_limit * pi_action  # Scales the action to match the action limit.  # Skaliert die Aktion entsprechend der Aktionsbegrenzung.

        return pi_action, logp_pi  # Returns the action and log probability.  # Gibt die Aktion und die Log-Wahrscheinlichkeit zurück.

    def act(self, obs, test=False):  # Computes the action given an observation.  # Berechnet die Aktion basierend auf einer Beobachtung.
        import sys  # Imports the system module.  # Importiert das Systemmodul.
        size = sys.getsizeof(obs)  # Gets the size of the observation.  # Holt die Größe der Beobachtung.
        with torch.no_grad():  # Disables gradient computation.  # Deaktiviert die Gradientenberechnung.
            a, _ = self.forward(obs, test, False)  # Calls the forward method without log probability.  # Ruft die forward-Methode ohne Log-Wahrscheinlichkeit auf.
            return a.squeeze().cpu().numpy()  # Converts the action to a numpy array.  # Konvertiert die Aktion in ein Numpy-Array.

class EffNetQFunction(nn.Module):  # Defines the Q-function network.  # Definiert das Q-Funktionsnetzwerk.
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):  # Initializes the Q-function.  # Initialisiert die Q-Funktion.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)  # Calculates the total dimension of the observation space.  # Berechnet die gesamte Dimension des Beobachtungsraums.
        act_dim = act_space.shape[0]  # Gets the action space dimension.  # Holt die Dimension des Aktionsraums.
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)  # Creates an MLP for the Q-function.  # Erstellt ein MLP für die Q-Funktion.

    def forward(self, obs, act):  # Defines the forward pass of the Q-function.  # Definiert den Vorwärtsdurchlauf der Q-Funktion.
        x = torch.cat((*obs, act), -1)  # Concatenates observations and actions.  # Verbindet Beobachtungen und Aktionen.
        q = self.q(x)  # Passes the concatenated input through the MLP.  # Führt die zusammengefügte Eingabe durch das MLP.
        return torch.squeeze(q, -1)  # Ensures the output has the correct shape.  # Stellt sicher, dass die Ausgabe die richtige Form hat.

class EffNetActorCritic(nn.Module):  # Combines actor and critic networks.  # Kombiniert Actor- und Critic-Netzwerke.
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):  # Initializes the actor-critic model.  # Initialisiert das Actor-Critic-Modell.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        act_limit = action_space.high[0]  # Gets the action space limit.  # Holt die Aktionsraumgrenze.

        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)  # Creates the actor.  # Erstellt den Actor.
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)  # Creates the first Q-function.  # Erstellt die erste Q-Funktion.
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)  # Creates the second Q-function.  # Erstellt die zweite


def act(self, obs, test=False):  # Defines a method 'act' that takes 'obs' (observation) and 'test' as inputs.  # Definiert eine Methode 'act', die 'obs' (Beobachtung) und 'test' als Eingaben nimmt.
    with torch.no_grad():  # Context manager to disable gradient calculation, saving memory and computation.  # Kontext-Manager, um die Berechnung der Gradienten zu deaktivieren, wodurch Speicher und Rechenleistung gespart werden.
        a, _ = self.actor(obs, test, False)  # Calls the 'actor' method of the object with 'obs', 'test', and False as inputs. Returns action 'a' and ignores the second output.  # Ruft die 'actor'-Methode des Objekts mit 'obs', 'test' und False als Eingaben auf. Gibt die Aktion 'a' zurück und ignoriert die zweite Ausgabe.
        return a.squeeze().cpu().numpy()  # Removes any singleton dimensions from 'a', moves it to CPU, and converts it to a NumPy array.  # Entfernt alle Singleton-Dimensionen aus 'a', verschiebt es auf die CPU und konvertiert es in ein NumPy-Array.


# Vanilla CNN FOR GRAYSCALE IMAGES: ====================================================================================

# Function to calculate the number of flat features in a tensor after convolution
def num_flat_features(x):  
    size = x.size()[1:]  # Get the size of all dimensions except the batch size.  # Die Größe aller Dimensionen außer der Batch-Größe erhalten
    num_features = 1  # Initialize the variable to store the total number of features.  # Die Variable initialisieren, um die Gesamtzahl der Merkmale zu speichern
    for s in size:  # Iterate over the dimensions (height, width, etc.).  # Über die Dimensionen (Höhe, Breite usw.) iterieren
        num_features *= s  # Multiply the current size with the total number of features.  # Die aktuelle Größe mit der Gesamtzahl der Merkmale multiplizieren
    return num_features  # Return the total number of features.  # Die Gesamtzahl der Merkmale zurückgeben


# Function to calculate the output dimensions of a convolutional layer
def conv2d_out_dims(conv_layer, h_in, w_in):  
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)  # Calculate output height.  # Berechnung der Ausgabeleitungshöhe
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)  # Calculate output width.  # Berechnung der Ausgabebreite
    return h_out, w_out  # Return the output dimensions.  # Die Ausgabedimensionen zurückgeben


# Vanilla CNN class for grayscale images
class VanillaCNN(Module):  
    def __init__(self, q_net):  
        super(VanillaCNN, self).__init__()  # Initialize the parent class.  # Die Elternklasse initialisieren
        self.q_net = q_net  # Store the flag for whether q_net is being used.  # Das Flag speichern, ob q_net verwendet wird
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH  # Initialize height and width from config.  # Höhe und Breite aus der Konfiguration initialisieren
        hist = cfg.IMG_HIST_LEN  # Get the length of the image history from config.  # Die Länge der Bildhistorie aus der Konfiguration holen

        self.conv1 = Conv2d(hist, 64, 8, stride=2)  # First convolutional layer with input channels `hist`, output channels 64, kernel size 8x8, stride 2.  # Erste Convolutional-Schicht mit Eingabekanälen `hist`, Ausgabekanälen 64, Kernelgröße 8x8, Schrittweite 2
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)  # Update output dimensions after first conv layer.  # Die Ausgabedimensionen nach der ersten Convolutional-Schicht aktualisieren
        self.conv2 = Conv2d(64, 64, 4, stride=2)  # Second convolutional layer with input 64 channels, output 64 channels, kernel size 4x4, stride 2.  # Zweite Convolutional-Schicht mit Eingabekanälen 64, Ausgabekanälen 64, Kernelgröße 4x4, Schrittweite 2
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)  # Update output dimensions after second conv layer.  # Die Ausgabedimensionen nach der zweiten Convolutional-Schicht aktualisieren
        self.conv3 = Conv2d(64, 128, 4, stride=2)  # Third convolutional layer with input 64 channels, output 128 channels, kernel size 4x4, stride 2.  # Dritte Convolutional-Schicht mit Eingabekanälen 64, Ausgabekanälen 128, Kernelgröße 4x4, Schrittweite 2
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)  # Update output dimensions after third conv layer.  # Die Ausgabedimensionen nach der dritten Convolutional-Schicht aktualisieren
        self.conv4 = Conv2d(128, 128, 4, stride=2)  # Fourth convolutional layer with input 128 channels, output 128 channels, kernel size 4x4, stride 2.  # Vierte Convolutional-Schicht mit Eingabekanälen 128, Ausgabekanälen 128, Kernelgröße 4x4, Schrittweite 2
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)  # Update output dimensions after fourth conv layer.  # Die Ausgabedimensionen nach der vierten Convolutional-Schicht aktualisieren
        self.out_channels = self.conv4.out_channels  # Store the number of output channels from the last conv layer.  # Die Anzahl der Ausgabekanäle der letzten Convolutional-Schicht speichern
        self.flat_features = self.out_channels * self.h_out * self.w_out  # Calculate the total number of features after flattening.  # Die Gesamtzahl der Merkmale nach dem Flatten berechnen
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9  # Decide the number of features based on whether q_net is used.  # Die Anzahl der Merkmale je nachdem, ob q_net verwendet wird, festlegen
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]  # Define MLP layer architecture based on q_net.  # Die Architektur der MLP-Schichten je nach q_net definieren
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)  # Initialize the multi-layer perceptron (MLP).  # Das Multi-Layer Perceptron (MLP) initialisieren

    # Forward pass function to process the input through the network
    def forward(self, x):  
        if self.q_net:  
            speed, gear, rpm, images, act1, act2, act = x  # Extract inputs when using q_net.  # Eingaben extrahieren, wenn q_net verwendet wird
        else:  
            speed, gear, rpm, images, act1, act2 = x  # Extract inputs when not using q_net.  # Eingaben extrahieren, wenn q_net nicht verwendet wird

        x = F.relu(self.conv1(images))  # Apply ReLU activation after the first convolution.  # ReLU-Aktivierung nach der ersten Convolution anwenden
        x = F.relu(self.conv2(x))  # Apply ReLU activation after the second convolution.  # ReLU-Aktivierung nach der zweiten Convolution anwenden
        x = F.relu(self.conv3(x))  # Apply ReLU activation after the third convolution.  # ReLU-Aktivierung nach der dritten Convolution anwenden
        x = F.relu(self.conv4(x))  # Apply ReLU activation after the fourth convolution.  # ReLU-Aktivierung nach der vierten Convolution anwenden
        flat_features = num_flat_features(x)  # Flatten the output of the last convolution layer to calculate features.  # Das Ausgabeergebnis der letzten Convolution-Schicht flach machen, um Merkmale zu berechnen
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"  # Ensure the flattened features match the expected number.  # Sicherstellen, dass die flachen Merkmale der erwarteten Anzahl entsprechen
        x = x.view(-1, flat_features)  # Reshape the tensor for input into the MLP.  # Den Tensor umformen, um ihn in das MLP einzugeben
        if self.q_net:  
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)  # Concatenate all inputs when using q_net.  # Alle Eingaben zusammenführen, wenn q_net verwendet wird
        else:  
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)  # Concatenate all inputs when not using q_net.  # Alle Eingaben zusammenführen, wenn q_net nicht verwendet wird
        x = self.mlp(x)  # Pass the concatenated inputs through the MLP.  # Die zusammengeführten Eingaben durch das MLP weitergeben
        return x  # Return the output after processing through the MLP.  # Das Ergebnis nach der Verarbeitung durch das MLP zurückgeben



class SquashedGaussianVanillaCNNActor(TorchActorModule):  # Defines a class for an actor with a squashed Gaussian distribution and a CNN architecture.  # Definiert eine Klasse für einen Akteur mit einer gesquashten Gauß-Verteilung und einer CNN-Architektur.
    def __init__(self, observation_space, action_space):  # Initializes the actor, receives observation and action spaces as inputs.  # Initialisiert den Akteur und erhält Beobachtungs- und Aktionsräume als Eingaben.
        super().__init__(observation_space, action_space)  # Calls the parent class constructor with the observation and action spaces.  # Ruft den Konstruktor der Elternklasse mit den Beobachtungs- und Aktionsräumen auf.
        dim_act = action_space.shape[0]  # Extracts the dimension of the action space.  # Extrahiert die Dimension des Aktionsraums.
        act_limit = action_space.high[0]  # Gets the upper limit of the action space.  # Holt sich die obere Grenze des Aktionsraums.
        self.net = VanillaCNN(q_net=False)  # Initializes a Vanilla CNN, used for feature extraction.  # Initialisiert ein Vanilla CNN, das für die Merkmalsextraktion verwendet wird.
        self.mu_layer = nn.Linear(256, dim_act)  # Defines a linear layer to output the mean of the Gaussian distribution.  # Definiert eine lineare Schicht, die den Mittelwert der Gauß-Verteilung ausgibt.
        self.log_std_layer = nn.Linear(256, dim_act)  # Defines a linear layer to output the log standard deviation of the Gaussian distribution.  # Definiert eine lineare Schicht, die die logarithmische Standardabweichung der Gauß-Verteilung ausgibt.
        self.act_limit = act_limit  # Sets the action limit for scaling the actions.  # Legt das Aktionslimit zum Skalieren der Aktionen fest.

    def forward(self, obs, test=False, with_logprob=True):  # Defines the forward pass through the network.  # Definiert den Vorwärtsdurchlauf durch das Netzwerk.
        net_out = self.net(obs)  # Passes the observation through the CNN to extract features.  # Gibt die Beobachtung durch das CNN, um Merkmale zu extrahieren.
        mu = self.mu_layer(net_out)  # Computes the mean (mu) of the action distribution.  # Berechnet den Mittelwert (mu) der Aktionsverteilung.
        log_std = self.log_std_layer(net_out)  # Computes the log standard deviation (log_std) of the action distribution.  # Berechnet die logarithmische Standardabweichung (log_std) der Aktionsverteilung.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clamps the log_std to ensure it stays within valid bounds.  # Beschränkt log_std, damit es innerhalb gültiger Grenzen bleibt.
        std = torch.exp(log_std)  # Computes the standard deviation from the log_std.  # Berechnet die Standardabweichung aus log_std.

        pi_distribution = Normal(mu, std)  # Creates a normal distribution using mu and std.  # Erstellt eine Normalverteilung mit mu und std.
        if test:  # If in test mode, use the mean for action.  # Wenn im Testmodus, verwende den Mittelwert für die Aktion.
            pi_action = mu  # Assigns the mean as the action.  # Weist den Mittelwert der Aktion zu.
        else:  # Otherwise, sample from the distribution.  # Andernfalls, ziehe eine Probe aus der Verteilung.
            pi_action = pi_distribution.rsample()  # Samples an action from the distribution.  # Ziehe eine Aktion aus der Verteilung.

        if with_logprob:  # If log probabilities are requested, calculate them.  # Wenn Log-Wahrscheinlichkeiten angefordert sind, berechne diese.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Computes the log probability of the sampled action.  # Berechnet die Log-Wahrscheinlichkeit der gezogenen Aktion.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # A correction formula for the log probability.  # Eine Korrekturformel für die Log-Wahrscheinlichkeit.
        else:  # If log probabilities are not requested, set to None.  # Wenn Log-Wahrscheinlichkeiten nicht angefordert sind, setze auf None.
            logp_pi = None

        pi_action = torch.tanh(pi_action)  # Applies the tanh activation function to squash the action within [-1, 1].  # Wendet die Tanh-Aktivierungsfunktion an, um die Aktion auf [-1, 1] zu beschränken.
        pi_action = self.act_limit * pi_action  # Scales the action by the action limit.  # Skaliert die Aktion mit dem Aktionslimit.

        return pi_action, logp_pi  # Returns the action and log probability.  # Gibt die Aktion und die Log-Wahrscheinlichkeit zurück.

    def act(self, obs, test=False):  # Returns the action to take based on an observation.  # Gibt die Aktion zurück, die basierend auf einer Beobachtung ausgeführt werden soll.
        with torch.no_grad():  # Disables gradient calculation to save memory during inference.  # Deaktiviert die Gradientenberechnung, um während der Inferenz Speicher zu sparen.
            a, _ = self.forward(obs, test, False)  # Calls forward pass and returns action.  # Ruft den Vorwärtsdurchlauf auf und gibt die Aktion zurück.
            return a.squeeze().cpu().numpy()  # Squeezes the action tensor and converts to a numpy array.  # Komprimiert den Aktionstensor und konvertiert ihn in ein Numpy-Array.


class VanillaCNNQFunction(nn.Module):  # Defines a class for the Q-function using a Vanilla CNN.  # Definiert eine Klasse für die Q-Funktion unter Verwendung eines Vanilla CNN.
    def __init__(self, observation_space, action_space):  # Initializes the Q-function, receives observation and action spaces.  # Initialisiert die Q-Funktion und erhält Beobachtungs- und Aktionsräume als Eingaben.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        self.net = VanillaCNN(q_net=True)  # Initializes a Vanilla CNN specifically for the Q-function.  # Initialisiert ein Vanilla CNN, das speziell für die Q-Funktion verwendet wird.

    def forward(self, obs, act):  # Defines the forward pass for the Q-function.  # Definiert den Vorwärtsdurchlauf für die Q-Funktion.
        x = (*obs, act)  # Concatenates observation and action as input.  # Verkettet Beobachtung und Aktion als Eingabe.
        q = self.net(x)  # Passes the concatenated input through the CNN.  # Gibt die verketten Eingabe durch das CNN.
        return torch.squeeze(q, -1)  # Ensures the output has the correct shape.  # Stellt sicher, dass die Ausgabe die richtige Form hat.


class VanillaCNNActorCritic(nn.Module):  # Defines an Actor-Critic model with a Vanilla CNN.  # Definiert ein Actor-Critic-Modell mit einem Vanilla CNN.
    def __init__(self, observation_space, action_space):  # Initializes the Actor-Critic model.  # Initialisiert das Actor-Critic-Modell.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.

        # build policy and value functions  # Erbaut die Policy- und Wertfunktionen
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)  # Initializes the actor using the Squashed Gaussian Vanilla CNN actor.  # Initialisiert den Akteur unter Verwendung des Squashed Gaussian Vanilla CNN-Akteurs.
        self.q1 = VanillaCNNQFunction(observation_space, action_space)  # Initializes the first Q-function.  # Initialisiert die erste Q-Funktion.
        self.q2 = VanillaCNNQFunction(observation_space, action_space)  # Initializes the second Q-function.  # Initialisiert die zweite Q-Funktion.

    def act(self, obs, test=False):  # Returns the action to take based on an observation.  # Gibt die Aktion zurück, die basierend auf einer Beobachtung ausgeführt werden soll.
        with torch.no_grad():  # Disables gradient calculation to save memory during inference.  # Deaktiviert die Gradientenberechnung, um während der Inferenz Speicher zu sparen.
            a, _ = self.actor(obs, test, False)  # Calls the actor to get an action.  # Ruft den Akteur auf, um eine Aktion zu erhalten.
            return a.squeeze().cpu().numpy()  # Squeezes the action tensor and converts to a numpy array.  # Komprimiert den Aktionstensor und konvertiert ihn in ein Numpy-Array.

# Vanilla CNN FOR COLOR IMAGES: ========================================================================================


def remove_colors(images):  # Function definition to remove color channels from images.  # Funktion zur Definition des Entfernens der Farbkanäle aus Bildern.
    """  # Docstring explaining the function purpose.  # Docstring, der den Zweck der Funktion erklärt.
    We remove colors so that we can simply use the same structure as the grayscale model.  # Removing color to simplify model processing.  # Entfernen von Farben, um die Modellstruktur wie beim Graustufenmodell zu vereinfachen.
    
    The "color" default pipeline is mostly here for support, as our model effectively gets rid of 2 channels out of 3.  # Default pipeline retains color support, but the model only uses one channel.  # Die Standard-Pipeline behält die Farbabstimmung bei, aber das Modell nutzt nur einen Kanal.
    If you actually want to use colors, do not use the default pipeline.  # Warning to use a custom model if color is needed.  # Warnung, dass bei Farbnutzung ein benutzerdefiniertes Modell erforderlich ist.
    Instead, you need to code a custom model that doesn't get rid of them.  # Suggestion to create a custom model for color handling.  # Vorschlag, ein benutzerdefiniertes Modell für die Farbverarbeitung zu erstellen.
    """
    images = images[:, :, :, :, 0]  # Keep only the first channel (grayscale).  # Behalte nur den ersten Kanal (Graustufen).
    return images  # Return the processed images with colors removed.  # Rückgabe der verarbeiteten Bilder ohne Farben.

class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):  # Define a custom actor class inheriting from base actor.  # Definiere eine benutzerdefinierte Schauspielerklasse, die von der Basis-Klasse erbt.
    def forward(self, obs, test=False, with_logprob=True):  # Forward method for processing input observations.  # Vorwärtsmethode zur Verarbeitung von Eingabebeobachtungen.
        speed, gear, rpm, images, act1, act2 = obs  # Unpack observation tuple into individual components.  # Entpacke das Beobachtungs-Tuple in einzelne Komponenten.
        images = remove_colors(images)  # Remove color channels from the image part of the observations.  # Entferne Farbkanäle aus den Bildkomponenten der Beobachtungen.
        obs = (speed, gear, rpm, images, act1, act2)  # Repackage the modified observations.  # Packe die modifizierten Beobachtungen wieder zusammen.
        return super().forward(obs, test=False, with_logprob=True)  # Call the superclass forward method with modified observations.  # Rufe die Vorwärtsmethode der Oberklasse mit den modifizierten Beobachtungen auf.

class VanillaColorCNNQFunction(VanillaCNNQFunction):  # Define a custom Q-function class inheriting from base Q-function.  # Definiere eine benutzerdefinierte Q-Funktion, die von der Basis-Q-Funktion erbt.
    def forward(self, obs, act):  # Forward method to process observations and actions.  # Vorwärtsmethode zur Verarbeitung von Beobachtungen und Aktionen.
        speed, gear, rpm, images, act1, act2 = obs  # Unpack observation tuple into individual components.  # Entpacke das Beobachtungs-Tuple in einzelne Komponenten.
        images = remove_colors(images)  # Remove color channels from the image part of the observations.  # Entferne Farbkanäle aus den Bildkomponenten der Beobachtungen.
        obs = (speed, gear, rpm, images, act1, act2)  # Repackage the modified observations.  # Packe die modifizierten Beobachtungen wieder zusammen.
        return super().forward(obs, act)  # Call the superclass forward method with modified observations and actions.  # Rufe die Vorwärtsmethode der Oberklasse mit modifizierten Beobachtungen und Aktionen auf.

class VanillaColorCNNActorCritic(VanillaCNNActorCritic):  # Define a custom Actor-Critic class inheriting from base class.  # Definiere eine benutzerdefinierte Actor-Critic-Klasse, die von der Basis-Klasse erbt.
    def __init__(self, observation_space, action_space):  # Initialization method for the Actor-Critic class.  # Initialisierungsmethode für die Actor-Critic-Klasse.
        super().__init__(observation_space, action_space)  # Call the superclass initialization.  # Rufe die Initialisierung der Oberklasse auf.

        # build policy and value functions  # Build policy and value networks.  # Baue die Politik- und Wertfunktionen.
        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)  # Initialize actor with a custom color CNN.  # Initialisiere den Schauspieler mit einem benutzerdefinierten Farb-CNN.
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)  # Initialize first Q-function with a custom color CNN.  # Initialisiere die erste Q-Funktion mit einem benutzerdefinierten Farb-CNN.
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)  # Initialize second Q-function with a custom color CNN.  # Initialisiere die zweite Q-Funktion mit einem benutzerdefinierten Farb-CNN.


# UNSUPPORTED ==========================================================================================================


# RNN: ==========================================================

def rnn(input_size, rnn_size, rnn_len):  # Defines the function to create an RNN with specific input size, hidden size, and number of layers.  # Definiert die Funktion zur Erstellung eines RNN mit spezifischer Eingabegröße, verborgener Größe und Anzahl der Schichten.
    """
    sizes is ignored for now, expect first values and length  # This docstring tells that size values are ignored, focusing on the first values and length of the RNN.  # Dieser Docstring besagt, dass Größenwerte derzeit ignoriert werden, wobei der Fokus auf den ersten Werten und der Länge des RNN liegt.
    """
    num_rnn_layers = rnn_len  # Sets the number of RNN layers based on the parameter rnn_len.  # Legt die Anzahl der RNN-Schichten basierend auf dem Parameter rnn_len fest.
    assert num_rnn_layers >= 1  # Ensures that the number of layers is at least 1.  # Stellt sicher, dass die Anzahl der Schichten mindestens 1 ist.
    hidden_size = rnn_size  # Defines the hidden size for the RNN.  # Definiert die verborgene Größe für das RNN.

    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)  # Creates a GRU (Gated Recurrent Unit) with the specified parameters.  # Erstellt eine GRU (Gated Recurrent Unit) mit den angegebenen Parametern.
    return gru  # Returns the constructed GRU model.  # Gibt das erstellte GRU-Modell zurück.


class SquashedGaussianRNNActor(nn.Module):  # Defines a class for a neural network actor that uses an RNN and outputs a squashed Gaussian distribution.  # Definiert eine Klasse für einen neuronalen Netzwerk-Actor, der ein RNN verwendet und eine zusammengedrückte Gauß-Verteilung ausgibt.
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):  # Initializes the actor with observation space, action space, RNN parameters, MLP layers, and activation function.  # Initialisiert den Actor mit Beobachtungsraum, Aktionsraum, RNN-Parametern, MLP-Schichten und Aktivierungsfunktion.
        super().__init__()  # Calls the parent class initialization method.  # Ruft die Initialisierungsmethode der Elternklasse auf.
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)  # Calculates the total dimension of the observation space by multiplying the dimensions of each observation.  # Berechnet die Gesamtgröße des Beobachtungsraums, indem die Dimensionen jeder Beobachtung multipliziert werden.
        dim_act = act_space.shape[0]  # Gets the dimension of the action space.  # Holt sich die Dimension des Aktionsraums.
        act_limit = act_space.high[0]  # Sets the action limit from the maximum value of the action space.  # Legt das Aktionslimit aus dem Höchstwert des Aktionsraums fest.
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)  # Initializes the RNN using the given observation dimension, RNN size, and RNN length.  # Initialisiert das RNN mit der angegebenen Beobachtungsdimension, RNN-Größe und RNN-Länge.
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)  # Creates a Multi-Layer Perceptron (MLP) with the specified sizes and activation functions.  # Erstellt ein Multi-Layer Perceptron (MLP) mit den angegebenen Größen und Aktivierungsfunktionen.
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)  # Initializes the output layer for the mean (mu) of the action.  # Initialisiert die Ausgabeschicht für den Mittelwert (mu) der Aktion.
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)  # Initializes the output layer for the log of the standard deviation (log_std).  # Initialisiert die Ausgabeschicht für den Logarithmus der Standardabweichung (log_std).
        self.act_limit = act_limit  # Sets the action limit.  # Setzt das Aktionslimit.
        self.h = None  # Initializes the hidden state to None.  # Initialisiert den verborgenen Zustand auf None.
        self.rnn_size = rnn_size  # Stores the RNN size.  # Speichert die RNN-Größe.
        self.rnn_len = rnn_len  # Stores the RNN length.  # Speichert die RNN-Länge.

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):  # Defines the forward pass for the actor, computing the action distribution.  # Definiert den Vorwärtspass für den Actor, der die Aktionsverteilung berechnet.
        """
        obs: observation  # The observation input.  # Die Beobachtungs-Eingabe.
        h: hidden state  # The hidden state of the RNN.  # Der verborgene Zustand des RNN.
        Returns:
            pi_action, log_pi, h  # Returns the action, log probability of the action, and the updated hidden state.  # Gibt die Aktion, die Log-Wahrscheinlichkeit der Aktion und den aktualisierten verborgenen Zustand zurück.
        """
        self.rnn.flatten_parameters()  # Optimizes memory usage in the RNN.  # Optimiert die Speichernutzung im RNN.

        batch_size = obs_seq[0].shape[0]  # Gets the batch size from the first observation in the sequence.  # Holt sich die Batch-Größe aus der ersten Beobachtung in der Sequenz.

        if not save_hidden or self.h is None:  # Checks if the hidden state should be saved or needs initialization.  # Überprüft, ob der verborgene Zustand gespeichert werden soll oder initialisiert werden muss.
            device = obs_seq[0].device  # Gets the device (CPU or GPU) where the tensor is located.  # Holt sich das Gerät (CPU oder GPU), auf dem der Tensor gespeichert ist.
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)  # Initializes the hidden state with zeros.  # Initialisiert den verborgenen Zustand mit Nullen.
        else:  
            h = self.h  # Uses the previous hidden state if available.  # Verwendet den vorherigen verborgenen Zustand, falls dieser verfügbar ist.

        obs_seq_cat = torch.cat(obs_seq, -1)  # Concatenates the observation sequence into a single tensor.  # Verkettet die Beobachtungssequenz zu einem einzelnen Tensor.
        net_out, h = self.rnn(obs_seq_cat, h)  # Passes the concatenated observations through the RNN.  # Leitet die verketteten Beobachtungen durch das RNN.
        net_out = net_out[:, -1]  # Takes the last output of the RNN (last time step).  # Nimmt die letzte Ausgabe des RNN (letzter Zeitschritt).
        net_out = self.mlp(net_out)  # Passes the RNN output through the MLP.  # Leitet die RNN-Ausgabe durch das MLP.
        mu = self.mu_layer(net_out)  # Calculates the mean of the action from the MLP output.  # Berechnet den Mittelwert der Aktion aus der MLP-Ausgabe.
        log_std = self.log_std_layer(net_out)  # Calculates the log of the standard deviation from the MLP output.  # Berechnet den Logarithmus der Standardabweichung aus der MLP-Ausgabe.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # Clips the log of the standard deviation to a predefined range.  # Beschränkt den Logarithmus der Standardabweichung auf einen vordefinierten Bereich.
        std = torch.exp(log_std)  # Exponentiates the log of the standard deviation to get the actual standard deviation.  # Exponentiert den Logarithmus der Standardabweichung, um die tatsächliche Standardabweichung zu erhalten.

        # Pre-squash distribution and sample  # Pre-squash Verteilung und Stichprobe
        pi_distribution = Normal(mu, std)  # Creates a Normal distribution with the calculated mean and standard deviation.  # Erstellt eine Normalverteilung mit dem berechneten Mittelwert und der Standardabweichung.
        if test:  # If it's in test mode, the action is the mean of the distribution.  # Wenn es sich im Testmodus befindet, ist die Aktion der Mittelwert der Verteilung.
            pi_action = mu  # Sets the action to the mean.  # Setzt die Aktion auf den Mittelwert.
        else:  
            pi_action = pi_distribution.rsample()  # Samples from the distribution during training.  # Zieht eine Stichprobe aus der Verteilung während des Trainings.

        if with_logprob:  # If log probability is requested, calculate it.  # Wenn die Log-Wahrscheinlichkeit angefordert wird, berechne sie.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # Computes the log probability of the sampled action.  # Berechnet die Log-Wahrscheinlichkeit der gezogenen Aktion.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # Applies a correction for the Tanh squashing.  # Wendet eine Korrektur für das Tanh-Squashing an.
        else:  
            logp_pi = None  # If no log probability is needed, set it to None.  # Wenn keine Log-Wahrscheinlichkeit benötigt wird, setze sie auf None.

        pi_action = torch.tanh(pi_action)  # Applies the Tanh function to the action.  # Wendet die Tanh-Funktion auf die Aktion an.
        pi_action = self.act_limit * pi_action  # Scales the action by the action limit.  # Skaliert die Aktion mit dem Aktionslimit.

        pi_action = pi_action.squeeze()  # Removes any extra dimensions from the action tensor.  # Entfernt alle zusätzlichen Dimensionen aus dem Aktions-Tensor.

        if save_hidden:  # If hidden state should be saved, store it.  # Wenn der verborgene Zustand gespeichert werden soll, speichere ihn.
            self.h = h  # Saves the hidden state for the next step.  # Speichert den verborgenen Zustand für den nächsten Schritt.

        return pi_action, logp_pi  # Returns the action and log probability.  # Gibt die Aktion und die Log-Wahrscheinlichkeit zurück.

    def act(self, obs, test=False):  # Defines the action method which provides the action for given observations.  # Definiert die Aktionsmethode, die die Aktion für gegebene Beobachtungen liefert.
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # Reshapes the observations to add a sequence dimension.  # Ändert die Form der Beobachtungen, um eine Sequenzdimension hinzuzufügen.
        with torch.no_grad():  # Disables gradient calculation for this operation.  # Deaktiviert die Berechnung von Gradienten für diese Operation.
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)  # Gets the action from the forward pass.  # Holt sich die Aktion aus dem Vorwärtspass.
            return a.squeeze().cpu().numpy()  # Removes extra dimensions and converts the action to a numpy array.  # Entfernt zusätzliche Dimensionen und konvertiert die Aktion in ein numpy-Array.


class RNNQFunction(nn.Module):  # Class definition for RNNQFunction which is a neural network model.  # Deutsch: Klassendefinition für RNNQFunction, welches ein neuronales Netzwerk-Modell ist.
    """
    The action is merged in the latent space after the RNN  # Action wird nach dem RNN im latenten Raum zusammengeführt.
    """
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):  # Initialization method to set up the network structure.  # Initialisierungsmethode, um die Netzwerkstruktur festzulegen.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)  # Calculate the dimensionality of the observation space by summing up the product of dimensions of each observation.  # Berechnet die Dimensionalität des Beobachtungsraums, indem das Produkt der Dimensionen jeder Beobachtung summiert wird.
        dim_act = act_space.shape[0]  # Get the dimensionality of the action space.  # Bestimmt die Dimensionalität des Aktionsraums.
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)  # Initialize RNN layer with the observation dimension, rnn size, and length.  # Initialisiert die RNN-Schicht mit der Beobachtungsdimension, der RNN-Größe und der Länge.
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)  # Create a multi-layer perceptron (MLP) with specified layers and activation function.  # Erstellen eines Multi-Layer Perceptron (MLP) mit angegebenen Schichten und Aktivierungsfunktion.
        self.h = None  # Initialize hidden state to None.  # Initialisiert den verborgenen Zustand auf None.
        self.rnn_size = rnn_size  # Store the RNN size.  # Speichert die RNN-Größe.
        self.rnn_len = rnn_len  # Store the RNN length.  # Speichert die RNN-Länge.

    def forward(self, obs_seq, act, save_hidden=False):  # Forward method to perform a pass through the network.  # Vorwärtsmethode, um einen Durchgang durch das Netzwerk zu machen.
        """
        obs: observation  # obs: Beobachtung
        h: hidden state  # h: Verborgener Zustand
        Returns:  # Gibt zurück:
            pi_action, log_pi, h  # pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()  # Flattens the parameters of the RNN to optimize memory usage.  # Flacht die Parameter des RNN ab, um die Speichernutzung zu optimieren.

        batch_size = obs_seq[0].shape[0]  # Get the batch size from the first observation in the sequence.  # Bestimmt die Batchgröße aus der ersten Beobachtung in der Sequenz.

        if not save_hidden or self.h is None:  # Check if hidden state should be saved or initialized.  # Überprüft, ob der verborgene Zustand gespeichert oder initialisiert werden soll.
            device = obs_seq[0].device  # Get the device (e.g., CPU or GPU) where the observation is located.  # Bestimmt das Gerät (z.B. CPU oder GPU), auf dem sich die Beobachtung befindet.
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)  # Initialize hidden state as a zero tensor with the appropriate shape.  # Initialisiert den verborgenen Zustand als Null-Tensor mit der entsprechenden Form.
        else:  # Use the previously saved hidden state if available.  # Verwendet den zuvor gespeicherten verborgenen Zustand, falls verfügbar.
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)  # Concatenate the observation sequence along the last dimension.  # Verkettet die Beobachtungssequenz entlang der letzten Dimension.

        net_out, h = self.rnn(obs_seq_cat, h)  # Pass the concatenated observations and hidden state through the RNN.  # Leitet die verknüpften Beobachtungen und den verborgenen Zustand durch das RNN.
        net_out = net_out[:, -1]  # Select the last output of the RNN sequence (final state).  # Wählt die letzte Ausgabe der RNN-Sequenz (finaler Zustand) aus.
        net_out = torch.cat((net_out, act), -1)  # Concatenate the last RNN output with the action.  # Verkettet die letzte RNN-Ausgabe mit der Aktion.
        q = self.mlp(net_out)  # Pass the concatenated vector through the MLP.  # Leitet den verknüpften Vektor durch das MLP.

        if save_hidden:  # If saving the hidden state is requested.  # Wenn das Speichern des verborgenen Zustands angefordert wird.
            self.h = h  # Save the hidden state for the next forward pass.  # Speichert den verborgenen Zustand für den nächsten Vorwärtspass.

        return torch.squeeze(q, -1)  # Squeeze the output tensor to ensure it has the correct shape.  # Komprimiert den Ausgabetensor, um sicherzustellen, dass er die richtige Form hat.


class RNNActorCritic(nn.Module):  # Class definition for RNNActorCritic model.  # Klassendefinition für das RNNActorCritic-Modell.
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):  # Initialization method for RNNActorCritic.  # Initialisierungsmethode für RNNActorCritic.
        super().__init__()  # Calls the parent class constructor.  # Ruft den Konstruktor der Elternklasse auf.

        act_limit = action_space.high[0]  # Get the action limit from the action space.  # Bestimmt das Aktionslimit aus dem Aktionsraum.

        # build policy and value functions  # Erstelle Politik- und Wertfunktionen
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)  # Initializes the actor network with specified parameters.  # Initialisiert das Actor-Netzwerk mit den angegebenen Parametern.
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)  # Initializes the first Q-function network.  # Initialisiert das erste Q-Funktionsnetzwerk.
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)  # Initializes the second Q-function network.  # Initialisiert das zweite Q-Funktionsnetzwerk.
