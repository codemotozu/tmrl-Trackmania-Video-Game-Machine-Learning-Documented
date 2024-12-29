# standard library imports
from copy import deepcopy  # Import deepcopy function from the copy module.  # Importiert die deepcopy-Funktion aus dem copy-Modul
from dataclasses import InitVar, dataclass  # Import InitVar and dataclass from dataclasses.  # Importiert InitVar und dataclass aus dem Modul dataclasses

# third-party imports
import numpy as np  # Import numpy for numerical computations.  # Importiert numpy für numerische Berechnungen
import torch  # Import torch for deep learning computations.  # Importiert torch für Deep-Learning-Berechnungen
from torch.distributions import Distribution, Normal  # Import Distribution and Normal from torch.distributions for probabilistic models.  # Importiert Distribution und Normal aus torch.distributions für probabilistische Modelle
from torch.nn import Module  # Import Module from torch.nn to define neural network layers.  # Importiert Module aus torch.nn zur Definition von Neuronalen Netzwerk-Schichten
from torch.nn.init import calculate_gain, kaiming_uniform_, xavier_uniform_  # Import initialization functions for neural networks.  # Importiert Initialisierungsfunktionen für neuronale Netzwerke
from torch.nn.parameter import Parameter  # Import Parameter to define learnable parameters in a model.  # Importiert Parameter, um lernbare Parameter in einem Modell zu definieren

# local imports
from tmrl.util import partial  # Import partial from tmrl.util for partial function application.  # Importiert partial aus tmrl.util für die Teilanwendungsfunktion

def detach(x):  # Define a function to detach tensors from the computation graph.  # Definiert eine Funktion, um Tensoren vom Berechnungsgraphen zu trennen
    if isinstance(x, torch.Tensor):  # Check if x is a tensor.  # Überprüft, ob x ein Tensor ist
        return x.detach()  # Return the detached tensor.  # Gibt den getrennten Tensor zurück
    else:  # If x is not a tensor, apply the function recursively to its elements.  # Wenn x kein Tensor ist, wird die Funktion rekursiv auf seine Elemente angewendet
        return [detach(elem) for elem in x]  # Return a list of detached elements.  # Gibt eine Liste von getrennten Elementen zurück

def no_grad(model):  # Define a function to set requires_grad=False for all parameters in the model.  # Definiert eine Funktion, um requires_grad=False für alle Parameter im Modell festzulegen
    for p in model.parameters():  # Iterate over all parameters in the model.  # Iteriert über alle Parameter im Modell
        p.requires_grad = False  # Set requires_grad to False for each parameter.  # Setzt requires_grad für jeden Parameter auf False
    return model  # Return the model.  # Gibt das Modell zurück

def exponential_moving_average(averages, values, factor):  # Define a function for exponential moving average.  # Definiert eine Funktion für den exponentiellen gleitenden Durchschnitt
    with torch.no_grad():  # Disable gradient computation to avoid unnecessary updates.  # Deaktiviert die Gradientberechnung, um unnötige Updates zu vermeiden
        for a, v in zip(averages, values):  # Iterate over pairs of averages and values.  # Iteriert über Paare von Durchschnittswerten und Werten
            a += factor * (v - a)  # Update the average with the exponential moving average formula.  # Aktualisiert den Durchschnitt mit der Formel für den exponentiellen gleitenden Durchschnitt

def copy_shared(model_a):  # Define a function to create a deepcopy of a model but share the state_dict.  # Definiert eine Funktion, um ein deepcopy eines Modells zu erstellen, aber das state_dict zu teilen
    """Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`."""  # Dokumentation: Erstellt ein deepcopy eines Modells, aber mit gemeinsam genutztem state_dict. Zum Beispiel nützlich in Kombination mit `no_grad`.
    model_b = deepcopy(model_a)  # Create a deepcopy of model_a.  # Erstellt ein deepcopy von model_a
    sda = model_a.state_dict(keep_vars=True)  # Get the state_dict of model_a, keeping variables.  # Holt das state_dict von model_a, wobei Variablen beibehalten werden
    sdb = model_b.state_dict(keep_vars=True)  # Get the state_dict of model_b, keeping variables.  # Holt das state_dict von model_b, wobei Variablen beibehalten werden
    for key in sda:  # Iterate over the keys in the state_dict of model_a.  # Iteriert über die Schlüssel im state_dict von model_a
        a, b = sda[key], sdb[key]  # Get the parameters a and b for each key.  # Holt die Parameter a und b für jeden Schlüssel
        b.data = a.data  # Set b's data to be the same as a's data.  # Setzt b's Daten auf die gleichen wie a's Daten
        assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()  # Assert that both a and b point to the same memory location.  # Überprüft, ob sowohl a als auch b auf denselben Speicherort zeigen
    return model_b  # Return model_b.  # Gibt model_b zurück

class PopArt(Module):  # Define a class PopArt, which extends the Module class in PyTorch.  # Definiert eine Klasse PopArt, die die Module-Klasse in PyTorch erweitert
    """PopArt http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude"""  # Documentation: PopArt method from a research paper.  # Dokumentation: PopArt-Methode aus einem Forschungspapier
    def __init__(self, output_layer, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8):  # Initialize the PopArt class.  # Initialisiert die PopArt-Klasse
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super().__init__()  # Call the parent class's initializer.  # Ruft den Initialisierer der Elternklasse auf
        self.start_pop = start_pop  # Set the start_pop value.  # Setzt den Wert von start_pop
        self.beta = beta  # Set the beta value.  # Setzt den Wert von beta
        self.zero_debias = zero_debias  # Set the zero_debias flag.  # Setzt das zero_debias-Flag
        self.output_layers = output_layer if isinstance(output_layer, (tuple, list, torch.nn.ModuleList)) else (output_layer, )  # Ensure output_layers is a list.  # Stellt sicher, dass output_layers eine Liste ist
        shape = self.output_layers[0].bias.shape  # Get the shape of the bias of the first output layer.  # Holt die Form des Bias der ersten Ausgabeschicht
        device = self.output_layers[0].bias.device  # Get the device of the bias of the first output layer.  # Holt das Gerät des Bias der ersten Ausgabeschicht
        assert all(shape == x.bias.shape for x in self.output_layers)  # Assert that all layers have the same bias shape.  # Überprüft, dass alle Schichten die gleiche Bias-Form haben
        self.mean = Parameter(torch.zeros(shape, device=device), requires_grad=False)  # Initialize the mean parameter.  # Initialisiert den Mittelwert-Parameter
        self.mean_square = Parameter(torch.ones(shape, device=device), requires_grad=False)  # Initialize the mean square parameter.  # Initialisiert den Mittelwert-Quadrat-Parameter
        self.std = Parameter(torch.ones(shape, device=device), requires_grad=False)  # Initialize the standard deviation parameter.  # Initialisiert den Standardabweichungs-Parameter
        self.updates = 0  # Initialize the number of updates.  # Initialisiert die Anzahl der Updates

    @torch.no_grad()  # Decorator to disable gradient computation.  # Dekorator, um die Gradientberechnung zu deaktivieren
    def update(self, targets):  # Define the update method to adjust model parameters.  # Definiert die Update-Methode zur Anpassung der Modell-Parameter
        beta = max(1 / (self.updates + 1), self.beta) if self.zero_debias else self.beta  # Calculate the beta value.  # Berechnet den Wert von beta
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data
        new_mean = (1 - beta) * self.mean + beta * targets.mean(0)  # Update the mean.  # Aktualisiert den Mittelwert
        new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(0)  # Update the mean square.  # Aktualisiert das Mittelwert-Quadrat
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)  # Update the standard deviation.  # Aktualisiert die Standardabweichung

        if self.updates >= self.start_pop:  # If the number of updates exceeds start_pop, apply adjustments to the output layers.  # Wenn die Anzahl der Updates start_pop überschreitet, werden Anpassungen an den Ausgabeschichten vorgenommen
            for layer in self.output_layers:  # Iterate over the output layers.  # Iteriert über die Ausgabeschichten
                layer.weight *= (self.std / new_std)[:, None]  # Scale the weights by the ratio of standard deviations.  # Skaliert die Gewichtungen durch das Verhältnis der Standardabweichungen
                layer.bias *= self.std  # Scale the biases by the standard deviation.  # Skaliert die Biases durch die Standardabweichung
                layer.bias += self.mean - new_mean  # Adjust the biases.  # Passt die Biases an
                layer.bias /= new_std  # Normalize the biases by the new standard deviation.  # Normalisiert die Biases durch die neue Standardabweichung

        self.mean.copy_(new_mean)  # Update the mean.  # Aktualisiert den Mittelwert
        self.mean_square.copy_(new_mean_square)  # Update the mean square.  # Aktualisiert das Mittelwert-Quadrat
        self.std.copy_(new_std)  # Update the standard deviation.  # Aktualisiert die Standardabweichung
        self.updates += 1  # Increment the number of updates.  # Erhöht die Anzahl der Updates
        return self.normalize(targets)  # Return the normalized targets.  # Gibt die normalisierten Ziele zurück

    def normalize(self, x):  # Define a method to normalize the input.  # Definiert eine Methode zur Normalisierung des Eingabewerts
        return (x - self.mean) / self.std  # Normalize by subtracting the mean and dividing by the standard deviation.  # Normalisiert, indem der Mittelwert subtrahiert und durch die Standardabweichung geteilt wird

    def unnormalize(self, x):  # Define a method to unnormalize the input.  # Definiert eine Methode zur Rücknormalisierung des Eingabewerts
        return x * self.std + self.mean  # Unnormalize by multiplying by the standard deviation and adding the mean.  # Rücknormalisiert, indem mit der Standardabweichung multipliziert und der Mittelwert addiert wird

    def normalize_sum(self, s):  # Define a method to normalize the sum.  # Definiert eine Methode zur Normalisierung der Summe
        """normalize x.sum(1) preserving relative weightings between elements"""  # Documentation: Normalize the sum, preserving relative weightings between elements.  # Dokumentation: Normalisiert die Summe und bewahrt dabei die relativen Gewichtungen zwischen den Elementen
        return (s - self.mean.sum()) / self.std.norm()  # Normalize the sum by subtracting the sum of the mean and dividing by the norm of the standard deviation.  # Normalisiert die Summe, indem die Summe des Mittelwerts subtrahiert und durch die Norm der Standardabweichung geteilt wird

# noinspection PyAbstractClass
class TanhNormal(Distribution):  # Class definition: TanhNormal, inheriting from Distribution.  # Klassendefinition: TanhNormal, abgeleitet von Distribution.
    """Distribution of X ~ tanh(Z) where Z ~ N(mean, std)  # Docstring: Defining distribution X ~ tanh(Z) where Z is normally distributed.  # Docstring: Definiert die Verteilung X ~ tanh(Z), wobei Z normalverteilt ist.
    Adapted from https://github.com/vitchyr/rlkit  # Source attribution.  # Quellenangabe.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):  # Constructor for the TanhNormal class.  # Konstruktor der TanhNormal-Klasse.
        self.normal_mean = normal_mean  # Assign mean for the normal distribution.  # Weist den Mittelwert der Normalverteilung zu.
        self.normal_std = normal_std  # Assign standard deviation for the normal distribution.  # Weist die Standardabweichung der Normalverteilung zu.
        self.normal = Normal(normal_mean, normal_std)  # Creates a Normal distribution with the specified mean and standard deviation.  # Erstellt eine Normalverteilung mit dem angegebenen Mittelwert und der Standardabweichung.
        self.epsilon = epsilon  # Assign epsilon for numerical stability.  # Weist Epsilon für numerische Stabilität zu.
        super().__init__(self.normal.batch_shape, self.normal.event_shape)  # Initializes the parent class with batch and event shapes of the normal distribution.  # Initialisiert die Elternklasse mit Batch- und Event-Formen der Normalverteilung.

    def log_prob(self, x):  # Method to compute the log probability of x.  # Methode zur Berechnung der Log-Wahrscheinlichkeit von x.
        if hasattr(x, "pre_tanh_value"):  # Check if x has the attribute pre_tanh_value.  # Überprüft, ob x das Attribut pre_tanh_value hat.
            pre_tanh_value = x.pre_tanh_value  # Use pre_tanh_value if it exists.  # Verwendet pre_tanh_value, falls vorhanden.
        else:  # If pre_tanh_value doesn't exist, calculate it.  # Falls pre_tanh_value nicht existiert, wird es berechnet.
            pre_tanh_value = (torch.log(1 + x + self.epsilon) - torch.log(1 - x + self.epsilon)) / 2  # Computes the inverse tanh value from x.  # Berechnet den inversen Tanh-Wert von x.
        assert x.dim() == 2 and pre_tanh_value.dim() == 2  # Ensures that x and pre_tanh_value have 2 dimensions.  # Stellt sicher, dass x und pre_tanh_value 2 Dimensionen haben.
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - x * x + self.epsilon)  # Calculates log probability using the normal distribution.  # Berechnet die Log-Wahrscheinlichkeit unter Verwendung der Normalverteilung.

    def sample(self, sample_shape=torch.Size()):  # Method to sample from the distribution.  # Methode zum Abtasten aus der Verteilung.
        z = self.normal.sample(sample_shape)  # Sample from the normal distribution.  # Stichprobe aus der Normalverteilung.
        out = torch.tanh(z)  # Apply the tanh function to the sample.  # Wendet die Tanh-Funktion auf die Stichprobe an.
        out.pre_tanh_value = z  # Store the pre-tanh value in the output.  # Speichert den pre-tanh-Wert im Ergebnis.
        return out  # Return the output.  # Gibt das Ergebnis zurück.

    def rsample(self, sample_shape=torch.Size()):  # Method to reparameterize and sample.  # Methode zum Umparametrieren und Abtasten.
        z = self.normal.rsample(sample_shape)  # Sample from the normal distribution with reparameterization.  # Stichprobe aus der Normalverteilung mit Umparametrierung.
        out = torch.tanh(z)  # Apply the tanh function to the sample.  # Wendet die Tanh-Funktion auf die Stichprobe an.
        out.pre_tanh_value = z  # Store the pre-tanh value in the output.  # Speichert den pre-tanh-Wert im Ergebnis.
        return out  # Return the output.  # Gibt das Ergebnis zurück.


# noinspection PyAbstractClass
class Independent(torch.distributions.Independent):  # Class definition for Independent, inheriting from PyTorch's Independent distribution.  # Klassendefinition für Independent, abgeleitet von der PyTorch Independent-Verteilung.
    def sample_test(self):  # Sample test method.  # Methode zum Testen der Stichprobe.
        return torch.tanh(self.base_dist.normal_mean)  # Apply tanh to the mean of the base distribution.  # Wendet Tanh auf den Mittelwert der Basisverteilung an.


class TanhNormalLayer(torch.nn.Module):  # Class defining a custom layer using TanhNormal distribution.  # Klasse zur Definition einer benutzerdefinierten Schicht mit der TanhNormal-Verteilung.
    def __init__(self, n, m):  # Constructor to initialize the layer.  # Konstruktor zur Initialisierung der Schicht.
        super().__init__()  # Initialize the parent class.  # Initialisiert die Elternklasse.

        self.lin_mean = torch.nn.Linear(n, m)  # Defines a linear layer for the mean.  # Definiert eine lineare Schicht für den Mittelwert.
        # self.lin_mean.weight.data  # Commented-out code for accessing weight data.  # Auskommentierter Code zum Zugriff auf Gewichtsdaten.
        # self.lin_mean.bias.data  # Commented-out code for accessing bias data.  # Auskommentierter Code zum Zugriff auf Bias-Daten.

        self.lin_std = torch.nn.Linear(n, m)  # Defines a linear layer for the standard deviation.  # Definiert eine lineare Schicht für die Standardabweichung.
        self.lin_std.weight.data.uniform_(-1e-3, 1e-3)  # Initializes the weights of lin_std with a small uniform distribution.  # Initialisiert die Gewichte von lin_std mit einer kleinen gleichmäßigen Verteilung.
        self.lin_std.bias.data.uniform_(-1e-3, 1e-3)  # Initializes the biases of lin_std with a small uniform distribution.  # Initialisiert die Bias-Werte von lin_std mit einer kleinen gleichmäßigen Verteilung.

    def forward(self, x):  # Forward pass through the layer.  # Vorwärtsdurchlauf durch die Schicht.
        mean = self.lin_mean(x)  # Calculate the mean using the lin_mean layer.  # Berechnet den Mittelwert mit der lin_mean-Schicht.
        log_std = self.lin_std(x)  # Calculate the log of the standard deviation using the lin_std layer.  # Berechnet das Logarithmus der Standardabweichung mit der lin_std-Schicht.
        log_std = torch.clamp(log_std, -20, 2)  # Clamp the log standard deviation to a certain range for stability.  # Begrenzung des Logarithmus der Standardabweichung auf einen bestimmten Bereich zur Stabilität.
        std = torch.exp(log_std)  # Exponentiate to get the standard deviation.  # Exponentiert, um die Standardabweichung zu erhalten.
        # a = TanhTransformedDist(Independent(Normal(m, std), 1))  # Commented-out code for another distribution transformation.  # Auskommentierter Code für eine andere Verteilungsumwandlung.
        a = Independent(TanhNormal(mean, std), 1)  # Define an Independent distribution with TanhNormal.  # Definiert eine unabhängige Verteilung mit TanhNormal.
        return a  # Return the distribution.  # Gibt die Verteilung zurück.


class RlkitLinear(torch.nn.Linear):  # Custom Linear layer for the rlkit framework.  # Benutzerdefinierte lineare Schicht für das rlkit-Framework.
    def __init__(self, *args):  # Constructor to initialize the layer.  # Konstruktor zur Initialisierung der Schicht.
        super().__init__(*args)  # Initialize the parent class.  # Initialisiert die Elternklasse.
        # TODO: investigate the following  # TODO: Untersuchen Sie das Folgende.
        # this mistake seems to be in rlkit too  # Dieser Fehler scheint auch im rlkit vorhanden zu sein.
        # https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py  # Link to the source of the issue.  # Link zur Quelle des Problems.
        fan_in = self.weight.shape[0]  # this is actually fanout!!!  # Get the number of input features (fan-in).  # Holen Sie sich die Anzahl der Eingabefeatures (Fan-in).
        bound = 1. / np.sqrt(fan_in)  # Calculate the bound for weight initialization.  # Berechnet die Grenze für die Gewichtsinitalisierung.
        self.weight.data.uniform_(-bound, bound)  # Initialize weights with a uniform distribution.  # Initialisiert die Gewichte mit einer gleichmäßigen Verteilung.
        self.bias.data.fill_(0.1)  # Set biases to 0.1.  # Setzt die Bias-Werte auf 0,1.


class SacLinear(torch.nn.Linear):  # Custom Linear layer for the SAC (Soft Actor-Critic) algorithm.  # Benutzerdefinierte lineare Schicht für den SAC (Soft Actor-Critic) Algorithmus.
    def __init__(self, in_features, out_features):  # Constructor to initialize the layer.  # Konstruktor zur Initialisierung der Schicht.
        super().__init__(in_features, out_features)  # Initialize the parent class.  # Initialisiert die Elternklasse.
        with torch.no_grad():  # Disable gradient computation for weight initialization.  # Deaktiviert die Gradientenberechnung für die Gewichtsinitalisierung.
            self.weight.uniform_(-0.06, 0.06)  # Initialize weights with a small uniform distribution.  # Initialisiert die Gewichte mit einer kleinen gleichmäßigen Verteilung.
            self.bias.fill_(0.1)  # Set biases to 0.1.  # Setzt die Bias-Werte auf 0,1.


class BasicReLU(torch.nn.Linear):  # Defines a basic linear layer with ReLU activation.  # Definiert eine grundlegende lineare Schicht mit ReLU-Aktivierung.
    def forward(self, x):  # Forward pass through the layer.  # Vorwärtsdurchlauf durch die Schicht.
        x = super().forward(x)  # Apply the parent class's forward method.  # Wendet die Vorwärtsmethode der Elternklasse an.
        return torch.relu(x)  # Apply ReLU activation.  # Wendet die ReLU-Aktivierung an.


class AffineReLU(BasicReLU):  # Affine transformation followed by ReLU activation.  # Affine Transformation gefolgt von ReLU-Aktivierung.
    def __init__(self, in_features, out_features, init_weight_bound: float = 1., init_bias: float = 0.):  # Constructor to initialize the layer.  # Konstruktor zur Initialisierung der Schicht.
        super().__init__(in_features, out_features)  # Initialize the parent class.  # Initialisiert die Elternklasse.
        bound = init_weight_bound / np.sqrt(in_features)  # Calculate bound for weight initialization.  # Berechnet die Grenze für die Gewichtsinitalisierung.
        self.weight.data.uniform_(-bound, bound)  # Initialize weights with a uniform distribution.  # Initialisiert die Gewichte mit einer gleichmäßigen Verteilung.
        self.bias.data.fill_(init_bias)  # Set biases to the specified value.  # Setzt die Bias-Werte auf den angegebenen Wert.


class NormalizedReLU(torch.nn.Sequential):  # Defines a sequential module with Linear, LayerNorm, and ReLU.  # Definiert ein sequentielles Modul mit Linear, LayerNorm und ReLU.
    def __init__(self, in_features, out_features, prenorm_bias=True):  # Constructor for the module.  # Konstruktor für das Modul.
        super().__init__(torch.nn.Linear(in_features, out_features, bias=prenorm_bias), torch.nn.LayerNorm(out_features), torch.nn.ReLU())  # Initialize with Linear, LayerNorm, and ReLU layers.  # Initialisiert mit Linear-, LayerNorm- und ReLU-Schichten.


class KaimingReLU(torch.nn.Linear):  # Defines a linear layer with Kaiming initialization and ReLU activation.  # Definiert eine lineare Schicht mit Kaiming-Initalisierung und ReLU-Aktivierung.
    def __init__(self, in_features, out_features):  # Constructor for the layer.  # Konstruktor für die Schicht.
        super().__init__(in_features, out_features)  # Initialize the parent class.  # Initialisiert die Elternklasse.
        with torch.no_grad():  # Disable gradient computation for initialization.  # Deaktiviert die Gradientenberechnung für die Initialisierung.
            kaiming_uniform_(self.weight)  # Apply Kaiming uniform initialization to the weights.  # Wendet Kaiming gleichmäßige Initialisierung auf die Gewichte an.
            self.bias.fill_(0.)  # Set biases to 0.  # Setzt die Bias-Werte auf 0.

    def forward(self, x):  # Forward pass through the layer.  # Vorwärtsdurchlauf durch die Schicht.
        x = super().forward(x)  # Apply the parent class's forward method.  # Wendet die Vorwärtsmethode der Elternklasse an.
        return torch.relu(x)  # Apply ReLU activation.  # Wendet die ReLU-Aktivierung an.


Linear10 = partial(AffineReLU, init_bias=1.)  # Creates an AffineReLU layer with bias initialized to 1.  # Erstellt eine AffineReLU-Schicht mit auf 1 initialisiertem Bias.
Linear04 = partial(AffineReLU, init_bias=0.4)  # Creates an AffineReLU layer with bias initialized to 0.4.  # Erstellt eine AffineReLU-Schicht mit auf 0.4 initialisiertem Bias.
LinearConstBias = partial(AffineReLU, init_bias=0.1)  # Creates an AffineReLU layer with bias initialized to 0.1.  # Erstellt eine AffineReLU-Schicht mit auf 0.1 initialisiertem Bias.
LinearZeroBias = partial(AffineReLU, init_bias=0.)  # Creates an AffineReLU layer with bias initialized to 0.  # Erstellt eine AffineReLU-Schicht mit auf 0 initialisiertem Bias.
AffineSimon = partial(AffineReLU, init_weight_bound=0.01, init_bias=1.)  # Creates an AffineReLU layer with small weight initialization and bias 1.  # Erstellt eine AffineReLU-Schicht mit kleiner Gewichtsinitalisierung und Bias 1.


def dqn_conv(n):  # Defines convolutional layers for a DQN (Deep Q-Network).  # Definiert Convolutional-Schichten für ein DQN (Deep Q-Network).
    return torch.nn.Sequential(torch.nn.Conv2d(n, 32, kernel_size=8, stride=4), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, kernel_size=4, stride=2), torch.nn.ReLU(),
                               torch.nn.Conv2d(64, 64, kernel_size=3, stride=1), torch.nn.ReLU())  # Sequential convolutional layers with ReLU activations.  # Sequentielle Convolutional-Schichten mit ReLU-Aktivierungen.

def big_conv(n):  # Defines a larger convolutional network.  # Definiert ein größeres Convolutional-Netzwerk.
    return torch.nn.Sequential(  # Returns a sequential container of layers.  # Gibt einen sequentiellen Container von Schichten zurück.
        torch.nn.Conv2d(n, 64, 8, stride=2),  # First convolutional layer: from n input channels to 64 output channels, kernel size 8, stride 2.  # Erste Convolutional-Schicht: Von n Eingabekanälen auf 64 Ausgabekanäle, Kernelgröße 8, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(64, 64, 4, stride=2),  # Second convolutional layer: 64 input and output channels, kernel size 4, stride 2.  # Zweite Convolutional-Schicht: 64 Eingabe- und Ausgabekanäle, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(64, 128, 4, stride=2),  # Third convolutional layer: 64 input channels to 128 output channels, kernel size 4, stride 2.  # Dritte Convolutional-Schicht: 64 Eingabekanäle zu 128 Ausgabekanälen, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(128, 128, 4, stride=1),  # Fourth convolutional layer: 128 input and output channels, kernel size 4, stride 1.  # Vierte Convolutional-Schicht: 128 Eingabe- und Ausgabekanäle, Kernelgröße 4, Stride 1.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
    )  # A series of convolutional layers with LeakyReLU activations.  # Eine Reihe von Convolutional-Schichten mit LeakyReLU-Aktivierungen.


def hd_conv(n):  # Defines a high-dimensional convolutional network.  # Definiert ein hochdimensionales Convolutional-Netzwerk.
    return torch.nn.Sequential(  # Returns a sequential container of layers.  # Gibt einen sequentiellen Container von Schichten zurück.
        torch.nn.Conv2d(n, 32, 8, stride=2),  # First convolutional layer: from n input channels to 32 output channels, kernel size 8, stride 2.  # Erste Convolutional-Schicht: Von n Eingabekanälen auf 32 Ausgabekanäle, Kernelgröße 8, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(32, 64, 4, stride=2),  # Second convolutional layer: 32 input channels to 64 output channels, kernel size 4, stride 2.  # Zweite Convolutional-Schicht: 32 Eingabekanäle zu 64 Ausgabekanälen, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(64, 64, 4, stride=2),  # Third convolutional layer: 64 input and output channels, kernel size 4, stride 2.  # Dritte Convolutional-Schicht: 64 Eingabe- und Ausgabekanäle, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(64, 128, 4, stride=2),  # Fourth convolutional layer: 64 input channels to 128 output channels, kernel size 4, stride 2.  # Vierte Convolutional-Schicht: 64 Eingabekanäle zu 128 Ausgabekanälen, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
        torch.nn.Conv2d(128, 128, 4, stride=2),  # Fifth convolutional layer: 128 input and output channels, kernel size 4, stride 2.  # Fünfte Convolutional-Schicht: 128 Eingabe- und Ausgabekanäle, Kernelgröße 4, Stride 2.
        torch.nn.LeakyReLU(),  # Leaky ReLU activation function for non-linearity.  # Leaky ReLU Aktivierungsfunktion für Nichtlinearität.
    )  # A sequence of convolutional layers with LeakyReLU activations.  # Eine Reihe von Convolutional-Schichten mit LeakyReLU-Aktivierungen.
