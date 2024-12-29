# standard library imports
import itertools  # itertools provides efficient looping and combinatorial functions in Python.  # importiert itertools, eine Bibliothek für effizientes Schleifen und Kombinatorik in Python
from copy import deepcopy  # deepcopy is used to create a new object with the same values as the original, without referencing the same memory.  # deepcopy wird verwendet, um ein neues Objekt mit denselben Werten wie das Original zu erstellen, jedoch ohne auf denselben Speicher zu verweisen
from dataclasses import dataclass  # dataclass decorator is used to simplify the creation of classes that are primarily containers for data.  # dataclass-Dekorator vereinfacht die Erstellung von Klassen, die hauptsächlich Container für Daten sind

# third-party imports
import numpy as np  # numpy is a library for numerical computing in Python, providing support for large matrices and high-level mathematical functions.  # numpy ist eine Bibliothek für numerische Berechnungen in Python, die Unterstützung für große Matrizen und hochentwickelte mathematische Funktionen bietet
import torch  # torch is a library for machine learning, providing a framework for tensors and deep learning.  # torch ist eine Bibliothek für maschinelles Lernen, die ein Framework für Tensoren und Deep Learning bietet
from torch.optim import Adam, AdamW, SGD  # Adam, AdamW, and SGD are popular optimization algorithms used in training machine learning models.  # Adam, AdamW und SGD sind beliebte Optimierungsalgorithmen, die beim Training von maschinellen Lernmodellen verwendet werden

# local imports
import tmrl.custom.custom_models as core  # custom models from a local library used in training algorithms.  # benutzerdefinierte Modelle aus einer lokalen Bibliothek, die in Trainingsalgorithmen verwendet werden
from tmrl.custom.utils.nn import copy_shared, no_grad  # utility functions for managing neural network states.  # Hilfsfunktionen zum Verwalten von Zuständen von neuronalen Netzen
from tmrl.util import cached_property, partial  # cached_property creates a computed property that is cached. partial creates a function with some arguments frozen.  # cached_property erstellt eine berechnete Eigenschaft, die zwischengespeichert wird. partial erstellt eine Funktion, bei der einige Argumente "eingefroren" sind
from tmrl.training import TrainingAgent  # TrainingAgent is a base class for training reinforcement learning agents.  # TrainingAgent ist eine Basisklasse für das Training von Reinforcement-Learning-Agenten
import tmrl.config.config_constants as cfg  # imports configuration constants from a local module.  # importiert Konfigurationskonstanten aus einem lokalen Modul

import logging  # logging module is used to log messages during program execution.  # logging-Modul wird verwendet, um Nachrichten während der Programmausführung zu protokollieren


# Soft Actor-Critic ====================================================================================================

@dataclass(eq=0)  # Dataclass decorator to generate an immutable class without a default equality method.  # Dataclass-Dekorator zur Erstellung einer unveränderlichen Klasse ohne Standard-Gleichheitsmethode
class SpinupSacAgent(TrainingAgent):  # Adapted from Spinup, this class represents an SAC (Soft Actor-Critic) agent.  # Aus Spinup adaptiert, stellt diese Klasse einen SAC (Soft Actor-Critic) Agenten dar
    observation_space: type  # Type of the observation space (e.g., continuous or discrete).  # Typ des Beobachtungsraums (z.B. kontinuierlich oder diskret)
    action_space: type  # Type of the action space (e.g., continuous or discrete).  # Typ des Aktionsraums (z.B. kontinuierlich oder diskret)
    device: str = None  # device where the model will live (None for auto).  # Gerät, auf dem das Modell lebt (None für automatisch)
    model_cls: type = core.MLPActorCritic  # The class for the model (e.g., MLPActorCritic).  # Die Klasse für das Modell (z.B. MLPActorCritic)
    gamma: float = 0.99  # Discount factor for future rewards.  # Diskontfaktor für zukünftige Belohnungen
    polyak: float = 0.995  # Polyak averaging coefficient for target networks.  # Polyak-Averaging-Koeffizient für Zielnetzwerke
    alpha: float = 0.2  # Fixed or initial value of the entropy coefficient.  # Festwert oder Anfangswert des Entropie-Koeffizienten
    lr_actor: float = 1e-3  # Learning rate for the actor network.  # Lernrate für das Actor-Netzwerk
    lr_critic: float = 1e-3  # Learning rate for the critic network.  # Lernrate für das Critic-Netzwerk
    lr_entropy: float = 1e-3  # Learning rate for entropy autotuning (SAC v2).  # Lernrate für die Entropie-Autotuning (SAC v2)
    learn_entropy_coef: bool = True  # If True, SAC v2 is used, else, SAC v1 is used.  # Wenn True, wird SAC v2 verwendet, andernfalls SAC v1
    target_entropy: float = None  # If None, the target entropy for SAC v2 is set automatically.  # Wenn None, wird die Zielentropie für SAC v2 automatisch festgelegt
    optimizer_actor: str = "adam"  # Optimizer for the actor network (options: "adam", "adamw", "sgd").  # Optimierer für das Actor-Netzwerk (Optionen: "adam", "adamw", "sgd")
    optimizer_critic: str = "adam"  # Optimizer for the critic network (options: "adam", "adamw", "sgd").  # Optimierer für das Critic-Netzwerk (Optionen: "adam", "adamw", "sgd")
    betas_actor: tuple = None  # Betas for Adam and AdamW optimizers for the actor.  # Betas für die Adam- und AdamW-Optimierer des Actors
    betas_critic: tuple = None  # Betas for Adam and AdamW optimizers for the critic.  # Betas für die Adam- und AdamW-Optimierer des Critics
    l2_actor: float = None  # L2 regularization (weight decay) for the actor.  # L2-Regularisierung (Gewichtsverfall) für den Actor
    l2_critic: float = None  # L2 regularization (weight decay) for the critic.  # L2-Regularisierung (Gewichtsverfall) für den Critic

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))  # Cached property that returns a copy of the model with no gradient tracking.  # Zwischengespeicherte Eigenschaft, die eine Kopie des Modells ohne Gradiententracking zurückgibt

    def __post_init__(self):  # Called automatically after initialization of the class.  # Wird automatisch nach der Initialisierung der Klasse aufgerufen
        observation_space, action_space = self.observation_space, self.action_space  # Store observation and action space types.  # Speichert die Typen des Beobachtungs- und Aktionsraums
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to CUDA (GPU) if available, else CPU.  # Setzt das Gerät auf CUDA (GPU), wenn verfügbar, andernfalls auf CPU
        model = self.model_cls(observation_space, action_space)  # Initialize the model based on the class provided.  # Initialisiert das Modell basierend auf der angegebenen Klasse
        logging.debug(f" device SAC: {device}")  # Logs the device type being used.  # Protokolliert den verwendeten Gerätetyp
        self.model = model.to(device)  # Moves the model to the chosen device.  # Verschiebt das Modell zum gewählten Gerät
        self.model_target = no_grad(deepcopy(self.model))  # Creates a target model without gradient tracking.  # Erstellt ein Zielmodell ohne Gradiententracking

        # Set up optimizers for policy and q-function:
        self.optimizer_actor = self.optimizer_actor.lower()  # Convert the actor optimizer to lowercase for consistency.  # Wandelt den Actor-Optimierer in Kleinbuchstaben um, um Konsistenz zu gewährleisten
        self.optimizer_critic = self.optimizer_critic.lower()  # Convert the critic optimizer to lowercase for consistency.  # Wandelt den Critic-Optimierer in Kleinbuchstaben um, um Konsistenz zu gewährleisten
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:  # Checks if the provided actor optimizer is valid.  # Überprüft, ob der angegebene Actor-Optimierer gültig ist
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")  # Logs a warning if the actor optimizer is invalid.  # Protokolliert eine Warnung, wenn der Actor-Optimierer ungültig ist
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:  # Checks if the provided critic optimizer is valid.  # Überprüft, ob der angegebene Critic-Optimierer gültig ist
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")  # Logs a warning if the critic optimizer is invalid.  # Protokolliert eine Warnung, wenn der Critic-Optimierer ungültig ist
        if self.optimizer_actor == "adam":  # Checks if the actor optimizer is Adam.  # Überprüft, ob der Actor-Optimierer Adam ist
            pi_optimizer_cls = Adam  # Sets the optimizer class to Adam for the actor.  # Setzt die Optimierer-Klasse auf Adam für den Actor
        elif self.optimizer_actor == "adamw":  # Checks if the actor optimizer is AdamW.  # Überprüft, ob der Actor-Optimierer AdamW ist
            pi_optimizer_cls = AdamW  # Sets the optimizer class to AdamW for the actor.  # Setzt die Optimierer-Klasse auf AdamW für den Actor
        else:  # Default to SGD for the actor optimizer.  # Standardmäßig SGD für den Actor-Optimierer
            pi_optimizer_cls = SGD  # Sets the optimizer class to SGD for the actor.  # Setzt die Optimierer-Klasse auf SGD für den Actor
        pi_optimizer_kwargs = {"lr": self.lr_actor}  # Initializes the actor optimizer with the specified learning rate.  # Initialisiert den Actor-Optimierer mit der angegebenen Lernrate
        if self.optimizer_actor in ["adam", "adamw"] and self.betas_actor is not None:  # If the actor uses Adam or AdamW, include betas in optimizer.  # Wenn der Actor Adam oder AdamW verwendet, werden die Betas im Optimierer eingeschlossen
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)  # Adds the betas to the actor optimizer's arguments.  # Fügt die Betas den Argumenten des Actor-Optimierers hinzu
        if self.l2_actor is not None:  # Checks if L2 regularization is specified for the actor.  # Überprüft, ob eine L2-Regularisierung für den Actor angegeben wurde
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor  # Adds the L2 regularization (weight decay) to the actor optimizer.  # Fügt die L2-Regularisierung (Gewichtsverfall) dem Actor-Optimierer hinzu

        if self.optimizer_critic == "adam":  # Checks if the critic optimizer is Adam.  # Überprüft, ob der Critic-Optimierer Adam ist
            q_optimizer_cls = Adam  # Sets the optimizer class to Adam for the critic.  # Setzt die Optimierer-Klasse auf Adam für den Critic
        elif self.optimizer_critic == "adamw":  # Checks if the critic optimizer is AdamW.  # Überprüft, ob der Critic-Optimierer AdamW ist
            q_optimizer_cls = AdamW  # Sets the optimizer class to AdamW for the critic.  # Setzt die Optimierer-Klasse auf AdamW für den Critic
        else:  # Default to SGD for the critic optimizer.  # Standardmäßig SGD für den Critic-Optimierer
            q_optimizer_cls = SGD  # Sets the optimizer class to SGD for the critic.  # Setzt die Optimierer-Klasse auf SGD für den Critic
        q_optimizer_kwargs = {"lr": self.lr_critic}  # Initializes the critic optimizer with the specified learning rate.  # Initialisiert den Critic-Optimierer mit der angegebenen Lernrate
        if self.optimizer_critic in ["adam", "adamw"] and self.betas_critic is not None:  # If the critic uses Adam or AdamW, include betas in optimizer.  # Wenn der Critic Adam oder AdamW verwendet, werden die Betas im Optimierer eingeschlossen
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)  # Adds the betas to the critic optimizer's arguments.  # Fügt die Betas den Argumenten des Critic-Optimierers hinzu
        if self.l2_critic is not None:  # Checks if L2 regularization is specified for the critic.  # Überprüft, ob eine L2-Regularisierung für den Critic angegeben wurde
            q_optimizer_kwargs["weight_decay"] = self.l2_critic  # Adds the L2 regularization (weight decay) to the critic optimizer.  # Fügt die L2-Regularisierung (Gewichtsverfall) dem Critic-Optimierer hinzu

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)  # Initializes the actor optimizer with the specified arguments.  # Initialisiert den Actor-Optimierer mit den angegebenen Argumenten
        self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)  # Initializes the critic optimizer with the specified arguments.  # Initialisiert den Critic-Optimierer mit den angegebenen Argumenten

        # entropy coefficient:
        if self.target_entropy is None:  # Checks if the target entropy is not specified.  # Überprüft, ob die Zielentropie nicht angegeben wurde
            self.target_entropy = -np.prod(action_space.shape)  # Sets the target entropy automatically based on the action space.  # Setzt die Zielentropie automatisch basierend auf dem Aktionsraum
        else:  # If target entropy is provided, convert it to float.  # Wenn die Zielentropie angegeben ist, wird sie in einen Float umgewandelt
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:  # Checks if the entropy coefficient should be learned.  # Überprüft, ob der Entropie-Koeffizient gelernt werden soll
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)  # Initializes the log of the entropy coefficient as a learnable parameter.  # Initialisiert das Logarithmus des Entropie-Koeffizienten als lernbaren Parameter
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)  # Creates an optimizer for the entropy coefficient.  # Erstellt einen Optimierer für den Entropie-Koeffizienten
        else:  # If entropy coefficient is fixed, convert to tensor.  # Wenn der Entropie-Koeffizient festgelegt ist, wird er in einen Tensor umgewandelt
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):  # Returns the actor part of the model without gradient tracking.  # Gibt den Actor-Teil des Modells ohne Gradiententracking zurück
        return self.model_nograd.actor  # Returns the actor from the model without gradient tracking.  # Gibt den Actor aus dem Modell ohne Gradiententracking zurück


def train(self, batch):  # Define the train method that takes a batch of data as input.  # Definiert die Trainingsmethode, die ein Batch von Daten als Eingabe nimmt.
    o, a, r, o2, d, _ = batch  # Unpack the batch into variables: o = observations, a = actions, r = rewards, o2 = next observations, d = done flags.  # Entpacke das Batch in Variablen: o = Beobachtungen, a = Aktionen, r = Belohnungen, o2 = nächste Beobachtungen, d = Fertig-Flags.
    
    pi, logp_pi = self.model.actor(o)  # Get the action (pi) and log-probability (logp_pi) from the actor model given the observations.  # Hole die Aktion (pi) und Log-Wahrscheinlichkeit (logp_pi) aus dem Actor-Modell anhand der Beobachtungen.
    # FIXME? log_prob = log_prob.reshape(-1, 1)  # This line seems to be a placeholder or a future fix for reshaping log_prob.  # Diese Zeile scheint ein Platzhalter oder eine zukünftige Korrektur für das Umformen von log_prob zu sein.
    
    loss_alpha = None  # Initialize the loss for entropy coefficient.  # Initialisiere den Verlust für den Entropie-Koeffizienten.
    if self.learn_entropy_coef:  # Check if entropy coefficient learning is enabled.  # Überprüfe, ob das Lernen des Entropie-Koeffizienten aktiviert ist.
        alpha_t = torch.exp(self.log_alpha.detach())  # Detach log_alpha from the computation graph and apply the exponential function.  # Trenne log_alpha vom Berechnungsgraphen und wende die Exponentialfunktion an.
        loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()  # Calculate the entropy loss.  # Berechne den Entropie-Verlust.
    else:  # If entropy coefficient learning is disabled, use the pre-set alpha_t.  # Wenn das Lernen des Entropie-Koeffizienten deaktiviert ist, verwende das voreingestellte alpha_t.
        alpha_t = self.alpha_t  # Use predefined alpha_t.  # Verwende das vordefinierte alpha_t.

    if loss_alpha is not None:  # If the loss_alpha is defined, optimize the entropy coefficient.  # Wenn loss_alpha definiert ist, optimiere den Entropie-Koeffizienten.
        self.alpha_optimizer.zero_grad()  # Zero out the gradients for the alpha optimizer.  # Setze die Gradienten des Alpha-Optimierers zurück.
        loss_alpha.backward()  # Compute the gradients for the loss_alpha.  # Berechne die Gradienten für den Verlust alpha.
        self.alpha_optimizer.step()  # Perform the optimization step.  # Führe den Optimierungsschritt durch.

    q1 = self.model.q1(o, a)  # Get the Q-values from the first Q-network.  # Hole die Q-Werte aus dem ersten Q-Netzwerk.
    q2 = self.model.q2(o, a)  # Get the Q-values from the second Q-network.  # Hole die Q-Werte aus dem zweiten Q-Netzwerk.

    with torch.no_grad():  # Disable gradient computation for the following operations.  # Deaktiviere die Gradientenberechnung für die folgenden Operationen.
        a2, logp_a2 = self.model.actor(o2)  # Get the action and log-probability for the next state using the actor.  # Hole die Aktion und Log-Wahrscheinlichkeit für den nächsten Zustand mit dem Actor.
        
        q1_pi_targ = self.model_target.q1(o2, a2)  # Get the target Q-values from the target Q-network.  # Hole die Ziel-Q-Werte aus dem Ziel-Q-Netzwerk.
        q2_pi_targ = self.model_target.q2(o2, a2)  # Get the target Q-values from the second target Q-network.  # Hole die Ziel-Q-Werte aus dem zweiten Ziel-Q-Netzwerk.
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # Take the minimum of the two target Q-values.  # Nimm das Minimum der beiden Ziel-Q-Werte.
        backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)  # Compute the Bellman backup for the Q-values.  # Berechne das Bellman-Backup für die Q-Werte.

    loss_q1 = ((q1 - backup)**2).mean()  # Mean squared error loss for Q1 network.  # Mittlerer quadratischer Fehlerverlust für das Q1-Netzwerk.
    loss_q2 = ((q2 - backup)**2).mean()  # Mean squared error loss for Q2 network.  # Mittlerer quadratischer Fehlerverlust für das Q2-Netzwerk.
    loss_q = (loss_q1 + loss_q2) / 2  # Average the Q losses for consistency with REDQ.  # Durchschnitt der Q-Verluste für Konsistenz mit REDQ.

    self.q_optimizer.zero_grad()  # Zero out the gradients for the Q-optimizer.  # Setze die Gradienten des Q-Optimierers zurück.
    loss_q.backward()  # Compute the gradients for the Q-losses.  # Berechne die Gradienten für den Q-Verlust.
    self.q_optimizer.step()  # Perform the optimization step for Q networks.  # Führe den Optimierungsschritt für die Q-Netzwerke durch.

    self.model.q1.requires_grad_(False)  # Freeze gradients for Q1 network to prevent computation during policy optimization.  # Friere die Gradienten des Q1-Netzwerks ein, um Berechnungen während der Policy-Optimierung zu verhindern.
    self.model.q2.requires_grad_(False)  # Freeze gradients for Q2 network.  # Friere die Gradienten des Q2-Netzwerks ein.

    q1_pi = self.model.q1(o, pi)  # Compute Q1 value using the policy pi.  # Berechne den Q1-Wert unter Verwendung der Policy pi.
    q2_pi = self.model.q2(o, pi)  # Compute Q2 value using the policy pi.  # Berechne den Q2-Wert unter Verwendung der Policy pi.
    q_pi = torch.min(q1_pi, q2_pi)  # Take the minimum of the two Q-values.  # Nimm das Minimum der beiden Q-Werte.

    loss_pi = (alpha_t * logp_pi - q_pi).mean()  # Compute the loss for the policy using entropy regularization.  # Berechne den Verlust für die Policy unter Verwendung der Entropie-Regularisierung.

    self.pi_optimizer.zero_grad()  # Zero out the gradients for the policy optimizer.  # Setze die Gradienten des Policy-Optimierers zurück.
    loss_pi.backward()  # Compute the gradients for the policy loss.  # Berechne die Gradienten für den Policy-Verlust.
    self.pi_optimizer.step()  # Perform the optimization step for the policy.  # Führe den Optimierungsschritt für die Policy durch.

    self.model.q1.requires_grad_(True)  # Unfreeze gradients for Q1 network.  # Hebe das Einfrieren der Gradienten des Q1-Netzwerks auf.
    self.model.q2.requires_grad_(True)  # Unfreeze gradients for Q2 network.  # Hebe das Einfrieren der Gradienten des Q2-Netzwerks auf.

    with torch.no_grad():  # Disable gradient computation for the following target network updates.  # Deaktiviere die Gradientenberechnung für die folgenden Zielnetzwerk-Aktualisierungen.
        for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):  # Iterate over the model and target model parameters.  # Iteriere über die Parameter des Modells und des Zielmodells.
            p_targ.data.mul_(self.polyak)  # Polyak averaging update: multiply target parameters by polyak factor.  # Polyak-Averaging-Aktualisierung: Multipliziere die Zielparameter mit dem Polyak-Faktor.
            p_targ.data.add_((1 - self.polyak) * p.data)  # Add the weighted difference between model parameters and target parameters.  # Addiere den gewichteten Unterschied zwischen Modellparametern und Zielparametern.



# FIXME: remove debug info  # Markiert, dass Debug-Informationen entfernt werden müssen.
with torch.no_grad():  # Deaktiviert die Gradientenberechnung für diesen Block, um den Speicherverbrauch zu reduzieren und die Berechnungen zu beschleunigen.  # Deaktiviert die Berechnung von Gradienten, was Speicher spart und die Berechnung beschleunigt.
    
    if not cfg.DEBUG_MODE:  # Prüft, ob der Debug-Modus deaktiviert ist.  # Überprüft, ob der Debug-Modus ausgeschaltet ist.
        ret_dict = dict(  # Erstellt ein Dictionary, das Ergebnisse speichert.  # Erstellt ein Dictionary zur Speicherung der Ergebnisse.
            loss_actor=loss_pi.detach().item(),  # Speichert den Verlust des Akteurs.  # Speichert den Verlust des Akteurs.
            loss_critic=loss_q.detach().item(),  # Speichert den Verlust des Kritikers.  # Speichert den Verlust des Kritikers.
        )
    else:  # Wenn Debug-Modus aktiviert ist, werden weitere Details gesammelt.  # Wenn der Debug-Modus aktiviert ist, werden zusätzliche Debug-Informationen erfasst.
        q1_o2_a2 = self.model.q1(o2, a2)  # Berechnet den Wert von q1 mit den Eingaben o2 und a2.  # Berechnet den Wert von q1 für die Eingaben o2 und a2.
        q2_o2_a2 = self.model.q2(o2, a2)  # Berechnet den Wert von q2 mit den Eingaben o2 und a2.  # Berechnet den Wert von q2 für die Eingaben o2 und a2.
        q1_targ_pi = self.model_target.q1(o, pi)  # Berechnet den Zielwert von q1 für die Eingaben o und pi.  # Berechnet den Zielwert von q1 für die Eingaben o und pi.
        q2_targ_pi = self.model_target.q2(o, pi)  # Berechnet den Zielwert von q2 für die Eingaben o und pi.  # Berechnet den Zielwert von q2 für die Eingaben o und pi.
        q1_targ_a = self.model_target.q1(o, a)  # Berechnet den Zielwert von q1 für die Eingaben o und a.  # Berechnet den Zielwert von q1 für die Eingaben o und a.
        q2_targ_a = self.model_target.q2(o, a)  # Berechnet den Zielwert von q2 für die Eingaben o und a.  # Berechnet den Zielwert von q2 für die Eingaben o und a.

        diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()  # Berechnet die Differenz zwischen q1_pi_targ und q_pi_targ und trennt den Gradienten.  # Berechnet die Differenz zwischen q1_pi_targ und q_pi_targ und trennt den Gradienten.
        diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()  # Berechnet die Differenz zwischen q2_pi_targ und q_pi_targ und trennt den Gradienten.  # Berechnet die Differenz zwischen q2_pi_targ und q_pi_targ und trennt den Gradienten.
        diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()  # Berechnet die Differenz zwischen q1_o2_a2 und q1_pi_targ und trennt den Gradienten.  # Berechnet die Differenz zwischen q1_o2_a2 und q1_pi_targ und trennt den Gradienten.
        diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()  # Berechnet die Differenz zwischen q2_o2_a2 und q2_pi_targ und trennt den Gradienten.  # Berechnet die Differenz zwischen q2_o2_a2 und q2_pi_targ und trennt den Gradienten.
        diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()  # Berechnet die Differenz zwischen q1_pi und q1_targ_pi und trennt den Gradienten.  # Berechnet die Differenz zwischen q1_pi und q1_targ_pi und trennt den Gradienten.
        diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()  # Berechnet die Differenz zwischen q2_pi und q2_targ_pi und trennt den Gradienten.  # Berechnet die Differenz zwischen q2_pi und q2_targ_pi und trennt den Gradienten.
        diff_q1_q1t_a = (q1 - q1_targ_a).detach()  # Berechnet die Differenz zwischen q1 und q1_targ_a und trennt den Gradienten.  # Berechnet die Differenz zwischen q1 und q1_targ_a und trennt den Gradienten.
        diff_q2_q2t_a = (q2 - q2_targ_a).detach()  # Berechnet die Differenz zwischen q2 und q2_targ_a und trennt den Gradienten.  # Berechnet die Differenz zwischen q2 und q2_targ_a und trennt den Gradienten.
        diff_q1_backup = (q1 - backup).detach()  # Berechnet die Differenz zwischen q1 und dem Backup-Wert und trennt den Gradienten.  # Berechnet die Differenz zwischen q1 und dem Backup-Wert und trennt den Gradienten.
        diff_q2_backup = (q2 - backup).detach()  # Berechnet die Differenz zwischen q2 und dem Backup-Wert und trennt den Gradienten.  # Berechnet die Differenz zwischen q2 und dem Backup-Wert und trennt den Gradienten.
        diff_q1_backup_r = (q1 - backup + r).detach()  # Berechnet die Differenz zwischen q1, dem Backup-Wert und r, und trennt den Gradienten.  # Berechnet die Differenz zwischen q1, dem Backup-Wert und r und trennt den Gradienten.
        diff_q2_backup_r = (q2 - backup + r).detach()  # Berechnet die Differenz zwischen q2, dem Backup-Wert und r, und trennt den Gradienten.  # Berechnet die Differenz zwischen q2, dem Backup-Wert und r und trennt den Gradienten.

        ret_dict = dict(  # Erstellt ein Dictionary mit den verschiedenen Debug-Informationen und Verlusten.  # Erstellt ein Dictionary mit verschiedenen Debug-Informationen und Verlusten.
            loss_actor=loss_pi.detach().item(),  # Verlust des Akteurs.  # Verlust des Akteurs.
            loss_critic=loss_q.detach().item(),  # Verlust des Kritikers.  # Verlust des Kritikers.
            # debug:
            debug_log_pi=logp_pi.detach().mean().item(),  # Durchschnittlicher Wert von logp_pi.  # Durchschnittlicher Wert von logp_pi.
            debug_log_pi_std=logp_pi.detach().std().item(),  # Standardabweichung von logp_pi.  # Standardabweichung von logp_pi.
            debug_logp_a2=logp_a2.detach().mean().item(),  # Durchschnittlicher Wert von logp_a2.  # Durchschnittlicher Wert von logp_a2.
            debug_logp_a2_std=logp_a2.detach().std().item(),  # Standardabweichung von logp_a2.  # Standardabweichung von logp_a2.
            debug_q_a1=q_pi.detach().mean().item(),  # Durchschnittlicher Wert von q_pi.  # Durchschnittlicher Wert von q_pi.
            debug_q_a1_std=q_pi.detach().std().item(),  # Standardabweichung von q_pi.  # Standardabweichung von q_pi.
            debug_q_a1_targ=q_pi_targ.detach().mean().item(),  # Durchschnittlicher Wert von q_pi_targ.  # Durchschnittlicher Wert von q_pi_targ.
            debug_q_a1_targ_std=q_pi_targ.detach().std().item(),  # Standardabweichung von q_pi_targ.  # Standardabweichung von q_pi_targ.
            debug_backup=backup.detach().mean().item(),  # Durchschnittlicher Wert des Backup-Werts.  # Durchschnittlicher Wert des Backup-Werts.
            debug_backup_std=backup.detach().std().item(),  # Standardabweichung des Backup-Werts.  # Standardabweichung des Backup-Werts.
            debug_q1=q1.detach().mean().item(),  # Durchschnittlicher Wert von q1.  # Durchschnittlicher Wert von q1.
            debug_q1_std=q1.detach().std().item(),  # Standardabweichung von q1.  # Standardabweichung von q1.
            debug_q2=q2.detach().mean().item(),  # Durchschnittlicher Wert von q2.  # Durchschnittlicher Wert von q2.
            debug_q2_std=q2.detach().std().item(),  # Standardabweichung von q2.  # Standardabweichung von q2.
            debug_diff_q1=diff_q1_backup.mean().item(),  # Durchschnittlicher Wert von diff_q1_backup.  # Durchschnittlicher Wert von diff_q1_backup.
            debug_diff_q1_std=diff_q1_backup.std().item(),  # Standardabweichung von diff_q1_backup.  # Standardabweichung von diff_q1_backup.
            debug_diff_q2=diff_q2_backup.mean().item(),  # Durchschnittlicher Wert von diff_q2_backup.  # Durchschnittlicher Wert von diff_q2_backup.
            debug_diff_q2_std=diff_q2_backup.std().item(),  # Standardabweichung von diff_q2_backup.  # Standardabweichung von diff_q2_backup.
            debug_diff_r_q1=diff_q1_backup_r.mean().item(),  # Durchschnittlicher Wert von diff_q1_backup_r.  # Durchschnittlicher Wert von diff_q1_backup_r.
            debug_diff_r_q1_std=diff_q1_backup_r.std().item(),  # Standardabweichung von diff_q1_backup_r.  # Standardabweichung von diff_q1_backup_r.
            debug_diff_r_q2=diff_q2_backup_r.mean().item(),  # Durchschnittlicher Wert von diff_q2_backup_r.  # Durchschnittlicher Wert von diff_q2_backup_r.
            debug_diff_r_q2_std=diff_q2_backup_r.std().item(),  # Standardabweichung von diff_q2_backup_r.  # Standardabweichung von diff_q2_backup_r.
            debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),  # Durchschnittlicher Wert von diff_q1pt_qpt.  # Durchschnittlicher Wert von diff_q1pt_qpt.
            debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),  # Durchschnittlicher Wert von diff_q2pt_qpt.  # Durchschnittlicher Wert von diff_q2pt_qpt.
            debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),  # Durchschnittlicher Wert von diff_q1_q1t_a2.  # Durchschnittlicher Wert von diff_q1_q1t_a2.
            debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),  # Durchschnittlicher Wert von diff_q2_q2t_a2.  # Durchschnittlicher Wert von diff_q2_q2t_a2.
            debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),  # Durchschnittlicher Wert von diff_q1_q1t_pi.  # Durchschnittlicher Wert von diff_q1_q1t_pi.
            debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),  # Durchschnittlicher Wert von diff_q2_q2t_pi.  # Durchschnittlicher Wert von diff_q2_q2t_pi.
            debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),  # Durchschnittlicher Wert von diff_q1_q1t_a.  # Durchschnittlicher Wert von diff_q1_q1t_a.
            debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),  # Durchschnittlicher Wert von diff_q2_q2t_a.  # Durchschnittlicher Wert von diff_q2_q2t_a.
            debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),  # Standardabweichung von diff_q1pt_qpt.  # Standardabweichung von diff_q1pt_qpt.
            debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),  # Standardabweichung von diff_q2pt_qpt.  # Standardabweichung von diff_q2pt_qpt.
            debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),  # Standardabweichung von diff_q1_q1t_a2.  # Standardabweichung von diff_q1_q1t_a2.
            debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),  # Standardabweichung von diff_q2_q2t_a2.  # Standardabweichung von diff_q2_q2t_a2.
            debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),  # Standardabweichung von diff_q1_q1t_pi.  # Standardabweichung von diff_q1_q1t_pi.
            debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),  # Standardabweichung von diff_q2_q2t_pi.  # Standardabweichung von diff_q2_q2t_pi.
            debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),  # Standardabweichung von diff_q1_q1t_a.  # Standardabweichung von diff_q1_q1t_a.
            debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),  # Standardabweichung von diff_q2_q2t_a.  # Standardabweichung von diff_q2_q2t_a.
            debug_r=r.detach().mean().item(),  # Durchschnittlicher Wert von r.  # Durchschnittlicher Wert von r.
            debug_r_std=r.detach().std().item(),  # Standardabweichung von r.  # Standardabweichung von r.
            debug_d=d.detach().mean().item(),  # Durchschnittlicher Wert von d.  # Durchschnittlicher Wert von d.
            debug_d_std=d.detach().std().item(),  # Standardabweichung von d.  # Standardabweichung von d.
            debug_a_0=a[:, 0].detach().mean().item(),  # Durchschnittlicher Wert von a[:, 0].  # Durchschnittlicher Wert von a[:, 0].
            debug_a_0_std=a[:, 0].detach().std().item(),  # Standardabweichung von a[:, 0].  # Standardabweichung von a[:, 0].
            debug_a_1=a[:, 1].detach().mean().item(),  # Durchschnittlicher Wert von a[:, 1].  # Durchschnittlicher Wert von a[:, 1].
            debug_a_1_std=a[:, 1].detach().std().item(),  # Standardabweichung von a[:, 1].  # Standardabweichung von a[:, 1].
            debug_a_2=a[:, 2].detach().mean().item(),  # Durchschnittlicher Wert von a[:, 2].  # Durchschnittlicher Wert von a[:, 2].
            debug_a_2_std=a[:, 2].detach().std().item(),  # Standardabweichung von a[:, 2].  # Standardabweichung von a[:, 2].
            debug_a1_0=pi[:, 0].detach().mean().item(),  # Durchschnittlicher Wert von pi[:, 0].  # Durchschnittlicher Wert von pi[:, 0].
            debug_a1_0_std=pi[:, 0].detach().std().item(),  # Standardabweichung von pi[:, 0].  # Standardabweichung von pi[:, 0].
            debug_a1_1=pi[:, 1].detach().mean().item(),  # Durchschnittlicher Wert von pi[:, 1].  # Durchschnittlicher Wert von pi[:, 1].
            debug_a1_1_std=pi[:, 1].detach().std().item(),  # Standardabweichung von pi[:, 1].  # Standardabweichung von pi[:, 1].
            debug_a1_2=pi[:, 2].detach().mean().item(),  # Durchschnittlicher Wert von pi[:, 2].  # Durchschnittlicher Wert von pi[:, 2].
            debug_a1_2_std=pi[:, 2].detach().std().item(),  # Standardabweichung von pi[:, 2].  # Standardabweichung von pi[:, 2].
            debug_a2_0=a2[:, 0].detach().mean().item(),  # Durchschnittlicher Wert von a2[:, 0].  # Durchschnittlicher Wert von a2[:, 0].
            debug_a2_0_std=a2[:, 0].detach().std().item(),  # Standardabweichung von a2[:, 0].  # Standardabweichung von a2[:, 0].
            debug_a2_1=a2[:, 1].detach().mean().item(),  # Durchschnittlicher Wert von a2[:, 1].  # Durchschnittlicher Wert von a2[:, 1].
            debug_a2_1_std=a2[:, 1].detach().std().item(),  # Standardabweichung von a2[:, 1].  # Standardabweichung von a2[:, 1].
            debug_a2_2=a2[:, 2].detach().mean().item(),  # Durchschnittlicher Wert von a2[:, 2].  # Durchschnittlicher Wert von a2[:, 2].
            debug_a2_2_std=a2[:, 2].detach().std().item(),  # Standardabweichung von a2[:, 2].  # Standardabweichung von a2[:, 2].
        )
    
if self.learn_entropy_coef:  # Wenn die Entropie-Koeffizienten gelernt werden, fügen wir sie zum Ergebnis-Dictionary hinzu.  # Wenn der Entropie-Koeffizient gelernt wird, fügen wir ihn dem Ergebnis-Dictionary hinzu.
    ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()  # Speichert den Verlust des Entropie-Koeffizienten.  # Speichert den Verlust des Entropie-Koeffizienten.
    ret_dict["entropy_coef"] = alpha_t.item()  # Speichert den Wert des Entropie-Koeffizienten.  # Speichert den Wert des Entropie-Koeffizienten.

return ret_dict  # Gibt das Ergebnis-Dictionary zurück.  # Gibt das Ergebnis-Dictionary zurück.


@dataclass(eq=0)  # This is a decorator that marks the class as a dataclass, which automatically generates special methods like __init__, __repr__, etc. The eq=0 part disables automatic equality comparison.  # Deutsch: Dies ist ein Dekorator, der die Klasse als Dataklasse markiert, die automatisch spezielle Methoden wie __init__, __repr__ usw. generiert. eq=0 deaktiviert den automatischen Vergleich der Gleichheit.

class REDQSACAgent(TrainingAgent):  # Defining a class REDQSACAgent that inherits from TrainingAgent.  # Deutsch: Definiert eine Klasse REDQSACAgent, die von TrainingAgent erbt.

    observation_space: type  # The type of the observation space (the environment's input space).  # Deutsch: Der Typ des Beobachtungsraums (der Eingaberaum der Umgebung).
    action_space: type  # The type of the action space (the environment's action space).  # Deutsch: Der Typ des Aktionsraums (der Aktionsraum der Umgebung).
    device: str = None  # device where the model will live (None for auto)  # Deutsch: Gerät, auf dem das Modell ausgeführt wird (None bedeutet automatisch).
    model_cls: type = core.REDQMLPActorCritic  # The class type of the model to be used.  # Deutsch: Der Klassentyp des zu verwendenden Modells.
    gamma: float = 0.99  # Discount factor for future rewards in reinforcement learning.  # Deutsch: Diskontfaktor für zukünftige Belohnungen im Reinforcement Learning.
    polyak: float = 0.995  # Polyak averaging factor for target network update.  # Deutsch: Polyak-Durchschnittsfaktor für die Aktualisierung des Zielnetzwerks.
    alpha: float = 0.2  # Entropy coefficient for exploration in the agent's policy.  # Deutsch: Entropiekoeffizient für die Exploration in der Politik des Agenten.
    lr_actor: float = 1e-3  # Learning rate for the actor network.  # Deutsch: Lernrate für das Actor-Netzwerk.
    lr_critic: float = 1e-3  # Learning rate for the critic network.  # Deutsch: Lernrate für das Kritiker-Netzwerk.
    lr_entropy: float = 1e-3  # Learning rate for entropy coefficient tuning.  # Deutsch: Lernrate für die Feinabstimmung des Entropiekoeffizienten.
    learn_entropy_coef: bool = True  # Whether to learn the entropy coefficient during training.  # Deutsch: Ob der Entropiekoeffizient während des Trainings gelernt werden soll.
    target_entropy: float = None  # Target entropy value, if None, it will be set automatically.  # Deutsch: Zielwert für Entropie, wenn None, wird er automatisch gesetzt.
    n: int = 10  # Number of REDQ parallel Q networks.  # Deutsch: Anzahl der parallelen Q-Netzwerke in REDQ.
    m: int = 2  # Number of randomly sampled target networks in REDQ.  # Deutsch: Anzahl der zufällig ausgewählten Zielnetzwerke in REDQ.
    q_updates_per_policy_update: int = 1  # Number of Q updates per policy update.  # Deutsch: Anzahl der Q-Aktualisierungen pro Politikaktualisierung.

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))  # Defines a cached property that creates a model with no gradient tracking.  # Deutsch: Definiert eine zwischengespeicherte Eigenschaft, die ein Modell ohne Gradientenverfolgung erstellt.

    def __post_init__(self):  # Initialization function called after the dataclass is created.  # Deutsch: Initialisierungsfunktion, die nach der Erstellung der Dataklasse aufgerufen wird.
        observation_space, action_space = self.observation_space, self.action_space  # Assign the observation and action space to variables.  # Deutsch: Weist den Beobachtungs- und Aktionsraum Variablen zu.
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")  # Selects the device (GPU or CPU).  # Deutsch: Wählt das Gerät (GPU oder CPU) aus.
        model = self.model_cls(observation_space, action_space)  # Creates an instance of the model class.  # Deutsch: Erstellt eine Instanz der Modellklasse.
        logging.debug(f" device REDQ-SAC: {device}")  # Logs the selected device for debugging purposes.  # Deutsch: Protokolliert das ausgewählte Gerät zu Debugging-Zwecken.
        self.model = model.to(device)  # Transfers the model to the selected device.  # Deutsch: Überträgt das Modell auf das ausgewählte Gerät.
        self.model_target = no_grad(deepcopy(self.model))  # Creates a target model for the Q-value prediction with no gradient tracking.  # Deutsch: Erstellt ein Zielmodell für die Q-Wertvorhersage ohne Gradientenverfolgung.
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)  # Initializes the optimizer for the actor.  # Deutsch: Initialisiert den Optimierer für den Actor.
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]  # Initializes optimizers for each Q-network.  # Deutsch: Initialisiert Optimierer für jedes Q-Netzwerk.
        self.criterion = torch.nn.MSELoss()  # Defines the loss function (Mean Squared Error).  # Deutsch: Definiert die Verlustfunktion (Mittlere quadratische Abweichung).
        self.loss_pi = torch.zeros((1,), device=device)  # Initializes the loss for the actor to zero.  # Deutsch: Initialisiert den Verlust für den Actor mit Null.

        self.i_update = 0  # Initializes update counter for UTD ratio.  # Deutsch: Initialisiert den Aktualisierungszähler für das UTD-Verhältnis.

        if self.target_entropy is None:  # Checks if target_entropy is not set, then sets it automatically.  # Deutsch: Überprüft, ob target_entropy nicht gesetzt ist, und setzt es automatisch.
            self.target_entropy = -np.prod(action_space.shape)  # Sets the target entropy based on the action space shape.  # Deutsch: Setzt die Zielentropie basierend auf der Form des Aktionsraums.
        else:
            self.target_entropy = float(self.target_entropy)  # Converts target_entropy to float if it is set manually.  # Deutsch: Wandelt target_entropy in einen Float um, wenn es manuell gesetzt ist.

        if self.learn_entropy_coef:  # Checks if entropy coefficient learning is enabled.  # Deutsch: Überprüft, ob das Lernen des Entropiekoeffizienten aktiviert ist.
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)  # Initializes the entropy coefficient log_alpha with gradient tracking.  # Deutsch: Initialisiert den Entropiekoeffizienten log_alpha mit Gradientenverfolgung.
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)  # Initializes optimizer for learning the entropy coefficient.  # Deutsch: Initialisiert den Optimierer zum Lernen des Entropiekoeffizienten.
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)  # Sets alpha_t to a fixed value if entropy coefficient learning is disabled.  # Deutsch: Setzt alpha_t auf einen festen Wert, wenn das Lernen des Entropiekoeffizienten deaktiviert ist.

    def get_actor(self):  # Returns the actor network of the model.  # Deutsch: Gibt das Actor-Netzwerk des Modells zurück.
        return self.model_nograd.actor  # Deutsch: Gibt das Actor-Netzwerk des Modells ohne Gradientenverfolgung zurück.

    def train(self, batch):  # Training function that updates the agent's model.  # Deutsch: Trainingsfunktion, die das Modell des Agenten aktualisiert.
        self.i_update += 1  # Increments the update counter.  # Deutsch: Erhöht den Aktualisierungszähler.
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)  # Determines if policy update is due.  # Deutsch: Bestimmt, ob eine Politikaktualisierung fällig ist.

        o, a, r, o2, d, _ = batch  # Unpacks the training batch into variables: observation, action, reward, next observation, done, etc.  # Deutsch: Entpackt den Trainingsbatch in Variablen: Beobachtung, Aktion, Belohnung, nächste Beobachtung, abgeschlossen, usw.

        if update_policy:  # If a policy update is needed, calculate the policy's output.  # Deutsch: Wenn eine Politikaktualisierung erforderlich ist, berechne die Ausgabe der Politik.
            pi, logp_pi = self.model.actor(o)  # Get the policy and its log probability for the current observation.  # Deutsch: Hole die Politik und ihre Log-Wahrscheinlichkeit für die aktuelle Beobachtung.

        loss_alpha = None  # Initialize the loss for alpha if necessary.  # Deutsch: Initialisiere den Verlust für alpha, falls erforderlich.
        if self.learn_entropy_coef and update_policy:  # Check if entropy coefficient is to be learned and policy update is needed.  # Deutsch: Überprüft, ob der Entropiekoeffizient gelernt werden soll und eine Politikaktualisierung erforderlich ist.
            alpha_t = torch.exp(self.log_alpha.detach())  # Get the entropy coefficient alpha by exponentiating log_alpha.  # Deutsch: Hole den Entropiekoeffizienten alpha, indem log_alpha exponentiert wird.
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()  # Calculate the entropy coefficient loss.  # Deutsch: Berechne den Verlust des Entropiekoeffizienten.

        else:
            alpha_t = self.alpha_t  # Use the fixed entropy coefficient if learning is disabled.  # Deutsch: Verwende den festen Entropiekoeffizienten, wenn das Lernen deaktiviert ist.

        if loss_alpha is not None:  # If there is a loss for alpha, perform an optimization step.  # Deutsch: Wenn ein Verlust für alpha vorhanden ist, führe einen Optimierungsschritt aus.
            self.alpha_optimizer.zero_grad()  # Clear gradients for the alpha optimizer.  # Deutsch: Lösche die Gradienten für den Alpha-Optimierer.
            loss_alpha.backward()  # Backpropagate the loss for alpha.  # Deutsch: Führe den Backpropagation des Verlusts für alpha durch.
            self.alpha_optimizer.step()  # Update the alpha coefficient using the optimizer.  # Deutsch: Aktualisiere den Entropiekoeffizienten mit dem Optimierer.

        with torch.no_grad():  # Turn off gradient tracking for the following operations.  # Deutsch: Deaktiviere die Gradientenverfolgung für die folgenden Operationen.
            a2, logp_a2 = self.model.actor(o2)  # Get the next action and log probability for the next state.  # Deutsch: Hole die nächste Aktion und Log-Wahrscheinlichkeit für den nächsten Zustand.

            sample_idxs = np.random.choice(self.n, self.m, replace=False)  # Randomly sample target networks for the next Q prediction.  # Deutsch: Wähle zufällig Zielnetzwerke für die nächste Q-Vorhersage aus.

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]  # Calculate Q-value predictions for the next state-action pairs.  # Deutsch: Berechne Q-Wert-Vorhersagen für die nächsten Zustands-Aktions-Paare.

            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)  # Stack the Q-value predictions into a tensor.  # Deutsch: Staple die Q-Wert-Vorhersagen in einem Tensor.
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)  # Take the minimum Q-value from the predictions.  # Deutsch: Nimm den minimalen Q-Wert aus den Vorhersagen.
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))  # Calculate the backup target for Q learning.  # Deutsch: Berechne das Backup-Ziel für das Q-Lernen.

        q_prediction_list = [q(o, a) for q in self.model.qs]  # Get Q-values for the current state-action pairs.  # Deutsch: Hole die Q-Werte für die aktuellen Zustand-Aktions-Paare.

        q_prediction_cat = torch.stack(q_prediction_list, -1)  # Stack the Q-values into a tensor.  # Deutsch: Staple die Q-Werte in einem Tensor.

        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup  # Expand the backup tensor to match the number of Q-networks.  # Deutsch: Erweitere den Backup-Tensor, um mit der Anzahl der Q-Netzwerke übereinzustimmen.

        loss_q = self.criterion(q_prediction_cat, backup)  # Calculate the critic's loss using MSE between Q predictions and backup targets.  # Deutsch: Berechne den Verlust des Kritikers mit MSE zwischen Q-Vorhersagen und Backup-Zielen.

        for q in self.q_optimizer_list:  # Iterate over each Q-network optimizer.  # Deutsch: Iteriere über jeden Q-Netzwerk-Optimierer.
            q.zero_grad()  # Zero the gradients for each Q optimizer.  # Deutsch: Setze die Gradienten für jeden Q-Optimierer auf Null.
        loss_q.backward()  # Backpropagate the critic's loss.  # Deutsch: Führe den Backpropagation des Verlusts des Kritikers durch.

        if update_policy:  # If it's time to update the policy, calculate the actor's loss.  # Deutsch: Wenn es Zeit ist, die Politik zu aktualisieren, berechne den Verlust des Actors.
            for q in self.model.qs:  # Turn off gradients for each Q-network.  # Deutsch: Schalte die Gradienten für jedes Q-Netzwerk aus.
                q.requires_grad_(False)  # Deutsch: Schaltet Gradienten für jedes Q-Netzwerk aus.

            qs_pi = [q(o, pi) for q in self.model.qs]  # Get Q-values for the current policy.  # Deutsch: Hole Q-Werte für die aktuelle Politik.

            qs_pi_cat = torch.stack(qs_pi, -1)  # Stack the Q-values for the policy into a tensor.  # Deutsch: Staple die Q-Werte für die Politik in einem Tensor.

            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)  # Calculate the average Q-value across all Q-networks.  # Deutsch: Berechne den durchschnittlichen Q-Wert über alle Q-Netzwerke.
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()  # Calculate the policy's loss.  # Deutsch: Berechne den Verlust der Politik.

            self.pi_optimizer.zero_grad()  # Zero the gradients for the actor optimizer.  # Deutsch: Setze die Gradienten für den Actor-Optimierer auf Null.
            loss_pi.backward()  # Backpropagate the actor's loss.  # Deutsch: Führe den Backpropagation des Verlusts des Actors durch.

            for q in self.model.qs:  # Re-enable gradients for each Q-network.  # Deutsch: Aktiviere Gradienten für jedes Q-Netzwerk.
                q.requires_grad_(True)  # Deutsch: Aktiviere Gradienten für jedes Q-Netzwerk.

        for q_optimizer in self.q_optimizer_list:  # Step through each Q optimizer.  # Deutsch: Führe für jeden Q-Optimierer einen Schritt aus.
            q_optimizer.step()  # Deutsch: Führt den Optimierungsschritt aus.

        if update_policy:  # If policy was updated, step through the policy optimizer.  # Deutsch: Wenn die Politik aktualisiert wurde, führe den Schritt des Politik-Optimierers aus.
            self.pi_optimizer.step()  # Deutsch: Führe den Schritt des Politik-Optimierers aus.

        with torch.no_grad():  # No gradient tracking for parameter updates.  # Deutsch: Keine Gradientenverfolgung für die Parameteraktualisierungen.
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):  # Loop through the parameters of the model and target.  # Deutsch: Iteriere durch die Parameter des Modells und des Ziels.
                p_targ.data.mul_(self.polyak)  # Apply Polyak averaging to the target parameters.  # Deutsch: Wende Polyak-Averaging auf die Ziel-Parameter an.
                p_targ.data.add_((1 - self.polyak) * p.data)  # Update the target parameters with Polyak averaging.  # Deutsch: Aktualisiere die Ziel-Parameter mit Polyak-Averaging.

        if update_policy:  # If policy was updated, store the loss_pi for future reference.  # Deutsch: Wenn die Politik aktualisiert wurde, speichere den Verlust für zukünftige Referenz.
            self.loss_pi = loss_pi.detach()  # Deutsch: Speichert loss_pi für zukünftige Referenz.

        ret_dict = dict(  # Create a dictionary to return losses.  # Deutsch: Erstelle ein Wörterbuch, um die Verluste zurückzugeben.
            loss_actor=self.loss_pi.detach().item(),  # Return the actor's loss.  # Deutsch: Gibt den Verlust des Actors zurück.
            loss_critic=loss_q.detach().item(),  # Return the critic's loss.  # Deutsch: Gibt den Verlust des Kritikers zurück.
        )

        if self.learn_entropy_coef:  # If learning entropy coefficient, include it in the return dictionary.  # Deutsch: Wenn der Entropiekoeffizient gelernt wird, füge ihn zum Rückgabewörterbuch hinzu.
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()  # Return the entropy coefficient loss.  # Deutsch: Gibt den Verlust des Entropiekoeffizienten zurück.
            ret_dict["entropy_coef"] = alpha_t.item()  # Return the current entropy coefficient.  # Deutsch: Gibt den aktuellen Entropiekoeffizienten zurück.

        return ret_dict  # Return the dictionary with losses.  # Deutsch: Gibt das Wörterbuch mit den Verlusten zurück.

