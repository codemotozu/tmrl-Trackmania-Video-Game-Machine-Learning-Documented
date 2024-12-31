# standard library imports
import time  # Provides time-related functions.  # Zeitfunktionen bereitstellen.
from dataclasses import dataclass  # Allows defining classes with minimal boilerplate.  # Ermöglicht das Definieren von Klassen mit minimalem Aufwand.

# third-party imports
import torch  # PyTorch library for machine learning.  # PyTorch-Bibliothek für maschinelles Lernen.
from pandas import DataFrame  # Data structure for handling tabular data.  # Datenstruktur zur Verarbeitung tabellarischer Daten.

# local imports
from tmrl.util import pandas_dict  # Custom utility function for handling pandas objects.  # Benutzerdefinierte Hilfsfunktion zur Verarbeitung von Pandas-Objekten.

import logging  # Provides logging utilities.  # Stellt Logging-Dienste bereit.

__docformat__ = "google"  # Specifies Google-style docstrings for documentation.  # Gibt Google-Style-Dokumentation an.

@dataclass(eq=0)  # Defines a class with auto-generated methods (e.g., __init__). Disables equality comparison.  # Definiert eine Klasse mit automatisch generierten Methoden (z. B. __init__). Deaktiviert Gleichheitsvergleiche.
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.  # Trainings-Wrapper für Off-Policy-Algorithmen.
    """
    env_cls: type = None  # = GenericGymEnv. Dummy environment for observation and action spaces.  # Dummyelement für Beobachtungs- und Aktionsräume.
    memory_cls: type = None  # = TorchMemory. Class for replay memory.  # Klasse für Replay-Speicher.
    training_agent_cls: type = None  # = TrainingAgent. Class for the training agent.  # Klasse für den Trainingsagenten.
    epochs: int = 10  # Total number of epochs; saves agent each epoch.  # Gesamtanzahl der Epochen; speichert den Agenten jede Epoche.
    rounds: int = 50  # Number of rounds per epoch. Statistics generated every round.  # Anzahl der Runden pro Epoche. Statistiken werden jede Runde generiert.
    steps: int = 2000  # Training steps per round.  # Trainingsschritte pro Runde.
    update_model_interval: int = 100  # Steps between model updates.  # Schritte zwischen Modellaktualisierungen.
    update_buffer_interval: int = 100  # Steps between buffer retrievals.  # Schritte zwischen Buffer-Abrufen.
    max_training_steps_per_env_step: float = 1.0  # Max training-to-environment step ratio.  # Maximales Verhältnis von Trainings- zu Umweltschritten.
    sleep_between_buffer_retrieval_attempts: float = 1.0  # Sleep duration when waiting for samples.  # Wartezeit beim Warten auf Proben.
    profiling: bool = False  # Enable profiling for run_epoch if True.  # Profiling für run_epoch aktivieren, wenn True.
    agent_scheduler: callable = None  # Optional scheduler function for agent setup.  # Optionale Scheduler-Funktion für die Agenteneinrichtung.
    start_training: int = 0  # Minimum samples in buffer before training starts.  # Minimale Probenanzahl im Speicher vor Trainingsbeginn.
    device: str = None  # Device for memory and agent (e.g., CPU, GPU).  # Gerät für Speicher und Agent (z. B. CPU, GPU).

    total_updates = 0  # Total number of model updates.  # Gesamtanzahl der Modellaktualisierungen.

    def __post_init__(self):  # Post-initialization setup after dataclass creation.  # Nachinitialisierung nach der Erstellung der Datenklasse.
        device = self.device  # Store device information.  # Gerätedaten speichern.
        self.epoch = 0  # Initialize epoch counter.  # Initialisiert den Epochenzähler.
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)  # Create memory instance with steps and device.  # Erstellt eine Speicherinstanz mit Schritten und Gerät.
        if type(self.env_cls) == tuple:  # If env_cls is a tuple, unpack it as observation and action spaces.  # Wenn env_cls ein Tuple ist, wird es als Beobachtungs- und Aktionsräume entpackt.
            observation_space, action_space = self.env_cls
        else:  # Otherwise, create an environment instance to get spaces.  # Andernfalls eine Umgebungsinstanz erstellen, um Räume zu erhalten.
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,  # Create the training agent.  # Den Trainingsagenten erstellen.
                                             action_space=action_space,
                                             device=device)
        self.total_samples = len(self.memory)  # Total samples in memory.  # Gesamtanzahl der Speicherproben.
        logging.info(f" Initial total_samples:{self.total_samples}")  # Log initial sample count.  # Probenanzahl zu Beginn protokollieren.

    def update_buffer(self, interface):  # Update memory with new samples from the interface.  # Aktualisiert den Speicher mit neuen Proben aus der Schnittstelle.
        buffer = interface.retrieve_buffer()  # Retrieve buffer from interface.  # Ruft den Puffer aus der Schnittstelle ab.
        self.memory.append(buffer)  # Add retrieved samples to memory.  # Fügt die abgerufenen Proben dem Speicher hinzu.
        self.total_samples += len(buffer)  # Update total sample count.  # Aktualisiert die Gesamtanzahl der Proben.

    def check_ratio(self, interface):  # Check training-to-sample ratio and pause if necessary.  # Überprüft das Verhältnis von Training zu Proben und pausiert bei Bedarf.
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0  # Calculate ratio.  # Verhältnis berechnen.
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:  # If ratio exceeds limit or no samples, wait.  # Wenn das Verhältnis das Limit überschreitet oder keine Proben vorhanden sind, warten.
            logging.info(f" Waiting for new samples")  # Log waiting status.  # Wartezustand protokollieren.
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:  # Continue waiting if conditions persist.  # Weiter warten, wenn die Bedingungen andauern.
                self.update_buffer(interface)  # Update buffer while waiting.  # Puffer beim Warten aktualisieren.
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0  # Recalculate ratio.  # Verhältnis neu berechnen.
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:  # Sleep if necessary.  # Falls erforderlich, schlafen.
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)  # Sleep duration.  # Wartezeit.
            logging.info(f" Resuming training")  # Log resumption of training.  # Fortsetzung des Trainings protokollieren.

    def run_epoch(self, interface):  # Execute one training epoch.  # Führt eine Trainingsepoche aus.
        stats = []  # Initialize stats list.  # Initialisiert eine Statistikliste.
        state = None  # Initialize state.  # Initialisiert den Zustand.

        if self.agent_scheduler is not None:  # If agent_scheduler is defined, call it.  # Wenn agent_scheduler definiert ist, aufrufen.
            self.agent_scheduler(self.agent, self.epoch)  # Execute agent setup for this epoch.  # Führt die Agenteneinrichtung für diese Epoche aus.

        for rnd in range(self.rounds):  # Iterate over rounds.  # Iteriert über die Runden.
            logging.info(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))  # Log epoch and round progress.  # Protokolliert Fortschritt bei Epochen und Runden.
            logging.debug(f"(Training): current memory size:{len(self.memory)}")  # Log current memory size.  # Protokolliert die aktuelle Speichergröße.

            stats_training = []  # Initialize training stats for this round.  # Initialisiert Trainingsstatistiken für diese Runde.

            t0 = time.time()  # Start timing for ratio check.  # Beginnt das Timing für die Verhältnisüberprüfung.
            self.check_ratio(interface)  # Ensure training ratio is maintained.  # Stellt sicher, dass das Trainingsverhältnis eingehalten wird.
            t1 = time.time()  # End timing for ratio check.  # Beendet das Timing für die Verhältnisüberprüfung.

            if self.profiling:  # If profiling is enabled.  # Wenn Profiling aktiviert ist.
                from pyinstrument import Profiler  # Import Profiler for runtime analysis.  # Importiert Profiler zur Laufzeitanalyse.
                pro = Profiler()  # Initialize profiler.  # Initialisiert den Profiler.
                pro.start()  # Start profiler.  # Startet den Profiler.

            t2 = time.time()  # Capture time before sampling.  # Erfasst die Zeit vor dem Sampling.

            t_sample_prev = t2  # Store previous sampling time.  # Speichert die vorherige Sampling-Zeit.


for batch in self.memory:  # this samples a fixed number of batches  # Dies entnimmt eine feste Anzahl von Batches aus dem Speicher

    t_sample = time.time()  # record the time of sampling  # Zeit für das Abtasten wird aufgezeichnet

    if self.total_updates % self.update_buffer_interval == 0:  # checks if it's time to update the buffer  # Überprüft, ob es Zeit ist, den Puffer zu aktualisieren
        self.update_buffer(interface)  # retrieve local buffer in replay memory  # Ruft den lokalen Puffer im Replay-Speicher ab

    t_update_buffer = time.time()  # record the time of buffer update  # Zeit für die Pufferaktualisierung wird aufgezeichnet

    if self.total_updates == 0:  # checks if it's the first update  # Überprüft, ob es das erste Update ist
        logging.info(f"starting training")  # log the start of training  # Protokolliert den Beginn des Trainings

    stats_training_dict = self.agent.train(batch)  # train the agent with the current batch and store statistics  # Trainiert den Agenten mit dem aktuellen Batch und speichert Statistiken

    t_train = time.time()  # record the time of training  # Zeit für das Training wird aufgezeichnet

    stats_training_dict["return_test"] = self.memory.stat_test_return  # add test return stats to training dictionary  # Fügt Test-Rückgabewerte zu den Trainingsstatistiken hinzu
    stats_training_dict["return_train"] = self.memory.stat_train_return  # add training return stats to training dictionary  # Fügt Trainings-Rückgabewerte zu den Trainingsstatistiken hinzu
    stats_training_dict["episode_length_test"] = self.memory.stat_test_steps  # add test episode length to training dictionary  # Fügt die Länge der Testepisode zu den Trainingsstatistiken hinzu
    stats_training_dict["episode_length_train"] = self.memory.stat_train_steps  # add training episode length to training dictionary  # Fügt die Länge der Trainingsepisode zu den Trainingsstatistiken hinzu
    stats_training_dict["sampling_duration"] = t_sample - t_sample_prev  # add sampling duration to training dictionary  # Fügt die Abtastdauer zu den Trainingsstatistiken hinzu
    stats_training_dict["training_step_duration"] = t_train - t_update_buffer  # add training step duration to training dictionary  # Fügt die Trainingsschrittdauer zu den Trainingsstatistiken hinzu
    stats_training += stats_training_dict,  # appends the current training statistics  # Hängt die aktuellen Trainingsstatistiken an

    self.total_updates += 1  # increment the total update count  # Erhöht die Gesamtzahl der Updates um 1

    if self.total_updates % self.update_model_interval == 0:  # checks if it's time to update the model  # Überprüft, ob es Zeit ist, das Modell zu aktualisieren
        interface.broadcast_model(self.agent.get_actor())  # broadcast the updated model to the interface  # Sendet das aktualisierte Modell an die Schnittstelle

    self.check_ratio(interface)  # checks the update ratio with the interface  # Überprüft das Aktualisierungsverhältnis mit der Schnittstelle

    t_sample_prev = time.time()  # update the previous sample time  # Aktualisiert die vorherige Abtastzeit

t3 = time.time()  # record the final time for the round  # Aufzeichnen der Endzeit für die Runde

round_time = t3 - t0  # calculate the total round time  # Berechnet die gesamte Rundendauer
idle_time = t1 - t0  # calculate the idle time  # Berechnet die Leerlaufzeit
update_buf_time = t2 - t1  # calculate the buffer update time  # Berechnet die Pufferaktualisierungszeit
train_time = t3 - t2  # calculate the training time  # Berechnet die Trainingszeit
logging.debug(f"round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}")  # logs the time statistics  # Protokolliert die Zeitstatistiken

stats += pandas_dict(memory_len=len(self.memory), round_time=round_time, idle_time=idle_time, **DataFrame(stats_training).mean(skipna=True)),  # appends the round statistics to the stats list  # Hängt die Rundendaten an die Statistikliste an

logging.info(stats[-1].add_prefix("  ").to_string() + '\n')  # logs the last set of statistics  # Protokolliert den letzten Satz von Statistiken

if self.profiling:  # checks if profiling is enabled  # Überprüft, ob das Profiling aktiviert ist
    pro.stop()  # stops the profiling  # Stoppt das Profiling
    logging.info(pro.output_text(unicode=True, color=False, show_all=True))  # logs the profiling output  # Protokolliert die Profiling-Ausgabe

self.epoch += 1  # increment the epoch count  # Erhöht die Epoche um 1
return stats  # returns the collected statistics  # Gibt die gesammelten Statistiken zurück


class TorchTrainingOffline(TrainingOffline):  # Defines a subclass for offline training with PyTorch  # Definiert eine Unterklasse für das Offline-Training mit PyTorch
    """
    TrainingOffline for trainers based on PyTorch.  # TrainingOffline für Trainer, die auf PyTorch basieren.
    This class implements automatic device selection with PyTorch.  # Diese Klasse implementiert eine automatische Geräteeinstellung mit PyTorch.
    """
    def __init__(self,
                 env_cls: type = None,  # Environment class, for observation and action spaces  # Umgebungs-Klasse, für Beobachtungs- und Aktionsräume
                 memory_cls: type = None,  # Class for the replay memory  # Klasse für den Replay-Speicher
                 training_agent_cls: type = None,  # Class for the training agent  # Klasse für den Trainingsagenten
                 epochs: int = 10,  # Total number of epochs  # Gesamtzahl der Epochen
                 rounds: int = 50,  # Number of rounds per epoch  # Anzahl der Runden pro Epoche
                 steps: int = 2000,  # Number of training steps per round  # Anzahl der Trainingsschritte pro Runde
                 update_model_interval: int = 100,  # Number of steps between model broadcasts  # Anzahl der Schritte zwischen Modell-Übertragungen
                 update_buffer_interval: int = 100,  # Number of steps between buffer updates  # Anzahl der Schritte zwischen Pufferaktualisierungen
                 max_training_steps_per_env_step: float = 1.0,  # Max training steps per environment step  # Maximale Trainingsschritte pro Umweltschritt
                 sleep_between_buffer_retrieval_attempts: float = 1.0,  # Time to wait between buffer retrieval attempts  # Wartezeit zwischen Pufferabrufversuchen
                 profiling: bool = False,  # If True, enable profiling  # Wenn True, Profiling aktivieren
                 agent_scheduler: callable = None,  # Optional callable to adjust the agent  # Optionaler Aufruf, um den Agenten anzupassen
                 start_training: int = 0,  # Minimum samples before training starts  # Mindestanzahl an Beispielen, bevor das Training beginnt
                 device: str = None):  # Device to use for training (None for auto selection)  # Gerät, das für das Training verwendet wird (None für automatische Auswahl)
        """
        Same arguments as `TrainingOffline`, but when `device` is `None` it is selected automatically for torch.  # Dieselben Argumente wie `TrainingOffline`, aber wenn `device` auf `None` gesetzt ist, wird es automatisch für Torch ausgewählt.
        """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # selects the device based on availability  # Wählt das Gerät basierend auf der Verfügbarkeit aus
super().__init__(env_cls,  # initializes the parent class with environment class  # Initialisiert die Elternklasse mit der Umgebungs-Klasse
                 memory_cls,  # initializes the parent class with memory class  # Initialisiert die Elternklasse mit der Speicher-Klasse
                 training_agent_cls,  # initializes the parent class with training agent class  # Initialisiert die Elternklasse mit der Trainings-Agenten-Klasse
                 epochs,  # passes the number of epochs to the parent class  # Übergibt die Anzahl der Epochen an die Elternklasse
                 rounds,  # passes the number of rounds to the parent class  # Übergibt die Anzahl der Runden an die Elternklasse
                 steps,  # passes the number of steps to the parent class  # Übergibt die Anzahl der Schritte an die Elternklasse
                 update_model_interval,  # passes the interval for model updates  # Übergibt das Intervall für Modellaktualisierungen
                 update_buffer_interval,  # passes the interval for buffer updates  # Übergibt das Intervall für Pufferaktualisierungen
                 max_training_steps_per_env_step,  # passes the maximum number of training steps per environment step  # Übergibt die maximalen Trainingsschritte pro Umweltschritt
                 sleep_between_buffer_retrieval_attempts,  # passes the sleep time between buffer retrieval attempts  # Übergibt die Schlafenszeit zwischen den Pufferabrufversuchen
                 profiling,  # passes profiling settings for performance analysis  # Übergibt die Profiling-Einstellungen zur Leistungsanalyse
                 agent_scheduler,  # passes the agent scheduler for scheduling the agent's tasks  # Übergibt den Agenten-Scheduler zur Planung der Aufgaben des Agenten
                 start_training,  # indicates whether to start training immediately  # Gibt an, ob das Training sofort beginnen soll
                 device)  # calls the parent class constructor with device  # Ruft den Konstruktor der Elternklasse mit dem Gerät auf
