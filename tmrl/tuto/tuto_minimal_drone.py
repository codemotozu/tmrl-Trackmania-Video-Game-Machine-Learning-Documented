"""
Tutorial: a minimal TMRL pipeline for real-time robots.

This script works out-of-the-box for real-time environments with flat continuous observations and actions.
"""
# tutorial imports:
from threading import Thread  # Importing Thread from the threading library to enable multi-threading.  # Importieren von Thread aus der threading-Bibliothek zur Unterstützung von Multithreading.
from tuto_envs.dummy_rc_drone_interface import DUMMY_RC_DRONE_CONFIG  # Importing a configuration for the dummy RC drone.  # Importieren einer Konfiguration für die Dummy- RC-Drohne.

# TMRL imports:
from tmrl.networking import Server, RolloutWorker, Trainer  # Importing the necessary components for networking in the TMRL pipeline.  # Importieren der benötigten Komponenten für das Networking in der TMRL-Pipeline.
from tmrl.util import partial  # Importing 'partial' to create partially applied functions.  # Importieren von 'partial' zum Erstellen von partiell angewendeten Funktionen.
from tmrl.envs import GenericGymEnv  # Importing a generic environment class from TMRL.  # Importieren einer generischen Umgebungs-Klasse aus TMRL.
import tmrl.config.config_constants as cfg  # Importing the configuration constants under the alias 'cfg'.  # Importieren der Konfigurationskonstanten unter dem Alias 'cfg'.
from tmrl.training_offline import TorchTrainingOffline  # Importing a class for offline training with Torch.  # Importieren einer Klasse für Offline-Training mit Torch.
from tmrl.custom.custom_algorithms import SpinupSacAgent  # Importing a custom SAC agent for training.  # Importieren eines benutzerdefinierten SAC-Agenten für das Training.
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic  # Importing custom models for the agent.  # Importieren von benutzerdefinierten Modellen für den Agenten.
from tmrl.custom.custom_memories import GenericTorchMemory  # Importing a custom memory class for Torch.  # Importieren einer benutzerdefinierten Speicher-Klasse für Torch.

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False  # Setting debug mode flag for the pipeline (False means debugging is off).  # Setzen des Debug-Modus-Flags für die Pipeline (False bedeutet, dass Debugging deaktiviert ist).

# Name used for training checkpoints and models saved in the TmrlData folder.
# If you change anything, also change this name (or delete the saved files in TmrlData).
my_run_name = "tutorial_minimal_drone"  # Defining the name of the run, used for saving checkpoints.  # Definieren des Namens des Laufs, der zum Speichern von Checkpoints verwendet wird.

# First, you need to define your Gymnasium environment.
# TMRL is typically useful to train real-time robots.
# Thus, we use Real-Time Gym to define a dummy RC drone as an example.
# (Implemented in tuto_envs.dummy_rc_drone_interface)

# === Environment ======================================================================================================

# rtgym interface:

my_rtgym_config = DUMMY_RC_DRONE_CONFIG  # Assigning the RC drone configuration to a variable.  # Zuweisen der RC-Drohnen-Konfiguration zu einer Variablen.

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_rtgym_config})  # Creating a partial function to initialize the environment with specific configurations.  # Erstellen einer partiellen Funktion zur Initialisierung der Umgebung mit spezifischen Konfigurationen.

# Observation and action space:

dummy_env = env_cls()  # Instantiating the environment class to create an environment object.  # Instanziieren der Umgebungsklasse, um ein Umweltobjekt zu erstellen.
act_space = dummy_env.action_space  # Extracting the action space from the environment.  # Extrahieren des Aktionsraums aus der Umgebung.
obs_space = dummy_env.observation_space  # Extracting the observation space from the environment.  # Extrahieren des Beobachtungsraums aus der Umgebung.

print(f"action space: {act_space}")  # Printing the action space.  # Drucken des Aktionsraums.
print(f"observation space: {obs_space}")  # Printing the observation space.  # Drucken des Beobachtungsraums.

# Now that we have defined our environment, let us train an agent with the generic TMRL pipeline.
# TMRL pipelines have a central communication Server, a Trainer, and one to several RolloutWorkers.

# === TMRL Server ======================================================================================================

# The TMRL Server is the central point of communication between TMRL entities.
# The Trainer and the RolloutWorkers connect to the Server.

security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.  # Dies ist in lokalen sicheren Netzwerken ausreichend. Für das Internet verwenden Sie stattdessen "TLS".
password = cfg.PASSWORD  # This is the password defined in TmrlData/config/config.json.  # Dies ist das Passwort, das in TmrlData/config/config.json definiert ist.

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.  # Dies ist die IP-Adresse des Hosts. Ändern Sie sie für Ihre öffentliche IP, wenn Sie im Internet arbeiten möchten.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.  # Im Internet muss der Server über diesen Port erreichbar sein.

if __name__ == "__main__":  # If this script is being run directly (not imported).  # Wenn dieses Skript direkt ausgeführt wird (nicht importiert).
    # Instantiating a TMRL Server is straightforward.
    # More arguments are available for, e.g., using TLS. Please refer to the TMRL documentation.
    my_server = Server(security=security, password=password, port=server_port)  # Creating a server instance with the specified parameters.  # Erstellen einer Serverinstanz mit den angegebenen Parametern.

# === TMRL Worker ======================================================================================================

# TMRL RolloutWorkers are responsible for collecting training samples.
# A RolloutWorker contains an ActorModule, which encapsulates its policy.

# ActorModule:

# SquashedGaussianMLPActor processes observations through an MLP.
# It is designed to work with the SAC algorithm.
actor_module_cls = partial(SquashedGaussianMLPActor)  # Using the 'partial' function to create a version of SquashedGaussianMLPActor.  # Verwendung der Funktion 'partial', um eine Version von SquashedGaussianMLPActor zu erstellen.

# Worker local files

weights_folder = cfg.WEIGHTS_FOLDER  # Defining the folder where weights will be saved.  # Definieren des Ordners, in dem die Gewichte gespeichert werden.
model_path = str(weights_folder / (my_run_name + ".tmod"))  # Defining the file path for saving the current model.  # Definieren des Dateipfads zum Speichern des aktuellen Modells.
model_path_history = str(weights_folder / (my_run_name + "_"))  # Defining the file path for saving model history.  # Definieren des Dateipfads zum Speichern der Modellhistorie.
model_history = -1  # Let us not save a model history.  # Wir speichern keine Modellhistorie.



if __name__ == "__main__":  # Checks if this script is being run directly.  # Überprüft, ob dieses Skript direkt ausgeführt wird.
    my_worker = RolloutWorker(  # Creates an instance of the RolloutWorker class.  # Erstellt eine Instanz der RolloutWorker-Klasse.
        env_cls=env_cls,  # The environment class for the worker.  # Die Umgebungsklasse für den Worker.
        actor_module_cls=actor_module_cls,  # The class of the actor module used in the worker.  # Die Klasse des Actor-Moduls, das im Worker verwendet wird.
        sample_compressor=None,  # Optionally, a compressor for samples (not used here).  # Optional ein Kompressor für Proben (wird hier nicht verwendet).
        device="cpu",  # The device to run the worker on (CPU in this case).  # Das Gerät, auf dem der Worker ausgeführt wird (hier CPU).
        server_ip=server_ip,  # IP address of the server.  # IP-Adresse des Servers.
        server_port=server_port,  # Port number of the server.  # Portnummer des Servers.
        password=password,  # Password to connect to the server.  # Passwort zum Verbinden mit dem Server.
        max_samples_per_episode=1000,  # Maximum number of samples per episode.  # Maximale Anzahl von Proben pro Episode.
        model_path=model_path,  # Path to the model.  # Pfad zum Modell.
        model_history=model_history,  # History of the model used for rollback or update.  # Historie des Modells, die für Rollback oder Updates verwendet wird.
        crc_debug=CRC_DEBUG)  # Option to enable CRC debugging.  # Option zur Aktivierung der CRC-Debugging-Option.

    # Note: at this point, the RolloutWorker is not collecting samples yet.  # Hinweis: Zu diesem Zeitpunkt sammelt der RolloutWorker noch keine Proben.
    # Nevertheless, it connects to the Server.  # Dennoch stellt er eine Verbindung zum Server her.

# === TMRL Trainer =====================================================================================================

# The TMRL Trainer is where your training algorithm lives.  # Der TMRL Trainer ist der Ort, an dem dein Trainingsalgorithmus ausgeführt wird.
# It connects to the Server, to retrieve training samples collected from the RolloutWorkers.  # Er verbindet sich mit dem Server, um Trainingsproben von den RolloutWorkern abzurufen.
# Periodically, it also sends updated policies to the Server, which forwards them to the RolloutWorkers.  # Periodisch sendet er auch aktualisierte Richtlinien an den Server, der sie an die RolloutWorker weiterleitet.

# TMRL Trainers contain a Training class. Currently, only TrainingOffline is supported.  # TMRL Trainer enthalten eine Trainingsklasse. Derzeit wird nur TrainingOffline unterstützt.
# TrainingOffline notably contains a Memory class, and a TrainingAgent class.  # TrainingOffline enthält insbesondere eine Memory-Klasse und eine TrainingAgent-Klasse.
# The Memory is a replay buffer. In TMRL, you are able and encouraged to define your own Memory.  # Der Memory ist ein Replay-Buffer. In TMRL kannst du deinen eigenen Memory definieren und dazu ermutigt werden.
# This is how you can implement highly optimized ad-hoc pipelines for your applications.  # So kannst du hochoptimierte ad-hoc Pipelines für deine Anwendungen implementieren.
# Nevertheless, TMRL also defines a generic, non-optimized Memory that can be used for any pipeline.  # Dennoch definiert TMRL auch einen generischen, nicht optimierten Memory, der für jede Pipeline verwendet werden kann.
# The TrainingAgent contains your training algorithm per-se.  # Der TrainingAgent enthält deinen Trainingsalgorithmus selbst.
# TrainingOffline is meant for asynchronous off-policy algorithms, such as Soft Actor-Critic.  # TrainingOffline ist für asynchrone Off-Policy-Algorithmen wie Soft Actor-Critic gedacht.

# Trainer local files:

weights_folder = cfg.WEIGHTS_FOLDER  # Directory where model weights are stored.  # Verzeichnis, in dem die Modellgewichte gespeichert werden.
checkpoints_folder = cfg.CHECKPOINTS_FOLDER  # Directory where checkpoints are stored.  # Verzeichnis, in dem die Checkpoints gespeichert werden.
model_path = str(weights_folder / (my_run_name + "_t.tmod"))  # Path to the model file.  # Pfad zur Modell-Datei.
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))  # Path to the checkpoints file.  # Pfad zur Checkpoints-Datei.

# Dummy environment OR (observation space, action space) tuple:
env_cls = (obs_space, act_space)  # The environment class or a tuple of observation and action space.  # Die Umgebungs-Klasse oder ein Tupel aus Beobachtungs- und Aktionsraum.

# Memory:

memory_cls = partial(GenericTorchMemory,  # Class for the memory used in training.  # Klasse für den im Training verwendeten Memory.
                     memory_size=1e6,  # Size of the memory buffer.  # Größe des Memory-Puffers.
                     batch_size=32,  # Number of samples per batch.  # Anzahl der Proben pro Batch.
                     crc_debug=CRC_DEBUG)  # Option to enable CRC debugging for the memory.  # Option zur Aktivierung des CRC-Debugging für den Memory.

# Training agent:

training_agent_cls = partial(SpinupSacAgent,  # Class for the training agent.  # Klasse für den Trainings-Agent.
                             model_cls=MLPActorCritic,  # Class for the model used by the agent.  # Klasse für das vom Agenten verwendete Modell.
                             gamma=0.99,  # Discount factor for future rewards.  # Abzinsungsfaktor für zukünftige Belohnungen.
                             polyak=0.995,  # Polyak averaging coefficient.  # Polyak-Averaging-Koeffizient.
                             alpha=0.2,  # Entropy regularization coefficient.  # Entropie-Regularisierungskoeffizient.
                             lr_actor=1e-3,  # Learning rate for the actor network.  # Lernrate für das Actor-Netzwerk.
                             lr_critic=1e-3,  # Learning rate for the critic network.  # Lernrate für das Critic-Netzwerk.
                             lr_entropy=1e-3,  # Learning rate for entropy regularization.  # Lernrate für die Entropie-Regularisierung.
                             learn_entropy_coef=True,  # Whether to learn the entropy coefficient.  # Ob der Entropie-Koeffizient gelernt werden soll.
                             target_entropy=None)  # Target entropy value.  # Zielwert der Entropie.

# Training parameters:

epochs = 10  # Maximum number of epochs. Usually set to np.inf for unlimited epochs.  # Maximale Anzahl von Epochen. Wird normalerweise auf np.inf für unbegrenzte Epochen gesetzt.
rounds = 10  # Number of rounds per epoch.  # Anzahl der Runden pro Epoche.
steps = 1000  # Number of steps per round.  # Anzahl der Schritte pro Runde.
update_buffer_interval = 100  # Interval for updating the memory buffer.  # Intervall für das Aktualisieren des Memory-Puffers.
update_model_interval = 100  # Interval for updating the model.  # Intervall für das Aktualisieren des Modells.
max_training_steps_per_env_step = 2.0  # Maximum number of training steps per environment step.  # Maximale Anzahl von Trainingsschritten pro Umweltschritt.
start_training = 400  # Number of steps before starting training.  # Anzahl der Schritte, bevor das Training beginnt.
device = None  # The device on which the training will run.  # Das Gerät, auf dem das Training ausgeführt wird.

# Training class:

training_cls = partial(
    TorchTrainingOffline,  # The class for offline training.  # Die Klasse für das Offline-Training.
    env_cls=env_cls,  # Environment class used in training.  # Die Umgebungs-Klasse, die im Training verwendet wird.
    memory_cls=memory_cls,  # Memory class used for storing experiences.  # Die Memory-Klasse, die zum Speichern von Erfahrungen verwendet wird.
    training_agent_cls=training_agent_cls,  # The training agent class.  # Die Trainings-Agenten-Klasse.
    epochs=epochs,  # Maximum number of epochs for training.  # Maximale Anzahl von Epochen für das Training.
    rounds=rounds,  # Number of rounds per epoch.  # Anzahl der Runden pro Epoche.
    steps=steps,  # Number of steps per round.  # Anzahl der Schritte pro Runde.
    update_buffer_interval=update_buffer_interval,  # Memory buffer update interval.  # Intervall für die Aktualisierung des Memory-Puffers.
    update_model_interval=update_model_interval,  # Model update interval.  # Intervall für die Modellaktualisierung.
    max_training_steps_per_env_step=max_training_steps_per_env_step,  # Maximum training steps per environment step.  # Maximale Trainingsschritte pro Umweltschritt.
    start_training=start_training,  # Number of steps before starting training.  # Anzahl der Schritte, bevor das Training beginnt.
    device=device)  # Device to run the training on.  # Gerät, auf dem das Training ausgeführt wird.

# Trainer instance:

if __name__ == "__main__":  # Checks if the script is being run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    my_trainer = Trainer(  # Creates an instance of the Trainer class.  # Erstellt eine Instanz der Trainer-Klasse.
        training_cls=training_cls,  # The class used for training.  # Die Klasse, die für das Training verwendet wird.
        server_ip=server_ip,  # IP address of the server.  # IP-Adresse des Servers.
        server_port=server_port,  # Port number of the server.  # Portnummer des Servers.
        password=password,  # Password for server authentication.  # Passwort zur Server-Authentifizierung.
        model_path=model_path,  # Path to save the trained model.  # Pfad zum Speichern des trainierten Modells.
        checkpoint_path=checkpoints_path)  # Path to save checkpoints during training.  # Pfad zum Speichern von Checkpoints während des Trainings.

# === Running the pipeline =============================================================================================

# Now we have everything we need.  # Jetzt haben wir alles, was wir brauchen.
# Typically, you will run your TMRL Server, Trainer and RolloutWorkers in different terminals / machines.  # Normalerweise würdest du den TMRL-Server, Trainer und RolloutWorker in verschiedenen Terminals/Maschinen ausführen.
# But for simplicity, in this tutorial, we run them in different threads instead.  # Aber aus Gründen der Einfachheit führen wir sie in diesem Tutorial in verschiedenen Threads aus.
# Note that the Server is already running (it started running when instantiated).  # Beachte, dass der Server bereits läuft (er wurde beim Instanziieren gestartet).

# Separate threads for running the RolloutWorker and Trainer:

def run_worker(worker):  # Function to run the worker.  # Funktion zum Ausführen des Workers.
    worker.run(test_episode_interval=10, verbose=True)  # Starts the worker with specified parameters.  # Startet den Worker mit den angegebenen Parametern.

def run_trainer(trainer):  # Function to run the trainer.  # Funktion zum Ausführen des Trainers.
    trainer.run()  # Starts the trainer.  # Startet den Trainer.

if __name__ == "__main__":  # Checks if the script is being run directly.  # Überprüft, ob das Skript direkt ausgeführt wird.
    daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)  # Creates a daemon thread for the worker.  # Erstellt einen Daemon-Thread für den Worker.
    daemon_thread_worker.start()  # Starts the worker daemon thread.  # Startet den Worker-Daemon-Thread.

    run_trainer(my_trainer)  # Starts the trainer.  # Startet den Trainer.

    # The worker daemon thread will be killed here.  # Der Worker-Daemon-Thread wird hier beendet.

