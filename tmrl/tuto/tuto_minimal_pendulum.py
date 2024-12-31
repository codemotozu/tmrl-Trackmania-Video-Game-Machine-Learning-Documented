# tutorial imports:
from threading import Thread  # Import threading library to run tasks in parallel.  # Importieren der Thread-Bibliothek, um Aufgaben parallel auszuführen.
import time  # Import time module for handling delays and time-based tasks.  # Importieren des Zeitmoduls für Verzögerungen und zeitbasierte Aufgaben.

# TMRL imports:
from tmrl.networking import Server, RolloutWorker, Trainer  # Import Server, RolloutWorker, and Trainer classes from the TMRL library.  # Importieren der Server-, RolloutWorker- und Trainer-Klassen aus der TMRL-Bibliothek.
from tmrl.util import partial  # Import partial for simplifying function argument binding.  # Importieren von partial, um Argumentbindung in Funktionen zu vereinfachen.
from tmrl.envs import GenericGymEnv  # Import GenericGymEnv for handling generic Gym environments.  # Importieren von GenericGymEnv zur Handhabung generischer Gym-Umgebungen.
import tmrl.config.config_constants as cfg  # Import TMRL configuration constants.  # Importieren von TMRL-Konfigurationskonstanten.
from tmrl.training_offline import TorchTrainingOffline  # Import offline training utility from TMRL.  # Importieren des Offline-Trainingsdienstprogramms von TMRL.
from tmrl.custom.custom_algorithms import SpinupSacAgent  # Import custom SAC algorithm from TMRL.  # Importieren des benutzerdefinierten SAC-Algorithmus aus TMRL.
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic  # Import neural network models for SAC.  # Importieren von neuronalen Netzwerkmodellen für SAC.
from tmrl.custom.custom_memories import GenericTorchMemory  # Import generic replay memory for training.  # Importieren eines generischen Replay-Speichers für das Training.

# Set this to True only for debugging your pipeline.
CRC_DEBUG = False  # Debug flag for pipeline (set to True to debug).  # Debug-Flag für die Pipeline (auf True setzen, um zu debuggen).

# Name used for training checkpoints and models saved in the TmrlData folder.
# If you change anything, also change this name (or delete the saved files in TmrlData).
my_run_name = "tutorial_minimal_pendulum"  # Name for saved models and checkpoints.  # Name für gespeicherte Modelle und Kontrollpunkte.

# === Environment ======================================================================================================

# Environment class:
env_cls = partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None})  # Define the environment class using Pendulum-v1.  # Definiert die Umgebungs-Klasse mit Pendulum-v1.

# Observation and action space:
dummy_env = env_cls()  # Instantiate the dummy environment to access its spaces.  # Instanziiert eine Dummy-Umgebung, um deren Räume zu verwenden.
act_space = dummy_env.action_space  # Extract action space from the environment.  # Extrahiert den Aktionsraum aus der Umgebung.
obs_space = dummy_env.observation_space  # Extract observation space from the environment.  # Extrahiert den Beobachtungsraum aus der Umgebung.

print(f"action space: {act_space}")  # Print the action space.  # Druckt den Aktionsraum aus.
print(f"observation space: {obs_space}")  # Print the observation space.  # Druckt den Beobachtungsraum aus.

# === TMRL Server ======================================================================================================

security = None  # This is fine for secure local networks. On the Internet, use "TLS" instead.  # Für lokale Netzwerke geeignet. Im Internet "TLS" verwenden.
password = cfg.PASSWORD  # Password for server authentication from the configuration file.  # Passwort für die Server-Authentifizierung aus der Konfigurationsdatei.

server_ip = "127.0.0.1"  # This is the localhost IP. Change it for your public IP if you want to run on the Internet.  # Lokale IP-Adresse. Für das Internet ändern.
server_port = 6666  # On the Internet, the machine hosting the Server needs to be reachable via this port.  # Im Internet muss der Server über diesen Port erreichbar sein.

if __name__ == "__main__":
    my_server = Server(security=security, password=password, port=server_port)  # Create the TMRL Server object.  # Erstellt das TMRL-Serverobjekt.

# === TMRL Worker ======================================================================================================

actor_module_cls = partial(SquashedGaussianMLPActor)  # Define the actor module using the Squashed Gaussian model.  # Definiert das Akteur-Modul mit dem Squashed-Gaussian-Modell.

weights_folder = cfg.WEIGHTS_FOLDER  # Folder to save model weights.  # Ordner zum Speichern der Modellgewichte.
model_path = str(weights_folder / (my_run_name + ".tmod"))  # Define path for saving the current model.  # Definiert den Pfad zum Speichern des aktuellen Modells.
model_path_history = str(weights_folder / (my_run_name + "_"))  # Define path for saving model history.  # Definiert den Pfad zum Speichern der Modellhistorie.
model_history = -1  # Disable model history saving.  # Deaktiviert das Speichern der Modellhistorie.

if __name__ == "__main__":
    my_worker = RolloutWorker(  # Instantiate the RolloutWorker object.  # Instanziert das RolloutWorker-Objekt.
        env_cls=env_cls,  # Use the previously defined environment class.  # Verwendet die zuvor definierte Umgebungs-Klasse.
        actor_module_cls=actor_module_cls,  # Use the actor module class defined earlier.  # Verwendet die zuvor definierte Akteur-Modul-Klasse.
        sample_compressor=None,  # No sample compression is used.  # Keine Kompression von Proben wird verwendet.
        device="cpu",  # Use the CPU for computations.  # Verwendet die CPU für Berechnungen.
        server_ip=server_ip,  # Connect to the server IP defined earlier.  # Verbindet sich mit der zuvor definierten Server-IP.
        server_port=server_port,  # Connect to the server port defined earlier.  # Verbindet sich mit dem zuvor definierten Server-Port.
        password=password,  # Use the server password for authentication.  # Verwendet das Serverpasswort für die Authentifizierung.
        max_samples_per_episode=1000,  # Maximum number of samples per episode.  # Maximale Anzahl von Proben pro Episode.
        model_path=model_path,  # Specify the model path for the worker.  # Gibt den Modellpfad für den Worker an.
        model_history=model_history,  # Disable model history tracking.  # Deaktiviert die Modellhistorienverfolgung.
        crc_debug=CRC_DEBUG)  # Use the debug flag for debugging purposes.  # Verwendet das Debug-Flag für Debugging-Zwecke.

# === TMRL Trainer =====================================================================================================

weights_folder = cfg.WEIGHTS_FOLDER  # Folder for model weights.  # Ordner für Modellgewichte.
checkpoints_folder = cfg.CHECKPOINTS_FOLDER  # Folder for saving checkpoints.  # Ordner zum Speichern von Kontrollpunkten.
model_path = str(weights_folder / (my_run_name + "_t.tmod"))  # Path for saving the trainer's model.  # Pfad zum Speichern des Modells des Trainers.
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))  # Path for saving checkpoints.  # Pfad zum Speichern von Kontrollpunkten.

env_cls = (obs_space, act_space)  # Define environment using observation and action space.  # Definiert die Umgebung mit Beobachtungs- und Aktionsraum.

memory_cls = partial(GenericTorchMemory,  # Define the memory class for replay buffer.  # Definiert die Speicherklasse für den Replay-Speicher.
                     memory_size=1e6,  # Replay buffer size.  # Größe des Replay-Speichers.
                     batch_size=32,  # Batch size for training.  # Batch-Größe für das Training.
                     crc_debug=CRC_DEBUG)  # Use the debug flag for memory debugging.  # Verwendet das Debug-Flag für Speicher-Debugging.

training_agent_cls = partial(SpinupSacAgent,  # Define the training agent class for SAC algorithm.  # Definiert die Trainings-Agent-Klasse für den SAC-Algorithmus.
                             model_cls=MLPActorCritic,  # Use MLPActorCritic as the model.  # Verwendet MLPActorCritic als Modell.
                             gamma=0.99,  # Discount factor for future rewards.  # Diskontierungsfaktor für zukünftige Belohnungen.
                             polyak=0.995,  # Target network update factor.  # Update-Faktor für das Zielnetzwerk.
                             alpha=0.2,  # SAC entropy tuning parameter.  # SAC-Entropieabstimmungsparameter.
                             lr_actor=1e-3,  # Learning rate for the actor.  # Lernrate für den Akteur.
                             lr_critic=1e-3,  # Learning rate for the critic.  # Lernrate für den Kritiker.
                             lr_entropy=1e-3,  # Learning rate for entropy tuning.  # Lernrate für die Entropieabstimmung.
                             learn_entropy_coef=True,  # Enable learning of entropy coefficient.  # Aktiviert das Lernen des Entropiekoeffizienten.
                             target_entropy=None)  # Automatically calculate target entropy.  # Berechnet die Zielentropie automatisch.

# Training parameters:

epochs = 2  # Maximum number of epochs, usually set this to np.inf.  # Maximale Anzahl von Epochen, normalerweise auf np.inf gesetzt.
rounds = 10  # Number of rounds per epoch.  # Anzahl der Runden pro Epoche.
steps = 1000  # Number of training steps per round.  # Anzahl der Trainingsschritte pro Runde.
update_buffer_interval = 1  # The trainer checks for incoming samples at this interval of training steps.  # Der Trainer überprüft eingehende Proben in diesem Intervall von Trainingsschritten.
update_model_interval = 1  # The trainer broadcasts its updated model at this interval of training steps.  # Der Trainer sendet sein aktualisiertes Modell in diesem Intervall von Trainingsschritten.
max_training_steps_per_env_step = 0.2  # Trainer synchronization ratio (max training steps per collected env step).  # Trainer-Synchronisationsverhältnis (max. Trainingsschritte pro gesammeltem Umweltschritt).
start_training = 100  # Minimum number of collected environment steps before training starts.  # Mindestanzahl gesammelter Umweltschritte, bevor das Training beginnt.
device = None  # Training device (None for auto selection).  # Trainingsgerät (None für automatische Auswahl).

# Training class:

training_cls = partial(  # Create a partial function for training configuration.  # Erstellt eine Partial-Funktion für die Trainingskonfiguration.
    TorchTrainingOffline,  # Offline training class using Torch.  # Offline-Trainingsklasse mit Torch.
    env_cls=env_cls,  # Environment class for the training.  # Umgebungs-Klasse für das Training.
    memory_cls=memory_cls,  # Memory class for storing experiences.  # Speicherklasse zum Speichern von Erfahrungen.
    training_agent_cls=training_agent_cls,  # Training agent class for policy updates.  # Trainingsagent-Klasse für Policy-Updates.
    epochs=epochs,  # Pass the maximum number of epochs.  # Übergebe die maximale Anzahl von Epochen.
    rounds=rounds,  # Pass the number of rounds per epoch.  # Übergebe die Anzahl der Runden pro Epoche.
    steps=steps,  # Pass the number of steps per round.  # Übergebe die Anzahl der Schritte pro Runde.
    update_buffer_interval=update_buffer_interval,  # Pass the buffer update interval.  # Übergebe das Intervall für Puffer-Updates.
    update_model_interval=update_model_interval,  # Pass the model update interval.  # Übergebe das Intervall für Modell-Updates.
    max_training_steps_per_env_step=max_training_steps_per_env_step,  # Pass synchronization ratio.  # Übergebe das Synchronisationsverhältnis.
    start_training=start_training,  # Pass minimum steps to start training.  # Übergebe die Mindestanzahl an Schritten für den Trainingsstart.
    device=device)  # Pass the training device.  # Übergebe das Trainingsgerät.

# Trainer instance:

if __name__ == "__main__":  # Main script entry point.  # Haupteinstiegspunkt des Skripts.
    my_trainer = Trainer(  # Initialize a trainer instance.  # Initialisiert eine Trainerinstanz.
        training_cls=training_cls,  # Use the configured training class.  # Verwendet die konfigurierte Trainingsklasse.
        server_ip=server_ip,  # Server IP address.  # Server-IP-Adresse.
        server_port=server_port,  # Server port number.  # Server-Portnummer.
        password=password,  # Password for the server.  # Passwort für den Server.
        model_path=model_path,  # Path to save or load the model.  # Pfad zum Speichern oder Laden des Modells.
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints.  # None, wenn keine Checkpoints gespeichert werden sollen.

# === Running the pipeline =============================================================================================

# Now we have everything we need.  # Jetzt haben wir alles, was wir brauchen.
# Typically, you will run your TMRL Server, Trainer and RolloutWorkers in different terminals / machines.  # Normalerweise werden Server, Trainer und RolloutWorker in verschiedenen Terminals oder Maschinen ausgeführt.
# But for simplicity, in this tutorial, we run them in different threads instead.  # Der Einfachheit halber werden sie in diesem Tutorial in verschiedenen Threads ausgeführt.
# Note that the Server is already running (it started running when instantiated).  # Hinweis: Der Server läuft bereits (wurde beim Instanziieren gestartet).

# Separate threads for running the RolloutWorker and Trainer:

def run_worker(worker):  # Function to run the RolloutWorker.  # Funktion zum Ausführen des RolloutWorkers.
    # For non-real-time environments, we can use the run_synchronous method.  # Für nicht-echtzeitfähige Umgebungen kann die run_synchronous-Methode verwendet werden.
    # run_synchronous enables synchronizing RolloutWorkers with the Trainer.  # run_synchronous ermöglicht die Synchronisierung von RolloutWorkers mit dem Trainer.
    # More precisely, it enables limiting the number of collected steps per worker per model update.  # Genauer gesagt wird die Anzahl der gesammelten Schritte pro Worker pro Modell-Update begrenzt.
    # initial_steps is the number of environment steps performed before waiting for the first model update.  # initial_steps ist die Anzahl von Umweltschritten vor dem ersten Modell-Update.
    # max_steps_per_update is the RolloutWorker synchronization ratio (max environment steps per model update).  # max_steps_per_update ist das Synchronisationsverhältnis des RolloutWorkers.
    # end_episodes relaxes synchronization: the worker run episodes until they are terminated/truncated before waiting.  # end_episodes lockert die Synchronisation: Der Worker beendet Episoden vor dem Warten.

    # collect training samples synchronously:
    worker.run_synchronous(test_episode_interval=10,  # Collect one test episode every 10 train episodes.  # Erfasst eine Testepisode alle 10 Trainingsepisoden.
                           initial_steps=100,  # Initial number of samples.  # Anfangszahl an Beispielen.
                           max_steps_per_update=10,  # Synchronization ratio of 10 environment steps per training step.  # Synchronisationsverhältnis von 10 Umweltschritten pro Trainingsschritt.
                           end_episodes=True)  # Wait for the episodes to end before updating the model.  # Warten bis Episoden enden, bevor das Modell aktualisiert wird.

def run_trainer(trainer):  # Function to run the trainer.  # Funktion zum Ausführen des Trainers.
    trainer.run()  # Start the trainer's main process.  # Startet den Hauptprozess des Trainers.

if __name__ == "__main__":

    daemon_thread_worker = Thread(target=run_worker, args=(my_worker,), kwargs={}, daemon=True)  # Create a daemon thread for the worker.  # Erstellt einen Daemon-Thread für den Worker.
    daemon_thread_worker.start()  # Start the worker daemon thread.  # Startet den Daemon-Thread des Workers.

    run_trainer(my_trainer)  # Start the trainer.  # Startet den Trainer.

    print("Training complete. Lazily sleeping for 1 second so that our worker thread blocks...")  # Print status and pause for synchronization.  # Status ausgeben und Pause für Synchronisation.
    time.sleep(1.0)  # Sleep for 1 second.  # Pause für 1 Sekunde.

    print("Rendering the trained policy.")  # Notify about rendering the policy.  # Meldung über das Rendern der Policy.

    rendering_worker = RolloutWorker(  # Create a rendering worker.  # Erstellt einen Render-Worker.
        standalone=True,  # Standalone mode (not connected to server).  # Standalone-Modus (nicht mit Server verbunden).
        env_cls=partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": "human"}),  # Environment configuration.  # Umgebungs-Konfiguration.
        actor_module_cls=partial(SquashedGaussianMLPActor),  # Actor module configuration.  # Konfiguration des Akteursmoduls.
        sample_compressor=None,  # No sample compression.  # Keine Komprimierung der Proben.
        device="cpu",  # Use CPU for rendering.  # CPU für das Rendern verwenden.
        max_samples_per_episode=1000,  # Maximum samples per episode.  # Maximale Probenanzahl pro Episode.
        model_path=model_path)  # Path to the trained model.  # Pfad zum trainierten Modell.

    rendering_worker.run_episodes()  # Run the rendering worker's episodes.  # Führt die Episoden des Render-Workers aus.
