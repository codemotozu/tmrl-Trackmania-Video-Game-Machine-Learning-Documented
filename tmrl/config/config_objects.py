# third-party imports
# from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset  # Commented out imports for loading/saving datasets.  # Kommentierte Importe zum Laden/Speichern von Datensätzen.
# third-party imports

import numpy as np  # Import NumPy for numerical operations.  # Importiere NumPy für numerische Berechnungen.
import rtgym  # Import the real-time gym library for gym environments.  # Importiere die Echtzeit-Gym-Bibliothek für Gym-Umgebungen.

# local imports
import tmrl.config.config_constants as cfg  # Import constants for configuration from TMRL.  # Importiere Konfigurationskonstanten von TMRL.
from tmrl.training_offline import TorchTrainingOffline  # Import offline training logic.  # Importiere die Offline-Trainingslogik.
from tmrl.custom.custom_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarProgress  # Import TrackMania gym interfaces.  # Importiere TrackMania-Gym-Schnittstellen.
from tmrl.custom.custom_memories import MemoryTMFull, MemoryTMLidar, MemoryTMLidarProgress, get_local_buffer_sample_lidar, get_local_buffer_sample_lidar_progress, get_local_buffer_sample_tm20_imgs  # Import memory utilities and sampling functions.  # Importiere Speicherfunktionen und Sampling-Methoden.
from tmrl.custom.custom_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_tm_lidar_progress_act_in_obs  # Import observation preprocessing methods.  # Importiere Vorverarbeitungsmethoden für Beobachtungen.
from tmrl.envs import GenericGymEnv  # Import the generic gym environment.  # Importiere die generische Gym-Umgebung.
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic, REDQMLPActorCritic, RNNActorCritic, SquashedGaussianRNNActor, SquashedGaussianVanillaCNNActor, VanillaCNNActorCritic, SquashedGaussianVanillaColorCNNActor, VanillaColorCNNActorCritic  # Import actor-critic models and policy structures.  # Importiere Actor-Critic-Modelle und Politikstrukturen.
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent  # Import the SAC algorithm implementation.  # Importiere die SAC-Algorithmus-Implementierung.
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent  # Import the REDQ SAC algorithm implementation.  # Importiere die REDQ-SAC-Algorithmus-Implementierung.
from tmrl.custom.custom_checkpoints import update_run_instance  # Import checkpoint update functionality.  # Importiere Funktionen zum Aktualisieren von Checkpoints.
from tmrl.util import partial  # Import a utility function for partial application.  # Importiere eine Dienstprogrammfunktion für partielle Anwendung.

ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]  # Load algorithm configuration from TMRL config.  # Lade Algorithmuskonfiguration aus der TMRL-Konfiguration.
ALG_NAME = ALG_CONFIG["ALGORITHM"]  # Retrieve the algorithm name.  # Hole den Algorithmusnamen.
assert ALG_NAME in ["SAC", "REDQSAC"], f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."  # Ensure the algorithm name is valid.  # Stelle sicher, dass der Algorithmusname gültig ist.

# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_LIDAR:  # Check if LIDAR mode is enabled.  # Überprüfe, ob der LIDAR-Modus aktiviert ist.
    if cfg.PRAGMA_RNN:  # Check if RNN mode is enabled.  # Überprüfe, ob der RNN-Modus aktiviert ist.
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."  # Ensure only SAC is supported with RNN.  # Stelle sicher, dass nur SAC mit RNN unterstützt wird.
        TRAIN_MODEL = RNNActorCritic  # Set the model to RNN-based actor-critic.  # Setze das Modell auf RNN-basierte Actor-Critic.
        POLICY = SquashedGaussianRNNActor  # Set the policy to a squashed Gaussian RNN actor.  # Setze die Politik auf einen Squashed-Gaussian-RNN-Actor.
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic  # Choose MLP-based model based on algorithm.  # Wähle ein MLP-basiertes Modell basierend auf dem Algorithmus.
        POLICY = SquashedGaussianMLPActor  # Set the policy to a squashed Gaussian MLP actor.  # Setze die Politik auf einen Squashed-Gaussian-MLP-Actor.
else:
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"  # Ensure RNN is disabled in non-LIDAR mode.  # Stelle sicher, dass RNN im Nicht-LIDAR-Modus deaktiviert ist.
    assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."  # Only SAC is supported in this mode.  # Nur SAC wird in diesem Modus unterstützt.
    TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic  # Choose grayscale or color CNN model.  # Wähle ein Graustufen- oder Farb-CNN-Modell.
    POLICY = SquashedGaussianVanillaCNNActor if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor  # Set grayscale or color policy.  # Setze Graustufen- oder Farbpolitik.

if cfg.PRAGMA_LIDAR:  # If LIDAR mode is active.  # Wenn der LIDAR-Modus aktiv ist.
    if cfg.PRAGMA_PROGRESS:  # Check if progress tracking is enabled.  # Überprüfe, ob Fortschrittsverfolgung aktiviert ist.
        INT = partial(TM2020InterfaceLidarProgress, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)  # Use LIDAR with progress tracking interface.  # Verwende die LIDAR-Schnittstelle mit Fortschrittsverfolgung.
    else:
        INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)  # Use basic LIDAR interface.  # Verwende die grundlegende LIDAR-Schnittstelle.
else:
    INT = partial(TM2020Interface,  # Use the standard TrackMania interface.  # Verwende die Standard-TrackMania-Schnittstelle.
                  img_hist_len=cfg.IMG_HIST_LEN,  # Set the image history length.  # Setze die Länge des Bildverlaufs.
                  gamepad=cfg.PRAGMA_GAMEPAD,  # Enable or disable gamepad support.  # Aktiviere oder deaktiviere Gamepad-Unterstützung.
                  grayscale=cfg.GRAYSCALE,  # Set grayscale mode.  # Aktiviere den Graustufenmodus.
                  resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))  # Resize images to specified dimensions.  # Passe Bilder an die angegebenen Abmessungen an.

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()  # Create a copy of the default configuration dictionary.  # Erstelle eine Kopie des Standardkonfigurationsdictionaries.
CONFIG_DICT["interface"] = INT  # Assign the selected interface to the configuration.  # Weisen Sie der Konfiguration die ausgewählte Schnittstelle zu.
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]  # Load additional configuration modifiers.  # Lade zusätzliche Konfigurationsmodifikatoren.
for k, v in CONFIG_DICT_MODIFIERS.items():  # Iterate through configuration updates.  # Iteriere durch Konfigurationsaktualisierungen.
    CONFIG_DICT[k] = v  # Apply each modification.  # Wende jede Modifikation an.

# to compress a sample before sending it over the local network/Internet:
if cfg.PRAGMA_LIDAR:  # Check if lidar data processing is enabled.  # Überprüfen, ob Lidar-Datenverarbeitung aktiviert ist.
    if cfg.PRAGMA_PROGRESS:  # Check if progress tracking is enabled.  # Überprüfen, ob Fortschrittsverfolgung aktiviert ist.
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress  # Use lidar with progress sample compressor.  # Verwenden Sie Lidar mit Fortschrittsdaten-Komprimierung.
    else:  
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar  # Use lidar sample compressor without progress tracking.  # Verwenden Sie Lidar-Daten-Komprimierung ohne Fortschrittsverfolgung.
else:  
    SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs  # Use image-based sample compressor instead.  # Verwenden Sie stattdessen bildbasierte Datenkomprimierung.

# to preprocess observations that come out of the gymnasium environment:
if cfg.PRAGMA_LIDAR:  # Check if lidar data preprocessing is needed.  # Überprüfen, ob Lidar-Datenvorverarbeitung erforderlich ist.
    if cfg.PRAGMA_PROGRESS:  # Check if progress-related preprocessing is enabled.  # Überprüfen, ob Fortschrittsverarbeitung aktiviert ist.
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs  # Use lidar-progress observation preprocessor.  # Verwenden Sie Lidar-Fortschritts-Datenvorverarbeitung.
    else:  
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs  # Use lidar observation preprocessor without progress tracking.  # Verwenden Sie Lidar-Datenvorverarbeitung ohne Fortschrittsverfolgung.
else:  
    OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs  # Use the image-based observation preprocessor.  # Verwenden Sie die bildbasierte Datenvorverarbeitung.

# to augment data that comes out of the replay buffer:
SAMPLE_PREPROCESSOR = None  # No additional data augmentation is used.  # Keine zusätzliche Datenaugmentation wird verwendet.

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"  # Ensure RNNs are not used because they are not supported.  # Sicherstellen, dass RNNs nicht verwendet werden, da sie nicht unterstützt werden.

if cfg.PRAGMA_LIDAR:  # Check if lidar memory handling is needed.  # Überprüfen, ob Lidar-Speicherverarbeitung erforderlich ist.
    if cfg.PRAGMA_RNN:  # Check if RNN-specific lidar memory is required.  # Überprüfen, ob RNN-spezifische Lidar-Speicher benötigt werden.
        assert False, "not implemented"  # RNN with lidar is not implemented.  # RNN mit Lidar ist nicht implementiert.
    else:  
        if cfg.PRAGMA_PROGRESS:  # Check for lidar-progress memory handling.  # Überprüfen, ob Lidar-Fortschritts-Speicher verwendet wird.
            MEM = MemoryTMLidarProgress  # Use lidar-progress memory class.  # Verwenden Sie Lidar-Fortschritts-Speicherklasse.
        else:  
            MEM = MemoryTMLidar  # Use lidar memory class without progress tracking.  # Verwenden Sie Lidar-Speicherklasse ohne Fortschrittsverfolgung.
else:  
    MEM = MemoryTMFull  # Use the default full memory class for images.  # Verwenden Sie die Standard-Speicherklasse für Bilder.

MEMORY = partial(MEM,  # Configure the memory with parameters.  # Konfigurieren Sie den Speicher mit Parametern.
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],  # Set memory size.  # Speichergröße festlegen.
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],  # Set batch size for training.  # Batchgröße für das Training festlegen.
                 sample_preprocessor=SAMPLE_PREPROCESSOR,  # Set the sample preprocessor.  # Datenprozessor einstellen.
                 dataset_path=cfg.DATASET_PATH,  # Set dataset storage path.  # Pfad für Datenspeicherung einstellen.
                 imgs_obs=cfg.IMG_HIST_LEN,  # Set image observation history length.  # Länge des Bildverlaufs festlegen.
                 act_buf_len=cfg.ACT_BUF_LEN,  # Set action buffer length.  # Länge des Aktionspuffers festlegen.
                 crc_debug=cfg.CRC_DEBUG)  # Enable or disable CRC debugging.  # CRC-Debugging aktivieren oder deaktivieren.

# ALGORITHM: ===================================================

if ALG_NAME == "SAC":  # Check if the algorithm is "SAC".  # Prüfen, ob der Algorithmus "SAC" ist.
    AGENT = partial(  # Define the SAC agent with specified parameters.  # Definieren des SAC-Agenten mit den angegebenen Parametern.
        SAC_Agent,  # Use the SAC_Agent class.  # Verwendung der SAC_Agent-Klasse.
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',  # Use GPU if CUDA is enabled, otherwise CPU.  # GPU verwenden, wenn CUDA aktiviert ist, sonst CPU.
        model_cls=TRAIN_MODEL,  # Specify the training model class.  # Die Trainingsmodellklasse angeben.
        lr_actor=ALG_CONFIG["LR_ACTOR"],  # Learning rate for the actor.  # Lernrate für den Actor.
        lr_critic=ALG_CONFIG["LR_CRITIC"],  # Learning rate for the critic.  # Lernrate für den Critic.
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],  # Learning rate for the entropy coefficient.  # Lernrate für den Entropie-Koeffizienten.
        gamma=ALG_CONFIG["GAMMA"],  # Discount factor for future rewards.  # Abzinsungsfaktor für zukünftige Belohnungen.
        polyak=ALG_CONFIG["POLYAK"],  # Polyak averaging factor for target network updates.  # Polyak-Mittelungsfaktor für Zielnetzwerk-Updates.
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # Whether to learn the entropy coefficient.  # Ob der Entropie-Koeffizient gelernt werden soll.
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # Target entropy for temperature adjustment.  # Zielentropie für Temperaturanpassung.
        alpha=ALG_CONFIG["ALPHA"],  # Inverse reward scale (temperature parameter).  # Inverse Belohnungsskala (Temperaturparameter).
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],  # Optimizer for the actor.  # Optimierer für den Actor.
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],  # Optimizer for the critic.  # Optimierer für den Critic.
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,  # Betas for the actor optimizer.  # Betas für den Actor-Optimierer.
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,  # Betas for the critic optimizer.  # Betas für den Critic-Optimierer.
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,  # L2 regularization for the actor.  # L2-Regularisierung für den Actor.
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None  # L2 regularization for the critic.  # L2-Regularisierung für den Critic.
    )
else:  # Use the REDQ algorithm if the algorithm name is not "SAC".  # Verwenden des REDQ-Algorithmus, wenn der Algorithmusname nicht "SAC" ist.
    AGENT = partial(  # Define the REDQ agent with specified parameters.  # Definieren des REDQ-Agenten mit den angegebenen Parametern.
        REDQ_Agent,  # Use the REDQ_Agent class.  # Verwendung der REDQ_Agent-Klasse.
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',  # Use GPU if CUDA is enabled, otherwise CPU.  # GPU verwenden, wenn CUDA aktiviert ist, sonst CPU.
        model_cls=TRAIN_MODEL,  # Specify the training model class.  # Die Trainingsmodellklasse angeben.
        lr_actor=ALG_CONFIG["LR_ACTOR"],  # Learning rate for the actor.  # Lernrate für den Actor.
        lr_critic=ALG_CONFIG["LR_CRITIC"],  # Learning rate for the critic.  # Lernrate für den Critic.
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],  # Learning rate for the entropy coefficient.  # Lernrate für den Entropie-Koeffizienten.
        gamma=ALG_CONFIG["GAMMA"],  # Discount factor for future rewards.  # Abzinsungsfaktor für zukünftige Belohnungen.
        polyak=ALG_CONFIG["POLYAK"],  # Polyak averaging factor for target network updates.  # Polyak-Mittelungsfaktor für Zielnetzwerk-Updates.
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # Whether to learn the entropy coefficient.  # Ob der Entropie-Koeffizient gelernt werden soll.
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # Target entropy for temperature adjustment.  # Zielentropie für Temperaturanpassung.
        alpha=ALG_CONFIG["ALPHA"],  # Inverse reward scale (temperature parameter).  # Inverse Belohnungsskala (Temperaturparameter).
        n=ALG_CONFIG["REDQ_N"],  # Number of Q networks.  # Anzahl der Q-Netzwerke.
        m=ALG_CONFIG["REDQ_M"],  # Number of Q targets.  # Anzahl der Q-Ziele.
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]  # Q updates per policy update.  # Q-Updates pro Policy-Update.
    )

# TRAINER: =====================================================

def sac_v2_entropy_scheduler(agent, epoch):  # Defines a function to adjust the entropy target for the SAC algorithm based on the epoch. # Definiert eine Funktion, die das Entropie-Ziel für den SAC-Algorithmus basierend auf der Epoche anpasst.
    start_ent = -0.0  # Initial entropy value. # Anfangswert für die Entropie.
    end_ent = -7.0  # Final entropy value. # Endwert für die Entropie.
    end_epoch = 200  # Number of epochs over which the entropy changes. # Anzahl der Epochen, über die die Entropie verändert wird.
    if epoch <= end_epoch:  # Checks if the current epoch is within the specified range. # Überprüft, ob die aktuelle Epoche innerhalb des angegebenen Bereichs liegt.
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch  # Linearly interpolates the entropy value for the current epoch. # Interpoliert den Entropiewert für die aktuelle Epoche linear.

ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})  
# Partially initializes a gym environment with a specified ID and configuration dictionary.  
# Initialisiert teilweise eine Gym-Umgebung mit einer angegebenen ID und Konfigurations-Dictionary.

if cfg.PRAGMA_LIDAR:  # lidar  # Checks if LiDAR-based configurations are enabled. # Überprüft, ob LiDAR-basierte Konfigurationen aktiviert sind.
    TRAINER = partial(
        TorchTrainingOffline,  # Initializes the training class using offline Torch training. # Initialisiert die Trainingsklasse mit offline Torch-Training.
        env_cls=ENV_CLS,  # Specifies the environment class. # Gibt die Umgebungsklasse an.
        memory_cls=MEMORY,  # Specifies the memory class for the training process. # Gibt die Speicherklasse für den Trainingsprozess an.
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],  # Sets the maximum number of training epochs. # Legt die maximale Anzahl von Trainingsepochen fest.
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],  # Sets the number of rounds per epoch. # Legt die Anzahl der Runden pro Epoche fest.
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],  # Specifies the number of training steps per round. # Gibt die Anzahl der Trainingsschritte pro Runde an.
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],  # Sets how often the model is updated. # Legt fest, wie oft das Modell aktualisiert wird.
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],  # Sets how often the buffer is updated. # Legt fest, wie oft der Puffer aktualisiert wird.
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],  
        # Defines the maximum training steps per environment step. # Definiert die maximalen Trainingsschritte pro Umgebungsschritt.
        profiling=cfg.PROFILE_TRAINER,  # Enables or disables profiling for the trainer. # Aktiviert oder deaktiviert das Profiling für den Trainer.
        training_agent_cls=AGENT,  # Specifies the class of the training agent. # Gibt die Klasse des Trainingsagenten an.
        agent_scheduler=None,  # sac_v2_entropy_scheduler  # Currently no scheduler is defined for the agent. # Derzeit ist kein Scheduler für den Agenten definiert.
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  
        # Sets how many environment steps to run before starting training. # Legt fest, wie viele Umgebungsschritte vor Beginn des Trainings ausgeführt werden sollen.
else:  # images  # Runs the same logic as above but for image-based configurations. # Führt die gleiche Logik wie oben aus, jedoch für bildbasierte Konfigurationen.
    TRAINER = partial(
        TorchTrainingOffline,  # Initializes the training class using offline Torch training. # Initialisiert die Trainingsklasse mit offline Torch-Training.
        env_cls=ENV_CLS,  # Specifies the environment class. # Gibt die Umgebungsklasse an.
        memory_cls=MEMORY,  # Specifies the memory class for the training process. # Gibt die Speicherklasse für den Trainingsprozess an.
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],  # Sets the maximum number of training epochs. # Legt die maximale Anzahl von Trainingsepochen fest.
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],  # Sets the number of rounds per epoch. # Legt die Anzahl der Runden pro Epoche fest.
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],  # Specifies the number of training steps per round. # Gibt die Anzahl der Trainingsschritte pro Runde an.
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],  # Sets how often the model is updated. # Legt fest, wie oft das Modell aktualisiert wird.
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],  # Sets how often the buffer is updated. # Legt fest, wie oft der Puffer aktualisiert wird.
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],  
        # Defines the maximum training steps per environment step. # Definiert die maximalen Trainingsschritte pro Umgebungsschritt.
        profiling=cfg.PROFILE_TRAINER,  # Enables or disables profiling for the trainer. # Aktiviert oder deaktiviert das Profiling für den Trainer.
        training_agent_cls=AGENT,  # Specifies the class of the training agent. # Gibt die Klasse des Trainingsagenten an.
        agent_scheduler=None,  # sac_v2_entropy_scheduler  # Currently no scheduler is defined for the agent. # Derzeit ist kein Scheduler für den Agenten definiert.
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  
        # Sets how many environment steps to run before starting training. # Legt fest, wie viele Umgebungsschritte vor Beginn des Trainings ausgeführt werden sollen.

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset  
# Specifies a function for dumping run instances; none is used here. # Gibt eine Funktion zum Speichern von Run-Instanzen an; hier wird keine verwendet.
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset  
# Specifies a function for loading run instances; none is used here. # Gibt eine Funktion zum Laden von Run-Instanzen an; hier wird keine verwendet.
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC"] else None  
# Sets the updater function based on the algorithm name. # Legt die Updater-Funktion basierend auf dem Algorithmusnamen fest.
