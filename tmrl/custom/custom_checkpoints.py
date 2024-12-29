import os  # Importing the OS module to work with the filesystem.  # Importieren des OS-Moduls zur Arbeit mit dem Dateisystem.
import tarfile  # Importing the tarfile module to handle tar archives.  # Importieren des Tarfile-Moduls zur Handhabung von Tar-Archiven.
from pathlib import Path  # Importing Path from pathlib to handle filesystem paths.  # Importieren von Path aus pathlib zur Arbeit mit Dateipfaden.
import itertools  # Importing itertools for advanced iterator tools.  # Importieren von itertools für erweiterte Iterator-Tools.

from torch.optim import Adam  # Importing Adam optimizer from PyTorch.  # Importieren des Adam-Optimierers aus PyTorch.
import numpy as np  # Importing NumPy for numerical computations.  # Importieren von NumPy für numerische Berechnungen.
import torch  # Importing PyTorch for deep learning functionalities.  # Importieren von PyTorch für Deep-Learning-Funktionen.

from tmrl.config import config_constants as cfg  # Importing configuration constants from tmrl.  # Importieren von Konfigurationskonstanten aus tmrl.
from tmrl.util import dump, load  # Importing dump and load utility functions.  # Importieren der Dump- und Load-Hilfsfunktionen.
import logging  # Importing logging for logging messages.  # Importieren von Logging zum Protokollieren von Nachrichten.

def load_run_instance_images_dataset(checkpoint_path):  
    """
    Function used to load trainers from checkpoint path.  # Funktion zum Laden von Trainern aus dem Checkpoint-Pfad.
    Args:  
        checkpoint_path: the path where instances of run_cls are checkpointed.  # checkpoint_path: Pfad, an dem Instanzen von run_cls gespeichert werden.
    Returns:  
        An instance of run_cls loaded from checkpoint_path.  # Eine Instanz von run_cls, die aus checkpoint_path geladen wurde.
    """
    chk_path = Path(checkpoint_path)  # Converting checkpoint_path to a Path object.  # Konvertieren von checkpoint_path in ein Path-Objekt.
    parent_path = chk_path.parent.absolute()  # Getting the absolute path of the parent directory.  # Absoluten Pfad des übergeordneten Verzeichnisses abrufen.
    tar_path = str(parent_path / 'dataset.tar')  # Constructing the tar file path.  # Erstellen des Tar-Dateipfads.
    dataset_path = str(cfg.DATASET_PATH)  # Getting the dataset path from the configuration.  # Abrufen des Dataset-Pfads aus der Konfiguration.
    logging.debug(f" load: tar_path :{tar_path}")  # Logging the tar path for debugging.  # Protokollieren des Tar-Pfads zur Fehlerbehebung.
    logging.debug(f" load: dataset_path :{dataset_path}")  # Logging the dataset path for debugging.  # Protokollieren des Dataset-Pfads zur Fehlerbehebung.
    with tarfile.open(tar_path, 'r') as t:  # Opening the tar file in read mode.  # Öffnen der Tar-Datei im Lesemodus.
        t.extractall(dataset_path)  # Extracting all files to the dataset path.  # Extrahieren aller Dateien in den Dataset-Pfad.
    return load(checkpoint_path)  # Loading the checkpoint and returning it.  # Laden des Checkpoints und Rückgabe.

def dump_run_instance_images_dataset(run_instance, checkpoint_path):  
    """
    Function used to dump trainers to checkpoint path.  # Funktion zum Speichern von Trainern im Checkpoint-Pfad.
    Args:  
        run_instance: the instance of run_cls to checkpoint.  # run_instance: Instanz von run_cls, die gespeichert werden soll.
        checkpoint_path: the path where instances of run_cls are checkpointed.  # checkpoint_path: Pfad, an dem Instanzen von run_cls gespeichert werden.
    """
    chk_path = Path(checkpoint_path)  # Converting checkpoint_path to a Path object.  # Konvertieren von checkpoint_path in ein Path-Objekt.
    parent_path = chk_path.parent.absolute()  # Getting the absolute path of the parent directory.  # Absoluten Pfad des übergeordneten Verzeichnisses abrufen.
    tar_path = str(parent_path / 'dataset.tar')  # Constructing the tar file path.  # Erstellen des Tar-Dateipfads.
    dataset_path = str(cfg.DATASET_PATH)  # Getting the dataset path from the configuration.  # Abrufen des Dataset-Pfads aus der Konfiguration.
    logging.debug(f" dump: tar_path :{tar_path}")  # Logging the tar path for debugging.  # Protokollieren des Tar-Pfads zur Fehlerbehebung.
    logging.debug(f" dump: dataset_path :{dataset_path}")  # Logging the dataset path for debugging.  # Protokollieren des Dataset-Pfads zur Fehlerbehebung.
    with tarfile.open(tar_path, 'w') as tar_handle:  # Opening the tar file in write mode.  # Öffnen der Tar-Datei im Schreibmodus.
        for root, dirs, files in os.walk(dataset_path):  # Iterating through the dataset directory.  # Durchlaufen des Dataset-Verzeichnisses.
            for file in files:  # Iterating through files in the directory.  # Durchlaufen der Dateien im Verzeichnis.
                tar_handle.add(os.path.join(root, file), arcname=file)  # Adding files to the tar archive.  # Hinzufügen von Dateien zum Tar-Archiv.
    dump(run_instance, checkpoint_path)  # Saving the run_instance to checkpoint_path.  # Speichern der run_instance in checkpoint_path.

def update_memory(run_instance):  
    steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]  # Retrieving training steps per round from config.  # Abrufen der Trainingsschritte pro Runde aus der Konfiguration.
    memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]  # Retrieving memory size from config.  # Abrufen der Speichergröße aus der Konfiguration.
    batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]  # Retrieving batch size from config.  # Abrufen der Batch-Größe aus der Konfiguration.
    if run_instance.steps != steps \  # Checking if the current steps differ from config.  # Überprüfen, ob sich die aktuellen Schritte von der Konfiguration unterscheiden.
            or run_instance.memory.batch_size != batch_size \  # Checking if the batch size differs.  # Überprüfen, ob sich die Batch-Größe unterscheidet.
            or run_instance.memory.memory_size != memory_size:  # Checking if the memory size differs.  # Überprüfen, ob sich die Speichergröße unterscheidet.
        run_instance.steps = steps  # Updating steps in the run_instance.  # Aktualisieren der Schritte in der run_instance.
        run_instance.memory.nb_steps = steps  # Updating memory steps.  # Aktualisieren der Speicher-Schritte.
        run_instance.memory.batch_size = batch_size  # Updating batch size in memory.  # Aktualisieren der Batch-Größe im Speicher.
        run_instance.memory.memory_size = memory_size  # Updating memory size.  # Aktualisieren der Speichergröße.
        logging.info(f"Memory updated with steps:{steps}, batch size:{batch_size}, memory size:{memory_size}.")  # Logging updated values.  # Protokollieren der aktualisierten Werte.
    return run_instance  # Returning the updated run_instance.  # Rückgabe der aktualisierten run_instance.



def update_run_instance(run_instance, training_cls):  # Function to update a checkpoint with config values.  # Funktion, um ein Checkpoint mit Konfigurationswerten zu aktualisieren.
    """
    Updates the checkpoint after loading with compatible values from config.json

    Args:
        run_instance: the instance of the checkpoint to update
        training_cls: partially instantiated class of a new checkpoint (to replace run_instance if needed)

    Returns:
        run_instance: the updated checkpoint
    """  # Documentation for the function.  # Dokumentation für die Funktion.

    if "RESET_TRAINING" in cfg.TMRL_CONFIG and cfg.TMRL_CONFIG["RESET_TRAINING"]:  # Check if a reset is required.  # Überprüfen, ob ein Reset erforderlich ist.
        new_run_instance = training_cls()  # Create a new checkpoint instance.  # Erstellen einer neuen Checkpoint-Instanz.
        new_run_instance.memory = run_instance.memory  # Transfer memory to the new instance.  # Übertragen des Speichers auf die neue Instanz.
        new_run_instance = update_memory(new_run_instance)  # Update the memory of the new instance.  # Aktualisieren des Speichers der neuen Instanz.
        new_run_instance.total_samples = len(new_run_instance.memory)  # Update the sample count.  # Aktualisieren der Stichprobenanzahl.
        return new_run_instance  # Return the new instance.  # Rückgabe der neuen Instanz.

    ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]  # Load algorithm configuration.  # Laden der Algorithmus-Konfiguration.
    ALG_NAME = ALG_CONFIG["ALGORITHM"]  # Retrieve the algorithm name.  # Abrufen des Algorithmusnamens.
    assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not supported by this checkpoint updater."  # Ensure algorithm is supported.  # Sicherstellen, dass der Algorithmus unterstützt wird.

    if ALG_NAME in ["SAC", "REDQSAC"]:  # Check if algorithm is SAC or REDQSAC.  # Überprüfen, ob der Algorithmus SAC oder REDQSAC ist.
        lr_actor = ALG_CONFIG["LR_ACTOR"]  # Get learning rate for the actor.  # Abrufen der Lernrate für den Akteur.
        lr_critic = ALG_CONFIG["LR_CRITIC"]  # Get learning rate for the critic.  # Abrufen der Lernrate für den Kritiker.
        lr_entropy = ALG_CONFIG["LR_ENTROPY"]  # Get entropy learning rate.  # Abrufen der Entropie-Lernrate.
        gamma = ALG_CONFIG["GAMMA"]  # Get gamma value.  # Abrufen des Gamma-Werts.
        polyak = ALG_CONFIG["POLYAK"]  # Get polyak coefficient.  # Abrufen des Polyak-Koeffizienten.
        learn_entropy_coef = ALG_CONFIG["LEARN_ENTROPY_COEF"]  # Check if entropy coefficient is learned.  # Überprüfen, ob der Entropie-Koeffizient gelernt wird.
        target_entropy = ALG_CONFIG["TARGET_ENTROPY"]  # Get target entropy.  # Abrufen der Zielentropie.
        alpha = ALG_CONFIG["ALPHA"]  # Get alpha value.  # Abrufen des Alpha-Werts.

        if ALG_NAME == "SAC":  # Check if the algorithm is SAC.  # Überprüfen, ob der Algorithmus SAC ist.
            if run_instance.agent.lr_actor != lr_actor:  # Compare current actor learning rate.  # Vergleichen der aktuellen Lernrate des Akteurs.
                old = run_instance.agent.lr_actor  # Store old value.  # Alten Wert speichern.
                run_instance.agent.lr_actor = lr_actor  # Update actor learning rate.  # Lernrate des Akteurs aktualisieren.
                run_instance.agent.pi_optimizer = Adam(run_instance.agent.model.actor.parameters(), lr=lr_actor)  # Reinitialize actor optimizer.  # Akteur-Optimierer neu initialisieren.
                logging.info(f"Actor optimizer reinitialized with new lr: {lr_actor} (old lr: {old}).")  # Log the update.  # Update protokollieren.

            if run_instance.agent.lr_critic != lr_critic:  # Compare current critic learning rate.  # Vergleichen der aktuellen Lernrate des Kritikers.
                old = run_instance.agent.lr_critic  # Store old value.  # Alten Wert speichern.
                run_instance.agent.lr_critic = lr_critic  # Update critic learning rate.  # Lernrate des Kritikers aktualisieren.
                run_instance.agent.q_optimizer = Adam(itertools.chain(run_instance.agent.model.q1.parameters(), run_instance.agent.model.q2.parameters()), lr=lr_critic)  # Reinitialize critic optimizer.  # Kritiker-Optimierer neu initialisieren.
                logging.info(f"Critic optimizer reinitialized with new lr: {lr_critic} (old lr: {old}).")  # Log the update.  # Update protokollieren.

        if run_instance.agent.learn_entropy_coef != learn_entropy_coef:  # Check for entropy coefficient learning mismatch.  # Überprüfen von Abweichungen beim Lernen des Entropie-Koeffizienten.
            logging.warning(f"Cannot switch entropy learning.")  # Log a warning.  # Eine Warnung protokollieren.

        if run_instance.agent.lr_entropy != lr_entropy or run_instance.agent.alpha != alpha:  # Check for entropy or alpha mismatch.  # Überprüfen von Abweichungen bei Entropie oder Alpha.
            run_instance.agent.lr_entropy = lr_entropy  # Update entropy learning rate.  # Lernrate der Entropie aktualisieren.
            run_instance.agent.alpha = alpha  # Update alpha value.  # Alpha-Wert aktualisieren.
            device = run_instance.device or ("cuda" if torch.cuda.is_available() else "cpu")  # Determine device type.  # Gerätetyp bestimmen.
            if run_instance.agent.learn_entropy_coef:  # Check if entropy coefficient is learned.  # Überprüfen, ob der Entropie-Koeffizient gelernt wird.
                run_instance.agent.log_alpha = torch.log(torch.ones(1) * run_instance.agent.alpha).to(device).requires_grad_(True)  # Update alpha with gradient requirement.  # Alpha mit Gradientenanforderung aktualisieren.
                run_instance.agent.alpha_optimizer = Adam([run_instance.agent.log_alpha], lr=lr_entropy)  # Reinitialize alpha optimizer.  # Alpha-Optimierer neu initialisieren.
                logging.info(f"Entropy optimizer reinitialized.")  # Log reinitialization.  # Neuinitialisierung protokollieren.
            else:
                run_instance.agent.alpha_t = torch.tensor(float(run_instance.agent.alpha)).to(device)  # Update alpha tensor.  # Alpha-Tensor aktualisieren.
                logging.info(f"Alpha changed to {alpha}.")  # Log alpha change.  # Alpha-Änderung protokollieren.

        if run_instance.agent.gamma != gamma:  # Check gamma mismatch.  # Überprüfen von Gamma-Abweichungen.
            old = run_instance.agent.gamma  # Store old gamma value.  # Alten Gamma-Wert speichern.
            run_instance.agent.gamma = gamma  # Update gamma value.  # Gamma-Wert aktualisieren.
            logging.info(f"Gamma coefficient changed to {gamma} (old: {old}).")  # Log gamma change.  # Gamma-Änderung protokollieren.

        if run_instance.agent.polyak != polyak:  # Check polyak mismatch.  # Überprüfen von Polyak-Abweichungen.
            old = run_instance.agent.polyak  # Store old polyak value.  # Alten Polyak-Wert speichern.
            run_instance.agent.polyak = polyak  # Update polyak value.  # Polyak-Wert aktualisieren.
            logging.info(f"Polyak coefficient changed to {polyak} (old: {old}).")  # Log polyak change.  # Polyak-Änderung protokollieren.

        if target_entropy is None:  # Check for automatic entropy coefficient.  # Überprüfen des automatischen Entropie-Koeffizienten.
            action_space = run_instance.agent.action_space  # Retrieve action space.  # Aktionsraum abrufen.
            run_instance.agent.target_entropy = -np.prod(action_space.shape)  # Calculate target entropy.  # Zielentropie berechnen.
        else:
            run_instance.agent.target_entropy = float(target_entropy)  # Set target entropy.  # Zielentropie setzen.
        logging.info(f"Target entropy: {run_instance.agent.target_entropy}.")  # Log target entropy.  # Zielentropie protokollieren.

        if ALG_NAME == "REDQSAC":  # Check if the algorithm is REDQSAC.  # Überprüfen, ob der Algorithmus REDQSAC ist.
            m = ALG_CONFIG["REDQ_M"]  # Get m value for REDQ.  # m-Wert für REDQ abrufen.
            q_updates_per_policy_update = ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]  # Get Q updates per policy update.  # Q-Updates pro Politik-Update abrufen.

            if run_instance.agent.q_updates_per_policy_update != q_updates_per_policy_update:  # Check for Q update mismatch.  # Überprüfen von Abweichungen bei Q-Updates.
                old = run_instance.agent.q_updates_per_policy_update  # Store old Q update value.  # Alten Q-Update-Wert speichern.
                run_instance.agent.q_updates_per_policy_update = q_updates_per_policy_update  # Update Q update ratio.  # Q-Update-Verhältnis aktualisieren.
                logging.info(f"Q update ratio switched to {q_updates_per_policy_update} (old: {old}).")  # Log the change.  # Änderung protokollieren.

            if run_instance.agent.m != m:  # Check for m mismatch.  # Überprüfen von Abweichungen beim m-Wert.
                old = run_instance.agent.m  # Store old m value.  # Alten m-Wert speichern.
                run_instance.agent.m = m  # Update m value.  # m-Wert aktualisieren.
                logging.info(f"M switched to {m} (old: {old}).")  # Log m change.  # m-Änderung protokollieren.

    # Updating other configurations:
    epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]  # Get max epochs.  # Maximalanzahl der Epochen abrufen.
    rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]  # Get rounds per epoch.  # Runden pro Epoche abrufen.
    update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]  # Get model update interval.  # Modell-Update-Intervall abrufen.
    update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]  # Get buffer update interval.  # Puffer-Update-Intervall abrufen.
    max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]  # Get max training steps per environment step.  # Maximale Trainingsschritte pro Umgebungsschritt abrufen.
    profiling = cfg.PROFILE_TRAINER  # Get profiling configuration.  # Profiling-Konfiguration abrufen.
    start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]  # Get steps before training starts.  # Schritte vor Trainingsbeginn abrufen.

    if run_instance.epochs != epochs:  # Check for epoch mismatch.  # Überprüfen von Abweichungen bei Epochen.
        old = run_instance.epochs  # Store old value.  # Alten Wert speichern.
        run_instance.epochs = epochs  # Update max epochs.  # Maximale Epochen aktualisieren.
        logging.info(f"Max epochs changed to {epochs} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.rounds != rounds:  # Check for round mismatch.  # Überprüfen von Abweichungen bei Runden.
        old = run_instance.rounds  # Store old value.  # Alten Wert speichern.
        run_instance.rounds = rounds  # Update rounds per epoch.  # Runden pro Epoche aktualisieren.
        logging.info(f"Rounds per epoch changed to {rounds} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.update_model_interval != update_model_interval:  # Check for update model interval mismatch.  # Überprüfen von Abweichungen beim Modell-Update-Intervall.
        old = run_instance.update_model_interval  # Store old value.  # Alten Wert speichern.
        run_instance.update_model_interval = update_model_interval  # Update model interval.  # Modell-Intervall aktualisieren.
        logging.info(f"Model update interval changed to {update_model_interval} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.update_buffer_interval != update_buffer_interval:  # Check for buffer update interval mismatch.  # Überprüfen von Abweichungen beim Puffer-Update-Intervall.
        old = run_instance.update_buffer_interval  # Store old value.  # Alten Wert speichern.
        run_instance.update_buffer_interval = update_buffer_interval  # Update buffer interval.  # Puffer-Intervall aktualisieren.
        logging.info(f"Buffer update interval changed to {update_buffer_interval} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.max_training_steps_per_env_step != max_training_steps_per_env_step:  # Check for training step mismatch.  # Überprüfen von Abweichungen bei Trainingsschritten.
        old = run_instance.max_training_steps_per_env_step  # Store old value.  # Alten Wert speichern.
        run_instance.max_training_steps_per_env_step = max_training_steps_per_env_step  # Update training step ratio.  # Trainingsschritt-Verhältnis aktualisieren.
        logging.info(f"Max train/env step ratio changed to {max_training_steps_per_env_step} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.profiling != profiling:  # Check for profiling mismatch.  # Überprüfen von Abweichungen beim Profiling.
        old = run_instance.profiling  # Store old value.  # Alten Wert speichern.
        run_instance.profiling = profiling  # Update profiling configuration.  # Profiling-Konfiguration aktualisieren.
        logging.info(f"Profiling switched to {profiling} (old: {old}).")  # Log change.  # Änderung protokollieren.

    if run_instance.start_training != start_training:  # Check for training start mismatch.  # Überprüfen von Abweichungen beim Trainingsstart.
        old = run_instance.start_training  # Store old value.  # Alten Wert speichern.
        run_instance.start_training = start_training  # Update training start configuration.  # Trainingsstart-Konfiguration aktualisieren.
        logging.info(f"Number of environment steps before training changed to {start_training} (old: {old}).")  # Log change.  # Änderung protokollieren.

    run_instance = update_memory(run_instance)  # Update memory for the instance.  # Speicher für die Instanz aktualisieren.

    return run_instance  # Return the updated instance.  # Rückgabe der aktualisierten Instanz.

