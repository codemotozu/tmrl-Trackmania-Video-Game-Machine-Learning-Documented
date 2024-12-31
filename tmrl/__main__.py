import json  # Imports the JSON library for working with JSON data.  # Importiert die JSON-Bibliothek zur Verarbeitung von JSON-Daten.
import logging  # Imports the logging library for logging messages.  # Importiert die Logging-Bibliothek zum Aufzeichnen von Nachrichten.
import time  # Imports the time library to work with time-related functions.  # Importiert die Time-Bibliothek für zeitbezogene Funktionen.
from argparse import ArgumentParser, ArgumentTypeError  # Imports modules for parsing command-line arguments.  # Importiert Module zum Parsen von Kommandozeilenargumenten.

# local imports
import tmrl.config.config_constants as cfg  # Imports configuration constants from a local module.  # Importiert Konfigurationskonstanten aus einem lokalen Modul.
import tmrl.config.config_objects as cfg_obj  # Imports configuration objects from a local module.  # Importiert Konfigurationsobjekte aus einem lokalen Modul.
from tmrl.envs import GenericGymEnv  # Imports a class for creating gym environments.  # Importiert eine Klasse zur Erstellung von Gym-Umgebungen.
from tmrl.networking import Server, Trainer, RolloutWorker  # Imports classes for networking-related tasks.  # Importiert Klassen für netzwerkbezogene Aufgaben.
from tmrl.tools.check_environment import check_env_tm20lidar, check_env_tm20full  # Imports functions to check the environment.  # Importiert Funktionen zur Überprüfung der Umgebung.
from tmrl.tools.record import record_reward_dist  # Imports a function for recording reward distribution.  # Importiert eine Funktion zur Aufzeichnung der Belohnungsverteilung.
from tmrl.util import partial  # Imports the 'partial' function for creating partial functions.  # Importiert die 'partial'-Funktion zum Erstellen von Teufunktionen.

def main(args):  # Main function to handle the execution of different modes based on the input arguments.  # Hauptfunktion zur Ausführung verschiedener Modi basierend auf den Eingabeargumenten.
    if args.server:  # Checks if the 'server' argument is provided.  # Überprüft, ob das Argument 'server' angegeben wurde.
        serv = Server()  # Creates an instance of the Server class.  # Erstellt eine Instanz der Server-Klasse.
        while True:  # Infinite loop to keep the server running.  # Unendliche Schleife, um den Server am Laufen zu halten.
            time.sleep(1.0)  # Pauses the loop for 1 second.  # Pausiert die Schleife für 1 Sekunde.
    elif args.worker or args.test or args.benchmark or args.expert:  # Checks for other possible modes (worker, test, benchmark, or expert).  # Überprüft andere mögliche Modi (worker, test, benchmark oder expert).
        config = cfg_obj.CONFIG_DICT  # Loads the configuration dictionary.  # Lädt das Konfigurations-Wörterbuch.
        config_modifiers = args.config  # Gets the configuration modifiers from the input arguments.  # Holt die Konfigurationsmodifikatoren aus den Eingabeargumenten.
        for k, v in config_modifiers.items():  # Iterates over the modifiers and applies them to the configuration.  # Iteriert über die Modifikatoren und wendet sie auf die Konfiguration an.
            config[k] = v  # Modifies the configuration with new values.  # Modifiziert die Konfiguration mit neuen Werten.
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),  # Creates a RolloutWorker instance with environment and configuration details.  # Erstellt eine RolloutWorker-Instanz mit Umgebungs- und Konfigurationsdetails.
                           actor_module_cls=cfg_obj.POLICY,  # Sets the policy module class.  # Legt die Modulklasse der Politik fest.
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,  # Sets the sample compressor.  # Legt den Probenkompressor fest.
                           device='cuda' if cfg.CUDA_INFERENCE else 'cpu',  # Chooses 'cuda' if GPU is available, otherwise 'cpu'.  # Wählt 'cuda', wenn eine GPU verfügbar ist, andernfalls 'cpu'.
                           server_ip=cfg.SERVER_IP_FOR_WORKER,  # Specifies the server IP address for the worker.  # Gibt die Server-IP-Adresse für den Worker an.
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,  # Sets the maximum number of samples per episode.  # Legt die maximale Anzahl an Proben pro Episode fest.
                           model_path=cfg.MODEL_PATH_WORKER,  # Specifies the model path for the worker.  # Gibt den Modellpfad für den Worker an.
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,  # Sets the observation preprocessor.  # Legt den Beobachtungs-Präprozessor fest.
                           crc_debug=cfg.CRC_DEBUG,  # Enables CRC debug mode if true.  # Aktiviert den CRC-Debug-Modus, wenn wahr.
                           standalone=args.test)  # Sets whether to run in standalone mode (for testing).  # Legt fest, ob im Standalone-Modus (für Tests) ausgeführt werden soll.
        if args.worker:  # If the 'worker' argument is provided, runs the worker.  # Wenn das Argument 'worker' angegeben wurde, wird der Worker ausgeführt.
            rw.run()  # Starts the worker's execution.  # Startet die Ausführung des Workers.
        elif args.expert:  # If the 'expert' argument is provided, runs the expert mode.  # Wenn das Argument 'expert' angegeben wurde, wird der Expertenmodus ausgeführt.
            rw.run(expert=True)  # Starts the worker in expert mode.  # Startet den Worker im Expertenmodus.
        elif args.benchmark:  # If the 'benchmark' argument is provided, runs the benchmark mode.  # Wenn das Argument 'benchmark' angegeben wurde, wird der Benchmark-Modus ausgeführt.
            rw.run_env_benchmark(nb_steps=1000, test=False)  # Runs the environment benchmark for 1000 steps.  # Führt den Benchmark der Umgebung für 1000 Schritte aus.
        else:  # If none of the above, runs 10000 episodes.  # Wenn keiner der oben genannten Modi zutrifft, werden 10000 Episoden ausgeführt.
            rw.run_episodes(10000)  # Starts running 10000 episodes.  # Startet die Ausführung von 10000 Episoden.
    elif args.trainer:  # Checks if the 'trainer' argument is provided.  # Überprüft, ob das Argument 'trainer' angegeben wurde.
        trainer = Trainer(training_cls=cfg_obj.TRAINER,  # Creates a Trainer instance with training class details.  # Erstellt eine Trainer-Instanz mit Trainingsklassen-Details.
                          server_ip=cfg.SERVER_IP_FOR_TRAINER,  # Specifies the server IP address for the trainer.  # Gibt die Server-IP-Adresse für den Trainer an.
                          model_path=cfg.MODEL_PATH_TRAINER,  # Specifies the model path for the trainer.  # Gibt den Modellpfad für den Trainer an.
                          checkpoint_path=cfg.CHECKPOINT_PATH,  # Specifies the checkpoint path.  # Gibt den Checkpoint-Pfad an.
                          dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,  # Specifies the function for dumping run instances.  # Gibt die Funktion zum Speichern von Ausführungsinstanzen an.
                          load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,  # Specifies the function for loading run instances.  # Gibt die Funktion zum Laden von Ausführungsinstanzen an.
                          updater_fn=cfg_obj.UPDATER_FN)  # Specifies the function for updating models.  # Gibt die Funktion zum Aktualisieren von Modellen an.
        logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")  # Logs a message indicating the start of training.  # Protokolliert eine Nachricht, die den Beginn des Trainings anzeigt.
        if args.wandb:  # If the 'wandb' argument is provided, uses Weights & Biases for logging.  # Wenn das Argument 'wandb' angegeben wurde, wird Weights & Biases zum Protokollieren verwendet.
            trainer.run_with_wandb(entity=cfg.WANDB_ENTITY,  # Runs the trainer with Weights & Biases entity.  # Führt den Trainer mit einer Weights & Biases-Entität aus.
                                   project=cfg.WANDB_PROJECT,  # Specifies the project name for Weights & Biases.  # Gibt den Projektnamen für Weights & Biases an.
                                   run_id=cfg.WANDB_RUN_ID)  # Specifies the run ID for Weights & Biases.  # Gibt die Run-ID für Weights & Biases an.
        else:  # If 'wandb' is not provided, runs the trainer normally.  # Wenn 'wandb' nicht angegeben wurde, wird der Trainer normal ausgeführt.
            trainer.run()  # Starts the trainer's execution.  # Startet die Ausführung des Trainers.
    elif args.record_reward:  # If the 'record-reward' argument is provided, records the reward distribution.  # Wenn das Argument 'record-reward' angegeben wurde, wird die Belohnungsverteilung aufgezeichnet.
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=args.use_keyboard)  # Records reward distribution using the specified path.  # Zeichnet die Belohnungsverteilung unter Verwendung des angegebenen Pfades auf.
    elif args.check_env:  # If the 'check-environment' argument is provided, checks the environment.  # Wenn das Argument 'check-environment' angegeben wurde, wird die Umgebung überprüft.
        if cfg.PRAGMA_LIDAR:  # Checks if LIDAR environment is enabled.  # Überprüft, ob die LIDAR-Umgebung aktiviert ist.
            check_env_tm20lidar()  # Checks the environment using LIDAR.  # Überprüft die Umgebung mit LIDAR.
        else:  # If LIDAR is not enabled, checks the environment in full mode.  # Wenn LIDAR nicht aktiviert ist, wird die Umgebung im Vollmodus überprüft.
            check_env_tm20full()  # Checks the full environment.  # Überprüft die vollständige Umgebung.
    elif args.install:  # If the 'install' argument is provided, logs the TMRL folder path.  # Wenn das Argument 'install' angegeben wurde, wird der TMRL-Ordnerpfad protokolliert.
        logging.info(f"TMRL folder: {cfg.TMRL_FOLDER}")  # Logs the TMRL folder path.  # Protokolliert den TMRL-Ordnerpfad.
    else:  # If none of the above conditions are met, raises an error.  # Wenn keine der oben genannten Bedingungen zutrifft, wird ein Fehler ausgelöst.
        raise ArgumentTypeError('Enter a valid argument')  # Raises an error for invalid arguments.  # Löst einen Fehler für ungültige Argumente aus.

if __name__ == "__main__":  # Ensures the main function is executed when the script is run directly.  # Stellt sicher, dass die Hauptfunktion ausgeführt wird, wenn das Skript direkt ausgeführt wird.
    parser = ArgumentParser()  # Creates an ArgumentParser instance to parse command-line arguments.  # Erstellt eine ArgumentParser-Instanz zum Parsen von Kommandozeilenargumenten.
    parser.add_argument('--install', action='store_true', help='checks TMRL installation')  # Defines the 'install' argument.  # Definiert das Argument 'install'.
    parser.add_argument('--server', action='store_true', help='launches the server')  # Defines the 'server' argument.  # Definiert das Argument 'server'.
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')  # Defines the 'trainer' argument.  # Definiert das Argument 'trainer'.
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')  # Defines the 'worker' argument.  # Definiert das Argument 'worker'.
    parser.add_argument('--expert', action='store_true', help='launches an expert rollout worker (no model update)')  # Defines the 'expert' argument.  # Definiert das Argument 'expert'.
    parser.add_argument('--test', action='store_true', help='runs inference without training')  # Defines the 'test' argument.  # Definiert das Argument 'test'.
    parser.add_argument('--benchmark', action='store_true', help='runs a benchmark of the environment')  # Defines the 'benchmark' argument.  # Definiert das Argument 'benchmark'.
    parser.add_argument('--record-reward', dest='record_reward', action='store_true', help='utility to record a reward function in TM20')  # Defines the 'record-reward' argument.  # Definiert das Argument 'record-reward'.
    parser.add_argument('--use-keyboard', dest='use_keyboard', action='store_true', help='modifier for --record-reward')  # Defines the 'use-keyboard' argument.  # Definiert das Argument 'use-keyboard'.
    parser.add_argument('--check-environment', dest='check_env', action='store_true', help='utility to check the environment')  # Defines the 'check-environment' argument.  # Definiert das Argument 'check-environment'.
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='(use with --trainer) if you want to log results on Weights and Biases, use this option')  # Defines the 'wandb' argument.  # Definiert das Argument 'wandb'.
    parser.add_argument('-d', '--config', type=json.loads, default={}, help='dictionary containing configuration options (modifiers) for the rtgym environment')  # Defines the 'config' argument.  # Definiert das Argument 'config'.
    arguments = parser.parse_args()  # Parses the input arguments.  # Parst die Eingabeargumente.

    main(arguments)  # Calls the main function with parsed arguments.  # Ruft die Hauptfunktion mit den geparsten Argumenten auf.
