from argparse import ArgumentParser  # Importing a library to handle command-line arguments.  # Deutsch: Importieren einer Bibliothek zur Verarbeitung von Befehlszeilenargumenten.

# third-party imports
import numpy as np  # Importing NumPy for numerical operations.  # Deutsch: Import von NumPy für numerische Operationen.

# local imports
import tmrl.config.config_constants as cfg  # Importing configuration constants from a local module.  # Deutsch: Importieren von Konfigurationskonstanten aus einem lokalen Modul.
import tmrl.config.config_objects as cfg_obj  # Importing configuration objects from a local module.  # Deutsch: Importieren von Konfigurationsobjekten aus einem lokalen Modul.
from tmrl.envs import GenericGymEnv  # Importing a generic gym environment class.  # Deutsch: Importieren einer generischen Gym-Umgebungsklasse.
from tmrl.networking import RolloutWorker  # Importing the RolloutWorker class for handling rollouts.  # Deutsch: Importieren der RolloutWorker-Klasse für die Verarbeitung von Rollouts.
from tmrl.util import partial  # Importing a utility function `partial` from tmrl.  # Deutsch: Importieren einer Dienstprogrammfunktion `partial` aus tmrl.

def save_replays(nb_replays=np.inf):  # Function to save replays with a default number of infinite replays.  # Deutsch: Funktion zum Speichern von Wiederholungen mit einer Standardanzahl von unendlichen Wiederholungen.
    config = cfg_obj.CONFIG_DICT  # Loading the configuration dictionary from config objects.  # Deutsch: Laden des Konfigurationswörterbuchs aus Konfigurationsobjekten.
    config['interface_kwargs'] = {'save_replays': True}  # Enabling replay saving in the configuration.  # Deutsch: Aktivieren des Speicherns von Wiederholungen in der Konfiguration.
    rw = RolloutWorker(  # Creating a RolloutWorker instance with specific parameters.  # Deutsch: Erstellen einer RolloutWorker-Instanz mit bestimmten Parametern.
        env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),  # Specifying the environment class and configuration.  # Deutsch: Festlegen der Umgebungs-Klasse und Konfiguration.
        actor_module_cls=partial(cfg_obj.POLICY),  # Setting the policy class for the actor.  # Deutsch: Festlegen der Richtlinienklasse für den Akteur.
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,  # Using a sample compressor from the configuration.  # Deutsch: Verwenden eines Probenkompressors aus der Konfiguration.
        device='cuda' if cfg.CUDA_INFERENCE else 'cpu',  # Selecting the device based on CUDA availability.  # Deutsch: Auswahl des Geräts basierend auf der Verfügbarkeit von CUDA.
        server_ip=cfg.SERVER_IP_FOR_WORKER,  # Setting the server IP for the worker.  # Deutsch: Festlegen der Server-IP für den Worker.
        model_path=cfg.MODEL_PATH_WORKER,  # Providing the path to the model for the worker.  # Deutsch: Bereitstellung des Pfads zum Modell für den Worker.
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,  # Using an observation preprocessor from the configuration.  # Deutsch: Verwendung eines Beobachtungsprozessors aus der Konfiguration.
        crc_debug=cfg.CRC_DEBUG,  # Enabling or disabling CRC debugging based on the configuration.  # Deutsch: Aktivieren oder Deaktivieren des CRC-Debuggings basierend auf der Konfiguration.
        standalone=True  # Running the worker in standalone mode.  # Deutsch: Ausführen des Workers im Standalone-Modus.
    )

    rw.run_episodes(10000, nb_episodes=nb_replays)  # Running the worker for a maximum of 10,000 episodes or up to the specified number of replays.  # Deutsch: Ausführen des Workers für maximal 10.000 Episoden oder bis zur angegebenen Anzahl von Wiederholungen.

if __name__ == "__main__":  # Ensuring the script runs only when executed directly.  # Deutsch: Sicherstellen, dass das Skript nur bei direkter Ausführung ausgeführt wird.
    parser = ArgumentParser()  # Creating an argument parser instance.  # Deutsch: Erstellen einer Instanz des Argumentparsers.
    parser.add_argument('--nb_replays', type=int, default=np.inf, help='number of replays to record')  # Adding a command-line argument for the number of replays.  # Deutsch: Hinzufügen eines Befehlszeilenarguments für die Anzahl der Wiederholungen.
    args = parser.parse_args()  # Parsing the command-line arguments.  # Deutsch: Parsen der Befehlszeilenargumente.
    save_replays(args.nb_replays)  # Calling the save_replays function with the parsed argument.  # Deutsch: Aufrufen der Funktion save_replays mit dem geparsten Argument.
