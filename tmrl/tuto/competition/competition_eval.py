"""
This script is used to evaluate your submission to the TMRL competition.  # Script zum Bewerten Ihrer Einreichung für den TMRL-Wettbewerb.
It assumes the script where you implemented your ActorModule is in the same folder and is named "custom_actor_module.py".  # Es wird davon ausgegangen, dass sich das Skript mit Ihrem ActorModule im selben Ordner befindet und "custom_actor_module.py" heißt.
It also assumes your ActorModule implementation is named "MyActorModule".  # Es wird auch angenommen, dass Ihre ActorModule-Implementierung "MyActorModule" heißt.
When using this script, don't forget to set "SLEEP_TIME_AT_RESET" to 0.0 in config.json.  # Beim Verwenden dieses Skripts vergessen Sie nicht, "SLEEP_TIME_AT_RESET" in config.json auf 0.0 zu setzen.
"""

from tmrl.networking import RolloutWorker  # Import RolloutWorker for running environment rollouts.  # Importiert RolloutWorker für die Durchführung von Umgebungsdurchläufen.
from tmrl.util import partial  # Import the partial function to customize environment creation.  # Importiert die partial-Funktion zur Anpassung der Umgebungs-Erstellung.
from tmrl.envs import GenericGymEnv  # Import GenericGymEnv for creating a TrackMania Gymnasium environment.  # Importiert GenericGymEnv zur Erstellung einer TrackMania-Gymnasium-Umgebung.
import tmrl.config.config_constants as cfg  # Import configuration constants for setup.  # Importiert Konfigurationskonstanten für die Einrichtung.
import tmrl.config.config_objects as cfg_obj  # Import configuration objects for the environment.  # Importiert Konfigurationsobjekte für die Umgebung.

from custom_actor_module import MyActorModule  # change this to match your ActorModule name  # Ändern Sie dies, damit es mit Ihrem ActorModule-Namen übereinstimmt.

# rtgym environment class (full TrackMania2020 Gymnasium environment with replays enabled):  # rtgym-Umgebungsklasse (vollständige TrackMania2020-Gymnasium-Umgebung mit aktivierten Wiederholungen):
config = cfg_obj.CONFIG_DICT  # Load the configuration dictionary.  # Lädt das Konfigurationswörterbuch.
config['interface_kwargs'] = {'save_replays': True}  # Enable saving of replays in the environment.  # Aktiviert das Speichern von Wiederholungen in der Umgebung.
env_cls = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config})  # Create a partial environment class with specific configuration.  # Erstellt eine partielle Umgebungsklasse mit spezifischer Konfiguration.

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)  # Gerät, das für die Inferenz bei Workern verwendet wird (kann geändert werden, aber beachten Sie, dass die Wettbewerbsbewertung auf der CPU erfolgt).
device_worker = 'cpu'  # Set the inference device to CPU.  # Setzt das Inferenzgerät auf die CPU.

try:
    from custom_actor_module import obs_preprocessor  # Try importing a custom observation preprocessor.  # Versucht, einen benutzerdefinierten Beobachtungsvorverarbeiter zu importieren.
except Exception as e:
    obs_preprocessor = cfg_obj.OBS_PREPROCESSOR  # Fallback to default observation preprocessor in case of error.  # Fallback auf den Standard-Beobachtungsvorverarbeiter bei Fehlern.

if __name__ == "__main__":  # Main script entry point.  # Einstiegspunkt des Hauptskripts.
    rw = RolloutWorker(env_cls=env_cls,  # Initialize a RolloutWorker with environment class.  # Initialisiert einen RolloutWorker mit der Umgebungsklasse.
                       actor_module_cls=MyActorModule,  # Specify the actor module class.  # Gibt die ActorModule-Klasse an.
                       device=device_worker,  # Specify the device for inference.  # Gibt das Gerät für die Inferenz an.
                       obs_preprocessor=obs_preprocessor,  # Specify the observation preprocessor.  # Gibt den Beobachtungsvorverarbeiter an.
                       standalone=True)  # Run the worker in standalone mode.  # Führt den Worker im Standalone-Modus aus.
    rw.run_episodes()  # Start running episodes in the environment.  # Startet das Ausführen von Episoden in der Umgebung.
