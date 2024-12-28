# Command Line Interface
# English: Header for the command-line interface section.
# Deutsch: Überschrift für den Abschnitt zur Befehlszeilenschnittstelle.

======================
# English: Section separator.
# Deutsch: Abschnittstrenner.

``tmrl`` provides commands for users who wish to use the readily implemented example pipelines for TrackMania.
# English: The `tmrl` module offers commands for utilizing predefined example pipelines for the game TrackMania.
# Deutsch: Das `tmrl`-Modul bietet Befehle, um vordefinierte Beispiel-Pipelines für das Spiel TrackMania zu nutzen.

Examples:
---------
# English: Subsection for examples with a title separator.
# Deutsch: Unterabschnitt für Beispiele mit einem Titeltrenner.

Launch the default training pipeline for TrackMania on 3 possibly different machines:
# English: Explains how to start the default training pipeline on three different machines.
# Deutsch: Erklärt, wie man die Standard-Trainingspipeline auf drei verschiedenen Rechnern startet.

.. code-block:: bash
# English: Indicates that the following code block is in bash script format.
# Deutsch: Zeigt an, dass der folgende Codeblock im Bash-Skriptformat ist.

   python -m tmrl --server
   # English: Launches the server process for the training pipeline.
   # Deutsch: Startet den Serverprozess für die Trainingspipeline.

   python -m tmrl --trainer
   # English: Starts the trainer process to train models.
   # Deutsch: Startet den Trainerprozess, um Modelle zu trainieren.

   python -m tmrl --worker
   # English: Runs the worker process for performing tasks in the pipeline.
   # Deutsch: Führt den Worker-Prozess aus, um Aufgaben in der Pipeline durchzuführen.

Test (deploy) the readily trained example policy for TrackMania:
# English: Command to test or deploy a pre-trained policy for TrackMania.
# Deutsch: Befehl, um eine vortrainierte Policy für TrackMania zu testen oder bereitzustellen.

.. code-block:: bash
# English: Indicates the code format for the bash shell.
# Deutsch: Zeigt das Codeformat für die Bash-Shell an.

   python -m tmrl --test
   # English: Runs the test mode with the trained policy.
   # Deutsch: Startet den Testmodus mit der trainierten Policy.

Launch the reward recorder in your own track in TrackMania:
# English: Command to record rewards from a custom TrackMania track.
# Deutsch: Befehl, um Belohnungen auf einer benutzerdefinierten TrackMania-Strecke aufzuzeichnen.

.. code-block:: bash
# English: Designates the following block as bash commands.
# Deutsch: Bezeichnet den folgenden Block als Bash-Befehle.

   python -m tmrl --record-reward
   # English: Runs the reward recorder tool in TrackMania.
   # Deutsch: Führt das Tool zur Belohnungsaufzeichnung in TrackMania aus.

Check that the TrackMania environment is working as expected:
# English: Verifies the TrackMania environment configuration.
# Deutsch: Überprüft die Konfiguration der TrackMania-Umgebung.

.. code-block:: bash
# English: Marks the code block as bash commands.
# Deutsch: Markiert den Codeblock als Bash-Befehle.

   python -m tmrl --check-environment
   # English: Runs diagnostics to ensure the environment is correctly set up.
   # Deutsch: Führt Diagnosen aus, um sicherzustellen, dass die Umgebung korrekt eingerichtet ist.

Benchmark the RolloutWorker in TrackMania (requires `"benchmark":true` in `config.json`):
# English: Provides a command to benchmark the RolloutWorker functionality.
# Deutsch: Gibt einen Befehl zum Benchmarking der RolloutWorker-Funktionalität an.

.. code-block:: bash
# English: Declares the following section as bash commands.
# Deutsch: Deklariert den folgenden Abschnitt als Bash-Befehle.

   python -m tmrl --benchmark
   # English: Benchmarks the RolloutWorker if the config is set correctly.
   # Deutsch: Benchmarkt den RolloutWorker, wenn die Konfiguration korrekt gesetzt ist.

Launch the Trainer but disable logging to wandb.ai:
# English: Starts the Trainer process without logging data to wandb.ai.
# Deutsch: Startet den Trainerprozess ohne Protokollierung der Daten auf wandb.ai.

.. code-block:: bash
# English: Indicates that the following lines are bash commands.
# Deutsch: Zeigt an, dass die folgenden Zeilen Bash-Befehle sind.

   python -m tmrl --trainer --no-wandb
   # English: Disables logging to wandb.ai while running the Trainer.
   # Deutsch: Deaktiviert die Protokollierung auf wandb.ai, während der Trainer läuft.
