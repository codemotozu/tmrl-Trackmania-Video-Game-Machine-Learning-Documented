# Command Line Interface
# ======================
# tmrl provides commands for users who wish to use the readily implemented example pipelines for TrackMania.
# tmrl bietet Befehle für Benutzer, die die vorgefertigten Beispielpipelines für TrackMania nutzen möchten.

# Examples:
# ---------
# Launch the default training pipeline for TrackMania on 3 possibly different machines:
# Starte die Standard-Trainingspipeline für TrackMania auf 3 möglicherweise unterschiedlichen Maschinen:

.. code-block:: bash
   python -m tmrl --server  # Start the server for the training pipeline.  # Server für die Trainingspipeline starten.
   python -m tmrl --trainer  # Launch the trainer for the pipeline.  # Trainer für die Pipeline starten.
   python -m tmrl --worker  # Launch the worker to handle tasks in the pipeline.  # Arbeiter starten, um Aufgaben in der Pipeline zu verarbeiten.

# Test (deploy) the readily trained example policy for TrackMania:
# Teste (implementiere) die vorgefertigte Trainingspolitik für TrackMania:

.. code-block:: bash
   python -m tmrl --test  # Run the test to deploy the example policy.  # Test ausführen, um die Beispielrichtlinie zu implementieren.

# Launch the reward recorder in your own track in TrackMania:
# Starte den Belohnungsrekorder auf deiner eigenen Strecke in TrackMania:

.. code-block:: bash
   python -m tmrl --record-reward  # Record rewards for a custom TrackMania track.  # Belohnungen für eine benutzerdefinierte TrackMania-Strecke aufzeichnen.

# Check that the TrackMania environment is working as expected:
# Überprüfe, ob die TrackMania-Umgebung wie erwartet funktioniert:

.. code-block:: bash
   python -m tmrl --check-environment  # Verify if the environment is set up correctly.  # Überprüfen, ob die Umgebung korrekt eingerichtet ist.

# Benchmark the RolloutWorker in TrackMania (requires `"benchmark":true` in `config.json`):
# Benchmarking des RolloutWorkers in TrackMania (erfordert `"benchmark":true` in `config.json`):

.. code-block:: bash
   python -m tmrl --benchmark  # Run benchmarks for performance evaluation.  # Benchmarks für Leistungsevaluierung ausführen.

# Launch the Trainer but disable logging to wandb.ai:
# Starte den Trainer, aber deaktiviere das Logging zu wandb.ai:

.. code-block:: bash
   python -m tmrl --trainer --no-wandb  # Start the trainer without wandb.ai logging.  # Trainer starten ohne wandb.ai-Logging.
