# Minimal makefile for Sphinx documentation  # A minimal Makefile for Sphinx documentation.  # Minimales Makefile für Sphinx-Dokumentation.
#

# You can set these variables from the command line, and also  # These variables can be set via the command line or environment.  # Diese Variablen können über die Kommandozeile oder die Umgebung gesetzt werden.
# from the environment for the first two.  # Specifically for the first two variables below.  # Speziell für die ersten beiden Variablen unten.
SPHINXOPTS    ?=  # Optional Sphinx options, left empty by default.  # Optionale Sphinx-Optionen, standardmäßig leer.
SPHINXBUILD   ?= sphinx-build  # Command to invoke the Sphinx build system.  # Befehl zum Aufrufen des Sphinx-Build-Systems.
SOURCEDIR     = .  # Directory containing the source files (current directory).  # Verzeichnis mit den Quelldateien (aktuelles Verzeichnis).
BUILDDIR      = _build  # Directory where the output is placed.  # Verzeichnis, in dem die Ausgabedateien abgelegt werden.

# Put it first so that "make" without argument is like "make help".  # Ensures "make" defaults to "make help" if no argument is given.  # Stellt sicher, dass "make" ohne Argument standardmäßig "make help" aufruft.
help:  # Target for displaying help information.  # Ziel für die Anzeige von Hilfeinformationen.
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)  # Executes Sphinx with the "help" target.  # Führt Sphinx mit dem Ziel "help" aus.

.PHONY: help Makefile  # Declares targets that aren't associated with files.  # Deklariert Ziele, die nicht mit Dateien verbunden sind.

# Catch-all target: route all unknown targets to Sphinx using the new  # A generic target to forward all unrecognized commands to Sphinx.  # Ein allgemeines Ziel, um alle unbekannten Befehle an Sphinx weiterzuleiten.
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).  # Uses Sphinx's "make mode" with $(O) as a shortcut for options.  # Verwendet den "make mode" von Sphinx mit $(O) als Abkürzung für Optionen.
%: Makefile  # Target for any unspecified commands.  # Ziel für alle nicht spezifizierten Befehle.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)  # Forwards the target to Sphinx for processing.  # Leitet das Ziel zur Verarbeitung an Sphinx weiter.
