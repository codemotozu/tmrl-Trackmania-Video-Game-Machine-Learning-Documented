# Minimal makefile for Sphinx documentation  # Minimaler Makefile für Sphinx-Dokumentation
#

# You can set these variables from the command line, and also   # Diese Variablen können über die Kommandozeile oder
# from the environment for the first two.                      # aus der Umgebung für die ersten zwei festgelegt werden.
SPHINXOPTS    ?=                                             # Optionen für Sphinx, können leer bleiben.  # Sphinx-Optionen, die standardmäßig leer sind.
SPHINXBUILD   ?= sphinx-build                                # Befehl zum Starten des Sphinx-Build-Prozesses.  # Befehl für den Sphinx-Bauprozess.
SOURCEDIR     = .                                           # Quellverzeichnis, Standard ist das aktuelle Verzeichnis.  # Verzeichnis mit den Quellen.
BUILDDIR      = _build                                      # Zielverzeichnis für die generierten Dokumente.  # Zielverzeichnis.

# Put it first so that "make" without argument is like "make help".  # "make" ohne Argument ruft standardmäßig "make help" auf.  # "make help" als Standardziel.
help:                                                      # Ziel "help", um Hilfeinformationen zu zeigen.
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)  # Führt Sphinx-Build im Hilfemodus aus.  # Ruft Hilfe-Optionen von Sphinx ab.

.PHONY: help Makefile                                       # Deklariert "help" und "Makefile" als Phony-Ziele.  # "Phony-Ziele" werden nicht als Dateien behandelt.

# Catch-all target: route all unknown targets to Sphinx using the new  # Fängt alle unbekannten Ziele ab und leitet sie an Sphinx weiter.  # Für neue Sphinx-Make-Modi.
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).  # $(O) ist eine Abkürzung für $(SPHINXOPTS).  # Optionen weiterleiten.
%: Makefile                                                # Platzhalter-Ziel, das alle unbekannten Zielnamen abfängt.  # Wildcard-Ziel.
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)  # Führt den Build für ein angegebenes Ziel aus.  # Generiert Dokumentation.
