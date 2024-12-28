@ECHO OFF  # Disables command echoing (prevents each command from being printed before execution).  # Deaktiviert die Befehlsanzeige (verhindert, dass jeder Befehl vor der Ausführung gedruckt wird).

pushd %~dp0  # Changes the current directory to the location of the batch file.  # Wechselt das aktuelle Verzeichnis zum Speicherort der Batch-Datei.

REM Command file for Sphinx documentation  # This is a comment indicating the purpose of the script.  # Kommentar, der den Zweck des Skripts beschreibt.

if "%SPHINXBUILD%" == "" (  # Checks if the SPHINXBUILD environment variable is empty.  # Überprüft, ob die Umgebungsvariable SPHINXBUILD leer ist.
    set SPHINXBUILD=sphinx-build  # If SPHINXBUILD is not set, assigns the default value "sphinx-build" to it.  # Wenn SPHINXBUILD nicht gesetzt ist, wird der Standardwert "sphinx-build" zugewiesen.
)

set SOURCEDIR=.  # Sets the source directory to the current directory (.).  # Setzt das Quellverzeichnis auf das aktuelle Verzeichnis (.).

set BUILDDIR=_build  # Sets the build directory to "_build".  # Setzt das Zielverzeichnis auf "_build".

%SPHINXBUILD% >NUL 2>NUL  # Executes sphinx-build and redirects both stdout and stderr to NUL (discards the output).  # Führt sphinx-build aus und leitet sowohl stdout als auch stderr an NUL weiter (verwirft die Ausgabe).

if errorlevel 9009 (  # Checks if the previous command returned an error level 9009, indicating sphinx-build is not found.  # Überprüft, ob der vorherige Befehl den Fehlerlevel 9009 zurückgegeben hat, was darauf hinweist, dass sphinx-build nicht gefunden wurde.
    echo.  # Prints a blank line.  # Gibt eine Leerzeile aus.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx  # Prints an error message.  # Gibt eine Fehlermeldung aus.
    echo.installed, then set the SPHINXBUILD environment variable to point  # Prints additional instructions.  # Gibt zusätzliche Anweisungen aus.
    echo.to the full path of the 'sphinx-build' executable. Alternatively you  # Further instructions for setting the SPHINXBUILD environment variable.  # Weitere Anweisungen zum Setzen der Umgebungsvariable SPHINXBUILD.
    echo.may add the Sphinx directory to PATH.  # Suggests adding the Sphinx directory to the system PATH.  # Schlug vor, das Sphinx-Verzeichnis zum System-PATH hinzuzufügen.
    echo.  # Prints a blank line.  # Gibt eine Leerzeile aus.
    echo.If you don't have Sphinx installed, grab it from  # Suggests installing Sphinx if not installed.  # Schlug vor, Sphinx zu installieren, wenn es nicht installiert ist.
    echo.https://www.sphinx-doc.org/  # Provides the Sphinx website for installation instructions.  # Gibt die Sphinx-Website für Installationsanweisungen an.
    exit /b 1  # Exits the batch file with an error code of 1.  # Beendet die Batch-Datei mit dem Fehlercode 1.
)

if "%1" == "" goto help  # Checks if the first argument is empty. If so, it jumps to the help label.  # Überprüft, ob das erste Argument leer ist. Falls ja, wird zum Hilfebezeichner gesprungen.

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%  # Executes sphinx-build with the given arguments and options.  # Führt sphinx-build mit den angegebenen Argumenten und Optionen aus.

goto end  # Jumps to the end label to skip the help section.  # Springt zum Endbezeichner, um den Hilfebereich zu überspringen.

:help  # Label for the help section.  # Bezeichner für den Hilfebereich.
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%  # Executes sphinx-build with the "help" argument to show help information.  # Führt sphinx-build mit dem "help"-Argument aus, um Hilfeinformationen anzuzeigen.

:end  # End label, marks the end of the script.  # Endbezeichner, markiert das Ende des Skripts.

popd  # Restores the previous directory (undoes the pushd).  # Stellt das vorherige Verzeichnis wieder her (macht das pushd rückgängig).
