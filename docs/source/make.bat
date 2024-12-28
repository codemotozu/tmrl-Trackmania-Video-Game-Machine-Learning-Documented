@ECHO OFF  # Turns off command echoing to avoid displaying each command in the console.  # Deaktiviert die Befehlsausgabe, um zu vermeiden, dass jeder Befehl in der Konsole angezeigt wird.
pushd %~dp0  # Changes the current directory to the location of the batch script.  # Wechselt das aktuelle Verzeichnis zum Speicherort des Batch-Skripts.

REM Command file for Sphinx documentation  # A comment explaining that the script is used for Sphinx documentation.  # Ein Kommentar, der erklärt, dass das Skript für die Sphinx-Dokumentation verwendet wird.

if "%SPHINXBUILD%" == "" (  # Checks if the environment variable SPHINXBUILD is not set.  # Überprüft, ob die Umgebungsvariable SPHINXBUILD nicht gesetzt ist.
    set SPHINXBUILD=sphinx-build  # If SPHINXBUILD is empty, set it to 'sphinx-build' command.  # Wenn SPHINXBUILD leer ist, wird es auf den Befehl 'sphinx-build' gesetzt.
)
set SOURCEDIR=.  # Sets the source directory to the current directory (.).  # Setzt das Quellverzeichnis auf das aktuelle Verzeichnis (.) .
set BUILDDIR=_build  # Sets the build directory to '_build'.  # Setzt das Build-Verzeichnis auf '_build'.

%SPHINXBUILD% >NUL 2>NUL  # Executes the 'sphinx-build' command, redirecting both standard output and error to NUL (essentially discarding them).  # Führt den Befehl 'sphinx-build' aus und leitet sowohl Standardausgabe als auch Fehlerausgabe an NUL weiter (wird praktisch verworfen).
if errorlevel 9009 (  # Checks if the error level is 9009, indicating that the 'sphinx-build' command was not found.  # Überprüft, ob das Fehlerlevel 9009 ist, was darauf hinweist, dass der Befehl 'sphinx-build' nicht gefunden wurde.
    echo.  # Prints an empty line.  # Gibt eine leere Zeile aus.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx  # Prints a message indicating that the 'sphinx-build' command was not found.  # Gibt eine Nachricht aus, dass der Befehl 'sphinx-build' nicht gefunden wurde.
    echo.installed, then set the SPHINXBUILD environment variable to point  # Informs the user to install Sphinx and set the SPHINXBUILD variable.  # Informiert den Benutzer, Sphinx zu installieren und die Umgebungsvariable SPHINXBUILD zu setzen.
    echo.to the full path of the 'sphinx-build' executable. Alternatively you  # Suggests adding the Sphinx directory to the PATH.  # Schlägt vor, das Sphinx-Verzeichnis zum PATH hinzuzufügen.
    echo.may add the Sphinx directory to PATH.  # Informs about adding the Sphinx directory to the PATH.  # Informiert darüber, das Sphinx-Verzeichnis zum PATH hinzuzufügen.
    echo.  # Prints another empty line.  # Gibt eine weitere leere Zeile aus.
    echo.If you don't have Sphinx installed, grab it from  # Informs where to get Sphinx if not installed.  # Informiert darüber, wo man Sphinx herunterladen kann, falls es nicht installiert ist.
    echo.https://www.sphinx-doc.org/  # Provides the Sphinx website URL.  # Gibt die URL der Sphinx-Website an.
    exit /b 1  # Exits the batch script with an error code 1.  # Beendet das Batch-Skript mit dem Fehlercode 1.
)

if "%1" == "" goto help  # Checks if the first argument is empty, if so, jumps to the help section.  # Überprüft, ob das erste Argument leer ist, und springt bei Bedarf zum Hilfebereich.

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%  # Runs the 'sphinx-build' command with the given options to build documentation.  # Führt den Befehl 'sphinx-build' mit den angegebenen Optionen aus, um die Dokumentation zu erstellen.
goto end  # Jumps to the end section of the script.  # Springt zum Ende des Skripts.

:help  # A label for the help section of the script.  # Ein Label für den Hilfebereich des Skripts.
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%  # Runs the 'sphinx-build' command with the 'help' argument to display help information.  # Führt den Befehl 'sphinx-build' mit dem Argument 'help' aus, um Hilfeinformationen anzuzeigen.

:end  # A label marking the end of the script.  # Ein Label, das das Ende des Skripts markiert.
popd  # Returns to the previous directory before the script was executed.  # Wechselt zurück in das vorherige Verzeichnis, bevor das Skript ausgeführt wurde.
