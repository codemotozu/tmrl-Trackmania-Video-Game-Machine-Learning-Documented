# standard library imports
import functools  # Provides higher-order functions for functional programming.  # Bietet Funktionen höherer Ordnung für funktionale Programmierung.
import operator  # Contains functions that correspond to standard operators.  # Enthält Funktionen, die den Standardoperatoren entsprechen.
import inspect  # Allows introspection of live objects, such as functions.  # Ermöglicht die Introspektion von lebenden Objekten, wie Funktionen.
import io  # Provides the Python interfaces for handling file streams.  # Bietet die Python-Schnittstellen für die Verarbeitung von Datei-Streams.
import json  # Provides functions for working with JSON data.  # Bietet Funktionen zur Arbeit mit JSON-Daten.
import os  # Provides a way to interact with the operating system, like file and directory management.  # Ermöglicht die Interaktion mit dem Betriebssystem, z. B. Datei- und Verzeichnisverwaltung.
import pickle  # Implements serializing and deserializing of Python objects.  # Implementiert die Serialisierung und Deserialisierung von Python-Objekten.
import signal  # Provides mechanisms to use signal handlers in programs.  # Bietet Mechanismen zur Verwendung von Signalhandlern in Programmen.
import subprocess  # Allows running new applications or processes.  # Ermöglicht das Ausführen neuer Anwendungen oder Prozesse.
import weakref  # Allows the creation of weak references to objects.  # Ermöglicht das Erstellen von schwachen Verweisen auf Objekte.
from pathlib import Path  # Provides easy manipulation of filesystem paths.  # Ermöglicht die einfache Manipulation von Dateisystempfaden.
# from contextlib import contextmanager  # Allows defining context managers easily.  # Ermöglicht das einfache Definieren von Kontextmanagern.
# from dataclasses import Field, dataclass, fields, is_dataclass, make_dataclass  # Provides tools for defining data classes.  # Bietet Werkzeuge zum Definieren von Datenklassen.
from importlib import import_module  # Provides the ability to import a module programmatically.  # Ermöglicht das programmgesteuerte Importieren eines Moduls.
# from itertools import chain  # Allows working with iterators to chain iterables together.  # Ermöglicht das Arbeiten mit Iteratoren, um Iterables zusammenzuführen.
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Type, TypeVar, Union  # Imports for type hinting.  # Importiert Typen für die Typisierung.
# from weakref import WeakKeyDictionary  # Allows creating dictionaries where keys are weak references.  # Ermöglicht das Erstellen von Dictionaries, bei denen die Schlüssel schwache Verweise sind.

# third-party imports
import numpy as np  # Imports the numpy library for array and matrix operations.  # Importiert die Numpy-Bibliothek für Array- und Matrixoperationen.
import pandas as pd  # Imports pandas for data manipulation and analysis.  # Importiert Pandas für die Datenmanipulation und -analyse.
import logging  # Provides the logging module for tracking events in applications.  # Bietet das Logging-Modul zur Nachverfolgung von Ereignissen in Anwendungen.
import torch  # Imports PyTorch for machine learning and tensor computation.  # Importiert PyTorch für maschinelles Lernen und Tensorberechnungen.

T = TypeVar('T')  # helps with type inference in some editors  # Hilft mit der Typinferenz in einigen Editoren.

def pandas_dict(*args, **kwargs) -> pd.Series:  # Defines a function that returns a pandas Series from arguments.  # Definiert eine Funktion, die eine Pandas Series aus Argumenten zurückgibt.
    return pd.Series(dict(*args, **kwargs), dtype=object)  # Converts the arguments into a dictionary and then into a pandas Series.  # Wandelt die Argumente in ein Dictionary und dann in eine Pandas Series um.

def shallow_copy(obj: T) -> T:  # Defines a function to make a shallow copy of an object.  # Definiert eine Funktion zum Erstellen einer flachen Kopie eines Objekts.
    x = type(obj).__new__(type(obj))  # Creates a new instance of the object's type.  # Erstellt eine neue Instanz des Typs des Objekts.
    vars(x).update(vars(obj))  # Copies the attributes of the object into the new instance.  # Kopiert die Attribute des Objekts in die neue Instanz.
    return x  # Returns the shallow copy.  # Gibt die flache Kopie zurück.

# === collate, partition, etc ==========================================================================================
def collate_torch(batch, device=None):  # Defines a function that batches data into torch tensors.  # Definiert eine Funktion, die Daten in Torch-Tensoren zusammenfasst.
    """Turns a batch of nested structures with numpy arrays as leaves into into a single element of the same nested structure with batched torch tensors as leaves"""  # Function docstring explaining the purpose of the function.  # Funktionsdokumentation, die den Zweck der Funktion erklärt.
    elem = batch[0]  # Get the first element in the batch to inspect its type.  # Holt das erste Element der Charge, um den Typ zu überprüfen.
    if isinstance(elem, torch.Tensor):  # If the element is a torch tensor.  # Wenn das Element ein Torch-Tensor ist.
        if elem.numel() < 20000:  # If the tensor has fewer than 20000 elements.  # Wenn der Tensor weniger als 20000 Elemente hat.
            return torch.stack(batch).to(device)  # Stack the tensors in the batch and move them to the specified device.  # Stapelt die Tensoren in der Charge und verschiebt sie zum angegebenen Gerät.
        else:  # If the tensor has more than 20000 elements.  # Wenn der Tensor mehr als 20000 Elemente hat.
            return torch.stack([b.contiguous().to(device) for b in batch], 0)  # Stack tensors in batch with contiguous memory.  # Stapelt die Tensoren in der Charge mit zusammenhängendem Speicher.
    elif isinstance(elem, np.ndarray):  # If the element is a numpy array.  # Wenn das Element ein Numpy-Array ist.
        return collate_torch(tuple(torch.from_numpy(b) for b in batch), device)  # Recursively convert numpy arrays to torch tensors.  # Rekursiv Numpy-Arrays in Torch-Tensoren umwandeln.
    elif hasattr(elem, '__torch_tensor__'):  # If the element has a '__torch_tensor__' method.  # Wenn das Element eine Methode '__torch_tensor__' hat.
        return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)  # Call the method to convert to a tensor and stack them.  # Ruft die Methode auf, um in einen Tensor umzuwandeln und stapelt sie.
    elif isinstance(elem, Sequence):  # If the element is a sequence (list or tuple).  # Wenn das Element eine Sequenz (Liste oder Tupel) ist.
        transposed = zip(*batch)  # Transposes the batch to iterate over inner sequences.  # Transponiert die Charge, um über innere Sequenzen zu iterieren.
        return type(elem)(collate_torch(samples, device) for samples in transposed)  # Recursively process each inner sequence.  # Rekursiv jede innere Sequenz verarbeiten.
    elif isinstance(elem, Mapping):  # If the element is a dictionary-like mapping.  # Wenn das Element eine dictionary-ähnliche Zuordnung ist.
        return type(elem)((key, collate_torch(tuple(d[key] for d in batch), device)) for key in elem)  # Process each key-value pair in the mapping.  # Verarbeitet jedes Schlüssel-Wert-Paar in der Zuordnung.
    else:  # For other types of elements.  # Für andere Typen von Elementen.
        return torch.from_numpy(np.array(batch)).to(device)  # Convert the batch to a numpy array and then to a tensor.  # Wandelt die Charge in ein Numpy-Array und dann in einen Tensor um.

# === catched property =================================================================================================
class cached_property:  # Defines a class to create cached properties for objects.  # Definiert eine Klasse, um zwischengespeicherte Eigenschaften für Objekte zu erstellen.
    """Similar to `property` but after calling the getter/init function the result is cached.
    It can be used to create object attributes that aren't stored in the object's __dict__.
    This is useful if we want to exclude certain attributes from being pickled."""  # Docstring explaining the purpose of the class.  # Dokumentationsstring, der den Zweck der Klasse erklärt.
    def __init__(self, init=None):  # Initializes the cached property with an optional init function.  # Initialisiert die zwischengespeicherte Eigenschaft mit einer optionalen Init-Funktion.
        self.cache = {}  # Initializes an empty cache dictionary.  # Initialisiert ein leeres Cache-Wörterbuch.
        self.init = init  # Sets the init function for calculating the value.  # Setzt die Init-Funktion zur Berechnung des Werts.

    def __get__(self, instance, owner):  # Retrieves the cached value or computes it if not cached.  # Ruft den zwischengespeicherten Wert ab oder berechnet ihn, wenn er nicht zwischengespeichert ist.
        if id(instance) not in self.cache:  # Checks if the value is cached for this instance.  # Überprüft, ob der Wert für diese Instanz zwischengespeichert ist.
            if self.init is None: raise AttributeError()  # Raises an error if there is no init function provided.  # Wirft einen Fehler, wenn keine Init-Funktion bereitgestellt wird.
            self.__set__(instance, self.init(instance))  # Computes and stores the value using the init function.  # Berechnet und speichert den Wert unter Verwendung der Init-Funktion.
        return self.cache[id(instance)][0]  # Returns the cached value.  # Gibt den zwischengespeicherten Wert zurück.

    def __set__(self, instance, value):  # Sets the value in the cache and handles instance cleanup.  # Setzt den Wert im Cache und verwaltet die Instanzbereinigung.
        self.cache[id(instance)] = (value, weakref.ref(instance, functools.partial(self.cache.pop, id(instance))))  # Caches the value with a weak reference to the instance.  # Cacht den Wert mit einem schwachen Verweis auf die Instanz.

# === partial ==========================================================================================================
def default():  # Defines a dummy function to be used as a placeholder.  # Definiert eine Dummy-Funktion als Platzhalter.
    raise ValueError("This is a dummy function and not meant to be called.")  # Raises an error if the function is called.  # Wirft einen Fehler, wenn die Funktion aufgerufen wird.

def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:  # Defines a custom partial function implementation.  # Definiert eine benutzerdefinierte Partial-Funktion.
    """Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
     Then, the outer `partial` will insert the appropriate default value as the function. """  # Explains that the function acts like functools.partial but with additional behavior for keyword arguments.  # Erklärt, dass die Funktion wie functools.partial funktioniert, aber mit zusätzlichem Verhalten für Schlüsselwortargumente.
    if func is not default:  # Checks if a valid function was provided.  # Überprüft, ob eine gültige Funktion bereitgestellt wurde.
        for k, v in kwargs.items():  # Loops through the keyword arguments.  # Schleift durch die Schlüsselwortargumente.
            if isinstance(v, functools.partial) and v.func is default:  # Checks if the value is a dummy partial function.  # Überprüft, ob der Wert eine Dummy-Partial-Funktion ist.
                kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)  # Replaces the dummy function with the appropriate default value.  # Ersetzt die Dummy-Funktion durch den entsprechenden Standardwert.
    return functools.partial(func, *args, **kwargs)  # Returns the partial function with the specified arguments.  # Gibt die Partial-Funktion mit den angegebenen Argumenten zurück.

FKEY = '+'  # A constant value, likely used later in the code.  # Ein konstanter Wert, der wahrscheinlich später im Code verwendet wird.



def partial_to_dict(p: functools.partial, version="3"):  # Converts a partial function to a dictionary. / Wandelt eine teilweise Funktion in ein Wörterbuch um.
    """
    Only for wandb.  # Nur für wandb.
    This function has become lenient to work with Gymnasium.  # Diese Funktion wurde so angepasst, dass sie mit Gymnasium funktioniert.
    """
    assert not p.args, "So far only keyword arguments are supported, here"  # Asserts that no positional arguments are passed. / Stellt sicher, dass keine Positionsargumente übergeben werden.
    fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}  # Gets the default parameter values of the function. / Holt die Standardparameterwerte der Funktion.
    fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}  # Filters out empty parameters. / Filtert leere Parameter heraus.
    # diff = p.keywords.keys() - fields.keys()  # Difference between the keywords and function parameters. / Unterschied zwischen Schlüsselwörtern und Funktionsparametern.
    # assert not diff, f"{p} cannot be converted to dict. There are invalid keywords present: {diff}"  # Ensures no invalid keywords exist. / Stellt sicher, dass keine ungültigen Schlüsselwörter existieren.
    fields.update(p.keywords)  # Updates fields with the keyword arguments of the partial function. / Aktualisiert die Felder mit den Schlüsselwörtern der partiellen Funktion.
    nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}  # Recursively converts callable fields. / Wandelt rekursiv aufrufbare Felder um.
    simple = {k: v for k, v in fields.items() if k not in nested}  # Separates non-callable fields. / Trennt nicht-aufrufbare Felder.
    output = {FKEY: p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}  # Combines the module, function name with the simple and nested fields. / Kombiniert das Modul, den Funktionsnamen mit den einfachen und verschachtelten Feldern.
    return dict(output, __format_version__=version) if version else output  # Returns the dictionary with an optional version. / Gibt das Wörterbuch mit einer optionalen Version zurück.

# def partial_from_dict(d: dict):  # Function to convert a dictionary back to a partial function (commented out). / Funktion, um ein Wörterbuch wieder in eine partielle Funktion umzuwandeln (auskommentiert).
#     d = d.copy()  # Creates a copy of the dictionary. / Erstellt eine Kopie des Wörterbuchs.
#     assert d.pop("__format_version__", "3") == "3"  # Verifies the version is correct. / Überprüft, ob die Version korrekt ist.
#     d = {k: partial_from_dict(v) if isinstance(v, dict) and FKEY in v else v for k, v in d.items()}  # Recursively converts nested dictionaries back to partial functions. / Wandelt verschachtelte Wörterbücher rekursiv wieder in partielle Funktionen um.
#     func = get_class_or_function(d.pop(FKEY) or "tmrl.util:default")  # Gets the function referenced by the key. / Holt die Funktion, die durch den Schlüssel referenziert wird.
#     return partial(func, **d)  # Returns a partial function created with the given arguments. / Gibt eine partielle Funktion zurück, die mit den gegebenen Argumenten erstellt wird.

def get_class_or_function(func):  # Retrieves a class or function based on its name. / Holt eine Klasse oder Funktion basierend auf ihrem Namen.
    module, name = func.split(":")  # Splits the input into module and function name. / Teilt die Eingabe in Modul- und Funktionsnamen auf.
    return getattr(import_module(module), name)  # Imports the module and gets the function. / Importiert das Modul und holt die Funktion.

def partial_from_args(func: Union[str, callable], kwargs: Dict[str, str]):  # Converts arguments to a partial function. / Wandelt Argumente in eine partielle Funktion um.
    # logging.info(func, kwargs)  # useful to visualize the parsing process / nützlich, um den Parsing-Prozess zu visualisieren.
    func = get_class_or_function(func) if isinstance(func, str) else func  # Converts the function string to a callable. / Wandelt den Funktionsstring in ein Aufrufbares um.
    keys = {k.split('.')[0] for k in kwargs}  # Extracts the unique keys from the arguments. / Extrahiert die einzigartigen Schlüssel aus den Argumenten.
    keywords = {}  # Initializes an empty dictionary for the keywords. / Initialisiert ein leeres Wörterbuch für die Schlüsselwörter.
    for key in keys:  # Iterates over the extracted keys. / Iteriert über die extrahierten Schlüssel.
        params = inspect.signature(func).parameters  # Gets the parameters of the function. / Holt die Parameter der Funktion.
        assert key in params, f"'{key}' is not a valid parameter of {func}. Valid parameters are {tuple(params.keys())}."  # Ensures the key is a valid parameter. / Stellt sicher, dass der Schlüssel ein gültiger Parameter ist.
        param = params[key]  # Gets the parameter details. / Holt die Parameterdetails.
        value = kwargs.get(key, param.default)  # Gets the value for the parameter or the default if not provided. / Holt den Wert für den Parameter oder den Standardwert, wenn er nicht angegeben wird.
        if param.annotation is type:  # Checks if the parameter annotation is a type. / Überprüft, ob die Parameterannotation ein Typ ist.
            sub_keywords = {k.split('.', 1)[1]: v for k, v in kwargs.items() if k.startswith(key + '.')}  # Handles nested sub-parameters. / Behandelt verschachtelte Unterparameter.
            keywords[key] = partial_from_args(value, sub_keywords)  # Recursively processes nested parameters. / Verarbeitet rekursiv verschachtelte Parameter.
        elif param.annotation is bool:  # Checks if the parameter is a boolean. / Überprüft, ob der Parameter ein Boolean ist.
            keywords[key] = bool(eval(value))  # Converts the string to a boolean. / Wandelt den String in einen Boolean um.
        else:
            keywords[key] = param.annotation(value)  # Converts the value to the annotated type. / Wandelt den Wert in den annotierten Typ um.
    return partial(func, **keywords)  # Returns the partial function with the updated keywords. / Gibt die partielle Funktion mit den aktualisierten Schlüsselwörtern zurück.

# === git ==============================================================================================================

def get_output(*args, default='', **kwargs):  # Executes a command and returns its output. / Führt einen Befehl aus und gibt seine Ausgabe zurück.
    try:
        output = subprocess.check_output(*args, universal_newlines=True, **kwargs)  # Runs the command and gets the output. / Führt den Befehl aus und erhält die Ausgabe.
        return output.rstrip("\n")  # Skips trailing newlines, similar to bash. / Überspringt nachgestellte neue Zeilen, wie in bash.
    except subprocess.CalledProcessError:  # Catches errors if the command fails. / Fängt Fehler ab, wenn der Befehl fehlschlägt.
        return default  # Returns the default value if there's an error. / Gibt den Standardwert zurück, wenn ein Fehler auftritt.



def git_info(path=None):  # Defines a function git_info that accepts an optional path argument.  # Definiert eine Funktion git_info, die ein optionales Pfad-Argument akzeptiert.
    """returns a dict with information about the git repo at path (path can be a sub-directory of the git repo)  # Gibt ein Dictionary mit Informationen über das Git-Repository zurück (path kann ein Unterverzeichnis des Repositories sein)."""
    import __main__  # Importing the main module.  # Importiert das Hauptmodul.
    path = path or os.path.dirname(__main__.__file__)  # If no path is provided, use the current script's directory as the path.  # Wenn kein Pfad angegeben, wird der Pfad des aktuellen Scripts verwendet.
    rev = get_output('git rev-parse HEAD'.split(), cwd=path)  # Executes the git command to get the current commit hash.  # Führt den Git-Befehl aus, um den aktuellen Commit-Hash zu erhalten.
    count = int(get_output('git rev-list HEAD --count'.split(), default='-1', cwd=path))  # Executes the git command to count the number of commits in the current branch.  # Führt den Git-Befehl aus, um die Anzahl der Commits im aktuellen Branch zu zählen.
    status = get_output('git status --short'.split(), cwd=path)  # Executes the git command to get the status of the repository, showing uncommitted changes.  # Führt den Git-Befehl aus, um den Status des Repositories zu erhalten, der nicht committete Änderungen zeigt.
    commit_date = get_output("git show --quiet --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd".split(), cwd=path, env=dict(TZ='UTC'))  # Executes the git command to get the date of the last commit in UTC format.  # Führt den Git-Befehl aus, um das Datum des letzten Commits im UTC-Format zu erhalten.
    desc = get_output(['git', 'describe', '--long', '--tags', '--dirty', '--always', '--match', r'v[0-9]*\.[0-9]*'], cwd=path)  # Executes the git command to describe the current git state, including version tags.  # Führt den Git-Befehl aus, um den aktuellen Git-Status zu beschreiben, einschließlich Versions-Tags.
    message = desc + " " + ' '.join(get_output(['git', 'log', '--oneline', '--format=%B', '-n', '1', "HEAD"], cwd=path).splitlines())  # Combines the description with the latest commit message.  # Kombiniert die Beschreibung mit der neuesten Commit-Nachricht.
    url = get_output('git config --get remote.origin.url'.split(), cwd=path).strip()  # Retrieves the URL of the remote repository.  # Holt sich die URL des Remote-Repositories.
    if url.startswith('git@github.com:'):  # If the URL starts with git@github.com,  # Wenn die URL mit git@github.com beginnt.
        url = 'https://github.com/' + url[len('git@github.com:'):-len('.git')] + '/commit/' + rev  # Convert it to a HTTPS URL for the commit.  # Wandelt sie in eine https-URL für den Commit um.
    elif url.startswith('https://github.com'):  # If the URL starts with https://github.com,  # Wenn die URL mit https://github.com beginnt.
        url = url[:len('.git')] + '/commit/' + rev  # Modify the URL for the commit.  # Ändert die URL entsprechend.
    return dict(url=url, rev=rev, count=count, status=status, desc=desc, date=commit_date, message=message)  # Returns a dictionary with all the gathered git information.  # Gibt ein Dictionary mit allen gesammelten Git-Informationen zurück.

# === serialization ====================================================================================================

def dump(obj, path):  # Defines a function dump to save an object to a file.  # Definiert eine Funktion dump, um ein Objekt in eine Datei zu speichern.
    path = Path(path)  # Converts the path argument to a Path object for better file handling.  # Wandelt den Pfad in ein Path-Objekt um.
    tmp_path = path.with_suffix('.tmp')  # Creates a temporary path with a .tmp suffix.  # Erzeugt einen temporären Pfad mit der Endung .tmp.
    with DelayInterrupt():  # Ensures that the save operation continues even if interrupted.  # Stellt sicher, dass das Speichern auch bei Unterbrechung fortgesetzt wird.
        with open(tmp_path, 'wb') as f:  # Opens the temporary file in binary write mode.  # Öffnet die temporäre Datei im Binär-Schreibmodus.
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  # Serializes and saves the object using pickle with the highest protocol.  # Serialisiert und speichert das Objekt mit pickle im höchsten Protokoll.
        os.replace(tmp_path, path)  # Replaces the original file with the temporary file.  # Ersetzt die Originaldatei durch die temporäre Datei.

def load(path):  # Defines a function to load a pickled object from a file.  # Definiert eine Funktion, um ein Pickle-Objekt aus einer Datei zu laden.
    with open(path, 'rb') as f:  # Opens the file in binary read mode.  # Öffnet die Datei im Binär-Lese-Modus.
        return pickle.load(f)  # Loads and returns the pickled object.  # Lädt und gibt das Pickle-Objekt zurück.

def save_json(d, path):  # Defines a function to save a dictionary to a JSON file.  # Definiert eine Funktion, um ein Dictionary in einer JSON-Datei zu speichern.
    with open(path, 'w', encoding='utf-8') as f:  # Opens the file in write mode with UTF-8 encoding.  # Öffnet die Datei im Schreibmodus mit UTF-8-Codierung.
        json.dump(d, f, ensure_ascii=False, indent=2)  # Serializes and saves the dictionary to a JSON file with indentation.  # Serialisiert und speichert das Dictionary in einer JSON-Datei mit Einrückungen.

def load_json(path):  # Defines a function to load a dictionary from a JSON file.  # Definiert eine Funktion, um ein Dictionary aus einer JSON-Datei zu laden.
    with open(path, 'r', encoding='utf-8') as f:  # Opens the file in read mode with UTF-8 encoding.  # Öffnet die Datei im Lesemodus mit UTF-8-Codierung.
        return json.load(f)  # Loads and returns the JSON data as a dictionary.  # Lädt und gibt die JSON-Daten als Dictionary zurück.


# === signal handling ==================================================================================================

class DelayInterrupt:  # Define a class named DelayInterrupt.  # Definiert eine Klasse namens DelayInterrupt.
    """Catches SIGINT and SIGTERM and re-raises them after the context manager exits.
    
    Can be used in a context, e.g., `with DelayInterrupt():`  # Catches interrupt signals (SIGINT, SIGTERM) and re-raises them after the context manager is finished. # Fängt Unterbrechungssignale (SIGINT, SIGTERM) ab und wirft sie erneut, nachdem der Kontext-Manager beendet ist.

    """
    signal_received = False  # A class variable that tracks if a signal was received.  # Eine Klassenvariable, die verfolgt, ob ein Signal empfangen wurde.
    signals = (signal.SIGINT, signal.SIGTERM)  # A tuple containing the signals to catch (SIGINT and SIGTERM).  # Ein Tupel, das die zu fassenden Signale (SIGINT und SIGTERM) enthält.

    def __enter__(self):  # This method is called when entering the context manager (with statement).  # Diese Methode wird beim Betreten des Kontext-Managers (mit-Anweisung) aufgerufen.
        self.default_handlers = [signal.getsignal(s) for s in self.signals]  # Saves the default signal handlers for the signals.  # Speichert die Standard-Signalbehandler für die Signale.
        [signal.signal(s, self.on_signal) for s in self.signals]  # Sets a custom handler (on_signal) for the signals.  # Setzt einen benutzerdefinierten Behandler (on_signal) für die Signale.

    def on_signal(self, *args):  # This method is called when a signal is caught.  # Diese Methode wird aufgerufen, wenn ein Signal abgefangen wird.
        logging.info(f"tmrl.util:DelayInterrupt -- Signal received!", *args)  # Logs the signal reception.  # Protokolliert den Empfang des Signals.
        self.signal_received = True  # Sets the signal_received flag to True.  # Setzt das Signalempfangen-Flag auf True.

    def __exit__(self, *args):  # This method is called when exiting the context manager.  # Diese Methode wird beim Verlassen des Kontext-Managers aufgerufen.
        [signal.signal(s, d) for s, d in zip(self.signals, self.default_handlers)]  # Restores the default signal handlers.  # Stellt die Standard-Signalbehandler wieder her.
        if self.signal_received:  # Checks if a signal was received.  # Überprüft, ob ein Signal empfangen wurde.
            raise KeyboardInterrupt()  # Raises a KeyboardInterrupt exception if a signal was received.  # Löst eine KeyboardInterrupt-Ausnahme aus, wenn ein Signal empfangen wurde.

# === operations =======================================================================================================

def prod(iterable):  # Defines a function named prod that takes an iterable as input.  # Definiert eine Funktion namens prod, die ein Iterable als Eingabe erhält.
    return functools.reduce(operator.mul, iterable, 1)  # Multiplies all elements in the iterable using functools.reduce, starting with an initial value of 1.  # Multipliziert alle Elemente im Iterable mit functools.reduce, beginnend mit einem Anfangswert von 1.
