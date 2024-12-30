# adapted from https://stackoverflow.com/questions/10622179/how-to-find-identify-large-commits-in-git-history  # Reference to the original source of the script.  # Referenz auf die ursprüngliche Quelle des Skripts.
# this bash script displays sorted commit sizes in a human-readable fashion  # This script shows the sizes of Git objects in a readable format.  # Dieses Skript zeigt die Größen von Git-Objekten in einem lesbaren Format an.
git rev-list --objects --all |  # Lists all objects in the Git repository's history.  # Listet alle Objekte in der Git-Historie des Repositories auf.
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |  # Retrieves details about each object, such as type, name, and size.  # Ruft Details zu jedem Objekt ab, wie Typ, Name und Größe.
  sed -n 's/^blob //p' |  # Filters only 'blob' objects (file content) from the list.  # Filtert nur 'blob'-Objekte (Dateiinhalte) aus der Liste.
  sort --numeric-sort --key=2 |  # Sorts the objects numerically by their size.  # Sortiert die Objekte numerisch nach ihrer Größe.
  # cut -c 1-12,41- |  # (Optional) Cuts parts of the output for formatting (commented out).  # (Optional) Schneidet Teile der Ausgabe zur Formatierung (auskommentiert).
  $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest  # Converts sizes to a human-readable format (e.g., KB, MB).  # Konvertiert Größen in ein lesbares Format (z. B. KB, MB).
