import csv

# Lire le fichier tabulé
with open("parameters.csv", newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile, delimiter='\t')  # tab-separated
    rows = list(reader)

# Réécrire avec des virgules
with open("parameters_clean.csv", "w", newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    writer.writerows(rows)
