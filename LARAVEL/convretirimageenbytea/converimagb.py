import os
import psycopg2
from psycopg2.extensions import Binary
from PIL import Image
from io import BytesIO

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    dbname="ENSAJdb",
    user="postgres",
    password="AMINA123",
    host="localhost"
)

# Répertoire de destination pour enregistrer les images
repertoire_destination = 'photosconvertis'

# Exécution de la requête SQL pour récupérer les données binaires de l'image avec leurs IDs
cur = conn.cursor()
cur.execute("SELECT id, photo FROM public.\"Etudiant\" ;")
resultats = cur.fetchall()

# Parcourir les résultats et enregistrer les images avec leurs IDs comme noms de fichier
for row in resultats:
    etudiant_id, image_data = row
    # Ouvrir l'image à partir des données binaires en spécifiant le format JPEG
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Nom du fichier avec l'ID de l'étudiant
    nom_image = f'{etudiant_id}.jpg'
    chemin_image = os.path.join(repertoire_destination, nom_image)
    image.save(chemin_image)  # Enregistrer l'image dans le répertoire de destination

# Fermer le curseur et la connexion
cur.close()
conn.close()
