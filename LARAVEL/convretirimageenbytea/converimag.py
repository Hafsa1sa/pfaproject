import psycopg2
from psycopg2 import sql

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    dbname="ENSAJdb",
    user="postgres",
    password="AMINA123",
    host="localhost"
)

# Ouverture du fichier image en mode lecture binaire
with open('amy.jpg', 'rb') as f:
    image_bytes = psycopg2.Binary(f.read())

# Définition de la requête SQL pour insérer l'image dans la base de données
query = sql.SQL("INSERT INTO public.\"Etudiant\" (\"id\", \"CNE\", \"nom\", \"prénom\", \"date de naissance\", \"lieu de naissance\", \"CIN\", \"email\", \"GSM\", \"photo\", \"filière_nom\", \"niveau_id\") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")


# Exécution de la requête avec les données de l'image
cur = conn.cursor()
cur.execute(query, ('D134079891', 'iit2', 1, image_bytes))

# Commit des changements et fermeture de la connexion
conn.commit()

# Fermeture du curseur et de la connexion
cur.close()
conn.close()
