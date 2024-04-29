import psycopg2

# Établir une connexion à la base de données
conn = psycopg2.connect(
    dbname="ENSAJdb",
    user="postgres",
    password="AMINA123",
    host="localhost",
    port="5432"
)

# Créer un curseur pour exécuter des requêtes SQL
cur = conn.cursor()

# Appeler la fonction verifier_utilisateur
cur.execute("SELECT verifier_utilisateur(%s, %s)", ('CIN', 'mot_de_passe_hash'))

# Récupérer le résultat de la fonction
resultat = cur.fetchone()[0]

# Fermer le curseur et la connexion à la base de données
cur.close()
conn.close()

# Utilisez le résultat pour décider si l'utilisateur est authentifié
if resultat:
    print("L'utilisateur est authentifié.")
else:
    print("L'utilisateur n'est pas authentifié.")
