import psycopg2
import cv2
import numpy as np
import face_recognition

# Fonction pour convertir les données binaires en image
def convert_binary_to_image(binary_data):
    nparr = np.frombuffer(binary_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    dbname="ENSAJdb",
    user="postgres",
    password="AMINA123",
    host="localhost"
)

# Récupération des données des étudiants depuis la base de données
cur = conn.cursor()
cur.execute("SELECT id, nom, prénom, photo FROM public.\"Etudiant\";")
rows = cur.fetchall()

# Charger les visages connus à partir des données de la base de données
known_face_encodings = []
known_face_names = []
for row in rows:
    id_etudiant, nom, prenom, image_data = row
    image = convert_binary_to_image(image_data)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append((nom, prenom))  

# Configuration de la capture vidéo à partir de la webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture d'une image depuis la webcam
    ret, frame = video_capture.read()

    # Détection des visages dans l'image capturée
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Comparaison avec les visages connus
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Inconnu"

        if True in matches:
            match_index = matches.index(True)
            name = f"{known_face_names[match_index][1]} {known_face_names[match_index][0]}"  # Nom complet de l'étudiant

        face_names.append(name)

    # Affichage du résultat sur l'image
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Affichage de l'image avec les résultats
    cv2.imshow("Video", frame)

    # Quitter la boucle lorsque la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources et fermeture de la connexion
video_capture.release()
cv2.destroyAllWindows()
conn.close()
