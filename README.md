import cv2

# Menginisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Membaca gambar atau video
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mendeteksi wajah
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Menandai wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Menampilkan gambar dengan wajah yang terdeteksi
cv2.imshow('Deteksi Wajah', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import os

# Menginisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menginisialisasi recognizer wajah
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Membaca dataset pelatihan
def read_dataset(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=

5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_samples.append(gray[y:y+h, x:x+w])
            ids.append(1)  # Ganti dengan ID sesuai dengan identitas individu
    return face_samples, ids

# Melatih sistem pengenalan
dataset_path = 'dataset'
faces, ids = read_dataset(dataset_path)
recognizer.train(faces, np.array(ids))

# Menyimpan model pelatihan
recognizer.save('train_data.yml')
import cv2

# Menginisialisasi detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menginisialisasi recognizer wajah
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Mengimpor data pelatihan wajah
recognizer.read('train_data.yml')

# Membaca gambar atau video
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mendeteksi wajah
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Mengenali wajah
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    id_, confidence = recognizer.predict(roi_gray)
    if confidence <= 100:
        name = "Pengguna " + str(id_)
    else:
        name = "Tidak Dikenali"
    cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Menampilkan gambar dengan hasil pengenalan wajah
cv2.imshow('Pengenalan Wajah', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
