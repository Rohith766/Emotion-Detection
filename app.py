from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Load the Haarcascade frontal face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion recognition model
model = load_model('model.h5')
label_map = ['Anger', 'Sad', 'Fear', 'Happy', 'Neutral', 'Disgust', 'Surprise']

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def predict_emotion(img):
    # Preprocess the image
    resized_image = cv2.resize(img, (48, 48))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(grayscale_image) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    # Perform prediction
    pred = model.predict(img_array)
    pred_label = label_map[np.argmax(pred)]

    return pred_label

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
