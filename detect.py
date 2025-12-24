import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")
IMG_SIZE = 224

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = "ACCIDENT 🚨" if pred > 0.5 else "NORMAL"
    color = (0,0,255) if pred > 0.5 else (0,255,0)

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
