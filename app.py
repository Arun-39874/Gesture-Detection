import streamlit as st
import cv2
import mediapipe as mp_hands
import numpy as np
from keras.models import model_from_json
from function import *

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

actions = [...]
colors = [(245, 117, 16) for _ in range(20)]
threshold = 0.8

st.title("Gesture Recognition App")
output_text = st.empty()
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    sequence = []
    sentence = []
    accuracy = []
    predictions = []

    while st.checkbox("Start Video Stream"):
        ret, frame = cap.read()
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        image, results = mediapipe_detection(cropframe, hands)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if predicted_action != sentence[-1]:
                                sentence.append(predicted_action)
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(predicted_action)
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            pass

        output_text.text("Output: -" + ' '.join(sentence) + ''.join(accuracy))
        st.image(frame, channels="BGR", use_column_width=True, caption="Live Feed")
    cap.release()
cv2.destroyAllWindows()
