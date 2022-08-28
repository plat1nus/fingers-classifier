import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from utils import decode_predict


model = load_model("classifier.h5")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    scs, image = cap.read()
    image_nn = cv2.resize(image, (224, 224))
    pred = model.predict(image_nn.reshape(1, image_nn.shape[0], image_nn.shape[1], image_nn.shape[2], -1), verbose=0)
    print(decode_predict(pred))

    cv2.imshow("plat1nus", image)
    cv2.waitKey(1)
