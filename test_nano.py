print('Setting UP')
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
print("np imported")

import cv2

print("cv2 imported")

from tensorflow.keras.models import load_model
print("tf imported")

from PIL import Image
print("PIL imported")

def preProcess(img):
    img = img[60:135, :, :]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def run(img_path):
    image = Image.open(img_path)
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    print(steering)
    return steering

print("Loading a model")
model = load_model('models/model_phase_1.2.h5')
print("Model Loaded")



start = time.process_time()
for i in range(1,91):
    print(i)
    img_path = 'test_images/test({}).jpg'.format(i)
    run(img_path)
print("Time taken is {}".format(time.process_time() - start))
