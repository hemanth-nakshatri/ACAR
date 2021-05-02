print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from tensorflow.keras.models import load_model

from PIL import Image
import cv2





def preProcess(img):
    img = img[60:135, :, :]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

model = load_model('Car_Simulation\models\model_phase_1.2.h5')

def run(img_path):
    image = Image.open(img_path)
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    print(steering)
    return steering



for i in range(1,8):
    img_path = 'Car_Simulation\\test{}.jpg'.format(i)
    run(img_path)
