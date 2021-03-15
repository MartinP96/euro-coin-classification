'''
- script name: coin_detection_and_classification.py
- author: Martin P.
- description: This script is used for detection and classification of euro coins
in input image.
'''

import cv2
from math import sqrt
from math import floor
import tensorflow as tf
from tensorflow import keras
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def center_inside(gx, gy, gr, x, y):
    return sqrt((gx - x) ** 2 + (gy - y) ** 2) < gr


def is_inside(x, y, r, circles):
    for gx, gy, gr in circles:
        if center_inside(gx, gy, gr, x, y):
            return True
    return False


def equalize_gray_clahe(img, clipLimit=2.0, tileGridSize=(8, 8), show_clahed=False):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    img_clahed = clahe.apply(img)
    return img_clahed


def equalize_rgb_clahe(img, clipLimit=2.0, tileGridSize=(8, 8), show_clahed=False):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def crop_circle(img1, circles):
    index = 1
    ret = {}  # Store cropped images into dictionary
    for i in circles:
        # the values must be integers
        x = i[0]
        y = i[1]
        radius = i[2]
        radius = int(radius)

        # Create a mask:
        mask = np.zeros((img1.shape[0], img1.shape[1]), np.uint8)
        # Draw the circles on that mask (set thickness to -1 to fill the circle):
        circle_img = cv2.circle(mask, (x, y), radius, (255, 255, 255),
                                thickness=-1)
        # Copy that image using that mask:
        masked_data = cv2.bitwise_and(img1, img1, mask=circle_img)
        # Apply Threshold
        r, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # Find Contour
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        # Crop masked_data
        crop = masked_data[y:y + h, x:x + w]
        ret[index] = crop
        index += 1

    return ret


def find_coins(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=175, param2=35, minRadius=50, maxRadius=110)
    good_circles = []

    if not circles is None:
        # print(f"\tCircles before refining {circles.shape[1]}")
        for x, y, r in circles[0, :, :]:
            if not is_inside(x, y, r, good_circles):
                good_circles.append([x, y, r])

    return good_circles


def make_prediction(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    I = np.zeros((1, 100, 100, 3))
    I[0] = resized / 255.0
    p = model.predict(I)
    p = p[0]
    return p


# ---------DETECT COINS---------- #

# Load RGB image of coins
coins = cv2.imread('kovanci7.jpg')

# Convert image to gray
img = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 5)
img = equalize_gray_clahe(img)

# Find coins
circles = find_coins(img)
circles = np.uint16(np.around(circles))

# ---------CLASSIFY COINS---------- #

# Load model
model = keras.models.load_model('model')

# Dict of class labels
class_labels = {
    0: "1c",
    1: "1e",
    2: "2c",
    3: "2e",
    4: "5c",
    5: "10c",
    6: "20c",
    7: "50c"
}

# Crop coins from input images
coins_clahe = equalize_rgb_clahe(coins)  # Filter input image (in some cases classifications works better)
crop = crop_circle(coins_clahe, circles)

coin_predictions = []
for i in crop:
    p = make_prediction(crop[i])
    max_index = np.where(p == np.amax(p))
    max_index = max_index[0][0]
    print(f'Prediction: {class_labels[max_index]}: {p[max_index]}')
    coin_predictions.append((max_index, floor(p[max_index] * 100)))

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

# Draw circles on original image
for i in enumerate(circles):
    cv2.circle(coins, (i[1][0], i[1][1]), i[1][2], (0, 255, 0), 2)
    cv2.circle(coins, (i[1][0], i[1][1]), 2, (0, 0, 255), 3)

    current_prediction = coin_predictions[i[0]]

    bottomLeftCornerOfText = (i[1][0], i[1][1])
    cv2.putText(coins, f'{class_labels[current_prediction[0]]}; {current_prediction[1]}%',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

cv2.imshow('Detected coins', coins)
cv2.waitKey()
