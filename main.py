import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
# import dlib
from math import hypot

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 680)
segmentor = SelfiSegmentation()

# writer image in file
list_bg_img = os.listdir("Image/background/")
list_fil_img = os.listdir("Image/filter/")
# print(list_img)

# list image in file and resize it
bg_img_list = []
for img_path in list_bg_img:
    img = cv2.imread(f"Image/background/{img_path}")
    resizeimg = cv2.resize(img, (640, 480))
    bg_img_list.append(resizeimg)
# print(len(bg_img_list))

fil_img_list = []
for img_path in list_fil_img:
    img = cv2.imread(f"Image/filter/{img_path}")
    fil_img_list.append(img)
# print(len(fil_img_list))
# Creating mask
_, frame = cap.read()
rows, cols, _ = frame.shape
mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img_index = 0

while True:
    success, img = cap.read()
    img_out = segmentor.removeBG(img, bg_img_list[img_index], threshold=0.8)
    # cv2.imshow("Image", img_out)
    img_stacked = cvzone.stackImages([img_out], 1, 1)
    # print(img_index)

    cv2.imshow("Image", img_stacked)

    key = cv2.waitKey(1)
    if key == ord("a"):
        if img_index > 0:
            img_index -= 1
    elif key == ord("d"):
        if img_index < len(bg_img_list)-1:
            img_index += 1
    elif key == ord("q"):
        break
