import cv2
import numpy as np
import os

def mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    contours = sorted(contours, key=cv2.contourArea)
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
           break
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask, [contour],-1, 255, -1)
    mask = cv2.bitwise_and(img, img, mask = mask)
    result = mask.copy()
    result[threshold == 0] = (0, 0, 0)
    rename = "_MASK".join(os.path.splitext(data))
    cv2.imwrite(os.path.join('results/mask', rename), result)

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    th, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (y, x) = np.where(threshold == 255)
    (t_y, t_x) = (np.min(y), np.min(x))
    (b_y, b_x) = (np.max(y), np.max(x))
    cut = mask[(t_y-5):(b_y+5), (t_x-5):(b_x+5)]
    rename = "_CUT".join(os.path.splitext(data))
    cv2.imwrite(os.path.join('results/cut', rename), cut)

def resize(img, x, y):
    resized = cv2.resize(img, (0, 0), fx = x, fy = y)
    rename = "_RESIZE".join(os.path.splitext(data))
    cv2.imwrite(os.path.join('results/resize', rename), resized)

for data in os.listdir('test/'):
    img = cv2.imread(os.path.join('test/', data))
    mask(img)

for data in os.listdir('results/cut/'):
    img = cv2.imread(os.path.join('results/cut/', data))
    resize(img, 0.5, 0.5)



