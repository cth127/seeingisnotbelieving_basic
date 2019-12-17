import numpy as np
import cv2
import os

for i in range(1, 11) :
    os.chdir('C:/Users/taehun/Desktop/CHIC/light/prepro/ori')
    bgr = cv2.imread('level%s.jpg' %(i))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    os.chdir('C:/Users/taehun/Desktop/CHIC/light/prepro/cla')
    cv2.imwrite("level%s_cla.jpg" %(i), res)