import pyautogui as py
import sys
import cv2
from location import pick_point
import numpy as np
from time import sleep
import mss
from PIL import Image
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

def empty(a):
    pass
loc1,loc2 = pick_point()
cv2.namedWindow("Thresholds")
cv2.resizeWindow("Thresholds",640,240)
cv2.createTrackbar("Threshold1","Thresholds",150,255,empty)
cv2.createTrackbar("Threshold2","Thresholds",255,255,empty)

mon = {'top': loc1[1], 'left': loc1[0], 'width': loc2[0]-loc1[0], 'height': loc2[1]-loc1[1]}
print(mon)
tem = cv2.imread("/Users/ajaybati/Documents/riderAi/res/car3.png",cv2.IMREAD_GRAYSCALE)
sct = mss.mss()
while True:
    start = time.time()
    img_scrape = np.array(sct.grab(mon))
    img_blur = cv2.GaussianBlur(img_scrape,(7,7),1)
    img_gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1","Thresholds")
    threshold2 = cv2.getTrackbarPos("Threshold2","Thresholds")



    img_detect = cv2.Canny(img_gray,threshold1,threshold2)


    cv2.imshow("Edges",img_detect)
    print("FPS: "+str(round(1/(time.time()-start)))+ " Threshold1: "+str(threshold1)+" Threshold2: "+str(threshold2)+ " ",end='')
    print("\b"*100,end='',flush=True)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
