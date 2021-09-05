import sys
import cv2
from location import pick_point
import math
import numpy as np
from time import sleep
import mss
from PIL import Image
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

#get angle of rotation using the homography matrix (basic algebra search it up)
def get_angle(M):
    sx = np.linalg.norm(M[0,:2])
    sy = np.linalg.norm(M[1,:2])
    sub = M[:2,:2]

    sub[0] /= sx
    sub[1] /= sy

    return round(180*math.acos(sub[0,0])/math.pi,3)
loc1,loc2 = pick_point()

x=0

sift = cv2.xfeatures2d.SIFT_create()
width=loc2[0]-loc1[0]
height=loc2[1]-loc1[1]
mon = {'top': loc1[1], 'left': loc1[0], 'width': width, 'height': height}
print(mon)
tem = cv2.imread("/Users/ajaybati/Documents/riderAi/res/cartype21.png",cv2.IMREAD_GRAYSCALE)
print(tem.shape,"shape")
w,h = tem.shape[::-1]
kp1, des1 = sift.detectAndCompute(tem,None)
sct = mss.mss()
while True:
    start = time.time()
    img_scrape = np.array(sct.grab(mon))
    img_bw = cv2.cvtColor(img_scrape, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img_bw,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply Lowe's ratio test - best sample must be less than a certain ratio of distance away
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
    #find source points from match
    src = np.float32([kp1[g.queryIdx].pt for g in good]).reshape(-1,1,2)

    #find resultant points from match
    end = np.float32([kp2[g.trainIdx].pt for g in good]).reshape(-1,1,2)


    M, mask = cv2.findHomography(src, end, cv2.RANSAC,5.0)

    matchesMask = mask.ravel().tolist()
    pts = np.float32([ [0,0]]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts, M)
    dst = tuple(map(int,tuple(np.squeeze(dst,axis=1)[0])))

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    # img3 = cv2.drawMatches(tem,kp1,img_bw,kp2,good,None,**draw_params)
    RED =(0,0,255)
    # img3 = cv2.polylines(img_scrape, [np.int32(dst)], True, RED,3, cv2.LINE_AA)
    img3 = cv2.circle(img_scrape, dst, 10, RED, -1)
    img3= cv2.circle(img3, (int(width),int(height)),10,RED,-1)
    #
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = dst
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    img3 = cv2.putText(img3, 'Angle: '+ str(get_angle(M)), org, font,
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Boxing", img3)
    # if get_angle(M)>160:
    #     print(get_angle(M))
    #     cv2.imwrite("/Users/ajaybati/Documents/riderAi/key_angles/"+str(get_angle(M))+".png", img3)
    print(" "+str(round(1/(time.time()-start)))+" ",end='')
    print("\b"*10,end='',flush=True)
    # print(dst)
    # #
    # og_x=dst[1]-mon["left"]
    # og_y=dst[0]-mon["top"]
    middle_x=width
    middle_y=height
    mon["top"] = mon["top"]+(dst[0]-middle_y)
    mon["left"] = mon["left"]+(dst[1]-middle_x)
    print(dst[1]-middle_y)
    # print((mon))
    # print(dst, (middle_x,middle_y))
    # mon["left"] = (mon["left"]+dst[0]) - width/2
    # if dst[1] <= 50:
    #     mon["top"]+=100
    # print(mon)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    x+=1
