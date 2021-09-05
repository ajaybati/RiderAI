import pyautogui as py
import sys
import cv2
from time import sleep
print('Press Ctrl-C to quit.')

def pick_point():
    locs=[]
    for x in ["top left","bottom right"]:
        timer=2
        print("In " +str(timer) + " seconds please place your mouse at the "+ x + " of the screen.")
        for y in range(timer,0,-1):
            print(" "+str(y)+" ",end='')
            # print(y,end='')
            print('\b'*4, end='', flush=True)

            sleep(1)

        locs.append(py.position())
        print(locs[-1])
    return locs

def l():
    try:
        while True:

            x, y = py.position()


            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
            sleep(1)
            timecount+=1
    except KeyboardInterrupt:
        print('\n')
