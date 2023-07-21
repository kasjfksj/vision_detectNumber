import cv2
import Handtrack_module as htm
import numpy as np
import torch
import os
import time

import sys
sys.path.insert(1, '/Users/alexanderlee/computer-vision/model')
from model import Net
net = Net()
net.load_state_dict(torch.load("/Users/alexanderlee/computer-vision/model_data/model_parameter.pkl"))

folderPath = "header"
myList = os.listdir(folderPath)
video = cv2.VideoCapture(0)
overlayList = []
drawColor = (83,78,207)
Thinkness=15

xp=0
yp=0
imgCanvas = np.zeros((720,1280,3),np.uint8)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    cv2.imshow("Img",image)
    overlayList.append(image)
header = overlayList[0]

detector = htm.handdetector(detectionCon=0.85)


video.set(3,1280)
video.set(4,720)
t =0 
fingers = []
flag = True
while True:
    success, img = video.read()
    img = cv2.flip(img,1)

    img = detector.findHands(img)
    lmlist = detector.findposition(img,draw =False)
    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            if t>8:
                xp,yp=0,0

                t=0
            if y1<137:
                if 580<x1<720:
                    drawColor = (83,78,207)
                    Thinkness =15
                    flag=True
                elif 1020<x1<1190:
                    drawColor = (0,0,0)
                    Thinkness=60
                elif 100<x1<270 and flag:
                    canvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                    canvas = cv2.resize(canvas,(128,128),interpolation=cv2.INTER_AREA)
                    canvasReshape = np.reshape(canvas,(-1,1,128,128))

                    # cv2.imshow("afdjd",canvasReshape)
                    # cv2.waitKey(100)
                    canvas_tensor = torch.tensor(canvasReshape).float()
                    outputs = net(canvas_tensor)
                    prediction = torch.argmax(outputs,dim=1)
                    print(prediction)
                    imgCanvas = np.zeros((720,1280,3),np.uint8)
                    drawColor = (0,0,0)
                    flag=False
            cv2.rectangle(img,(x1-15,y1-15),(x2+15,y2+15),drawColor,cv2.FILLED)


        if fingers[1] and fingers[2] == False:
            
            if xp==0 and yp==0:
                xp=x1
                yp=y1
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            cv2.line(img,(xp,yp),(x1,y1),drawColor,Thinkness)
            cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,Thinkness)
            xp,yp=x1,y1


    img[0:173,0:1280] = header
    cv2.imshow("Img",img)
    cv2.imshow("ImgCanvas",imgCanvas)

    if t<=10:
        t+=1
    cv2.waitKey(1)