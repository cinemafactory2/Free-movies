import cv2,time
import numpy  as np

video = cv2.VideoCapture('E:\\DoNotTouch\\Python\\projects\\Motion Detector\\files\\video1.mp4')
#video = cv2.VideoCapture(0)
_, first_frame = video.read()
first_frame = cv2.resize(first_frame, (first_frame.shape[1]//2,first_frame.shape[0]//2))
_, frame2 = video.read()
frame2 = cv2.resize(frame2, (frame2.shape[1]//2,frame2.shape[0]//2))

while True:
    delta_frame = cv2.absdiff(first_frame,frame2)
    gray = cv2.cvtColor(delta_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(21,21),0)
    thresh_delta = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)[1]
    #thresh_delta = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,-9)
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=3)
    cnts, _ = cv2.findContours(thresh_delta, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 1200:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x,y), (x+w,y+h), (0,255,0), 3)
    
    #cv2.imshow('Capturing', gray)
    #cv2.imshow('delta', delta_frame)
    #cv2.imshow('thresh', thresh_delta)
    cv2.imshow('frame',frame2)
    #first_frame = frame2
    _, frame2 = video.read()
    frame2 = cv2.resize(frame2, (frame2.shape[1]//2,frame2.shape[0]//2))
    key = cv2.waitKey(2)
    if key == ord('q'):
        break

#video.release() if using webcame
cv2.destroyAllWindows()