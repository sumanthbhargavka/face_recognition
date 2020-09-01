import numpy as np
import serial
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("trainner\\trainner.yml")
id=0
c=0
m=0
font=cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        if conf<80:
            if id==1:
                id='naveen'
                cv2.putText(img,id,(x-w,y-h),font,0.5,(0,225,255),2,cv2.LINE_AA)
                break
            if id==2:
                id='sharath'
                cv2.putText(img,id,(x-w,y-h),font,0.5,(0,225,255),2,cv2.LINE_AA)
                break
        if conf>60:
                id='unknown'
                cv2.putText(img,id,(x-w,y-h),font,0.5,(0,225,255),2,cv2.LINE_AA)            
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
