import cv2

import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy
import time

Detector=cv2.face.LBPHFaceRecognizer_create();
Detector.read('recognizer/trainedData.yml')

camera = PiCamera()
rawCapture = PiRGBArray(camera, size = (608,800))
#camera.resolution = (1600, 1024)
camera.resolution = (608,800)
display_window = cv2.namedWindow("Faces")

#camera.start_preview()

face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Cascade files/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Cascade files/haarcascade_profileface.xml')

id=0

#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'id',(10,500), font, 4, (255,255,255),cv2.LINE_AA)

while(True):
    time.sleep(0.5)
    rawCapture.truncate(0)
    camera.capture(rawCapture, format="bgr")
    img = rawCapture.array


    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.flip(gray, 1)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profile_face = profile_cascade.detectMultiScale(gray, 1.3, 5)
    profile_face_flipped = profile_cascade.detectMultiScale(gray1, 1.3, 5)


    #print ('Found "+str(len(faces))+" face(s)')
    for (x2,y2,w2,h2) in faces:
         cv2.rectangle(img, (x2,y2), (x2+w2,y2+h2), (255,255,0), 2)
         id,conf=Detector.predict(gray[y2:y2+h2,x2:x2+w2])

         if (conf<50):
             

             if (id == 2):
                 id='Dr.Fathi'
             elif (id == 7):
                 id="Dr.Fathi_MSSite"
             elif (id == 9):
                 id="Mr.Farhangi"
             elif (id == 8):
                 id="Diba"
             elif (id == 5):
                 id ="Dr.Abolhasani"
             elif (id == 4):
                 id = "Dr.Feizi"
             elif (id == 3):
                 id="Dr.fathi"
             elif (id == 6):
                 id = "Dr.Mousazadeh"
             
         else:
             id = "unknown"


         
         cv2.putText(img, str(id), (x2-15, y2-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    for (x,y,w,h) in profile_face:
         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
         id,conf=Detector.predict(gray[y:y+h,x:x+w])


         if (conf<50):
             


             if (id == 2):
                 id='Dr.Fathi'
             elif (id == 7):
                 id="dr.Fathi_MSSite"
             elif (id == 8):
                 id="Diba"
             elif (id == 9):
                 id = "Rasoul"
             elif (id == 3):
                 id="Dr.Fathi"
             elif (id == 4):
                 id = "Dr.Feizi"
             elif (id == 5):
                 id = "Dr.Abolhasani"
             elif (id == 6):
                 id = "Dr.Musazadeh"
             
         else:
             id = "unknown"

         
         cv2.putText(img, str(id), (x-15, y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


    for (x1,y1,w1,h1) in profile_face_flipped:
         cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), (0,0,255), 2)
         id,conf=Detector.predict(gray1[y1:y1+h1,x1:x1+w1])

         if (conf<50):


             if (id == 2):
                 id='Dr.Fathi'
             elif (id == 7):
                 id="Dr.Fathi_MSSite"
             elif (id == 8):
                 id="Diba"
             elif (id == 9):
                 id="Rasoul Farhangi"
             elif (id == 6):
                 id="Dr.M.Mousazadeh"
             elif (id == 3):
                 id = "Dr.Fathi"
             elif (id == 4):
                 id = "Dr.Feizi"
             elif (id == 5):
                 id = "Dr.abolhasani"
             
         else:
             id = "unknown"





         
         cv2.putText(img, str(id), (x1-15, y1-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                 
    print (id)

         #cv2.rectangle(img, (x-22,y-90), (x+w+22, y-22), (255,255,0, -1))



    

         
    
    cv2.imshow("Faces",img);
    #cv2.imwrite('result.jpeg', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cv2.destroyAllWindows()

