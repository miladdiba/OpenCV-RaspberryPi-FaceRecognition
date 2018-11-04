

import cv2

from PIL import Image

import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy
import time

#start_preview
camera = PiCamera()
rawCapture = PiRGBArray(camera, size = (608, 800))
camera.resolution = (608, 800)




#camera.start_preview()
#display_window = cv2.namedWindow("Faces")





#Requesting an id for the person ( must be integer)
face_id = input("enter person's id: ")
id=0




#Addressing the cascade_classifier to detect faces
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Sentex/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('/home/pi/Desktop/Cascade files/haarcascade_profileface.xml')


#Defining a counter to stop creating DataSets as counter reaches maximum
count = 0
count1 = 0
count2 = 0

#creating an Infinite Loop
while(True):
     time.sleep(0.1)
     rawCapture.truncate(0)

     #Capturing BGR faces
     camera.capture(rawCapture, format="bgr")




     

     #Converting Captured Data to Matrices
     img = rawCapture.array




     

     #Converting to Grayscale since CascadeClassifier uses grayscale
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


     #Flipping in the case that it is left Profile
     gray1 = cv2.flip(gray, 1)
     #cv2.imshow('flipped', gray1)
     #cv2.imshow('original', gray)




     

     #Detecting faces in grayscale-ScaleFactor-minNeighbors
     
     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
     profile_face = profile_cascade.detectMultiScale(gray, 1.5, 5)
     profile_face_flipped = profile_cascade.detectMultiScale(gray1, 1.5, 5)


     for (x2,y2,w2,h2) in faces:
          cv2.rectangle(img, (x2,y2), (x2+w2,y2+h2), (0,255,255), 3)
          count2 = count2 + 1
          print ('found ' + str(len(faces)) + 'frontal_face')
          cv2.imwrite("datasets/User." + str(face_id) + '.' + str(count2) + ".jpg", gray[y2:y2+h2,x2:x2+w2])
          cv2.imshow("Frontal Face", img)
     if cv2.waitKey(100) & 0xFF == ord('q'):
         break
     elif count2 >= 20:
         break



     
     for (x,y,w,h) in profile_face:
          cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 3)
          count = count + 1
          print ('found ' + str(len(profile_face)) + 'profile_face')
          cv2.imwrite("datasets/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
          cv2.imshow("Profile Face", img)
     if cv2.waitKey(100) & 0xFF == ord('q'):
         break
     elif count >= 20:
         break
     


     for (x1,y1,w1,h1) in profile_face_flipped:
          cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), (255,0,255), 3)
          count1 = count1 + 1
          print ('found ' + str(len(profile_face_flipped)) + 'profile_face_Flipped')
          cv2.imwrite("datasets/User." + str(face_id) + '.' + str(count1) + ".jpg", gray1[y1:y1+h1,x1:x1+w1])
          cv2.imshow("Profile Face", img)
     if cv2.waitKey(100) & 0xFF == ord('q'):
         break
     elif count1 >= 20:
         break
     

#camera.stop.preview()
cv2.destroyAllWindows()

