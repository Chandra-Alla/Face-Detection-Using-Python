import cv2
import numpy as np
import face_recognition

#To Present the actual image
imgElon=face_recognition.load_image_file('ImagesBasic/ElonMusk.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

#To Present the test image
imgTest=face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#This is for the First Image
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),3)

#This is for the Second Image
faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),3)


#Comparing the Distance between the Images.
results= face_recognition.compare_faces([encodeElon],encodeTest)
# the distance between two face
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
#Printing the the true value on the test image
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,255),2)


#To show the images.
cv2.imshow('ElonMusk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)
