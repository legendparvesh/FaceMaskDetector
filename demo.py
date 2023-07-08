import cv2
import numpy
import matplotlib.pyplot as plt
img=cv2.imread('test/with_mask/1-with-mask.jpg',0)
#print(img)
#img2=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#converting rgb to hsv
#img3=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
#to resize an image
img4=cv2.resize(img,(1080,1090))
facecas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face=facecas.detectMultiScale(img,1.1,4)

for (x, y, w, h) in face:
    cv2.rectangle(img,(x, y), (x+w, y+h), (255,0,0),2)
    #Display the output

#to display the image
cv2.imshow('image',img)
cv2.imshow('image2',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
print(img2.shape)