import cv2
import numpy as np
from google.colab.patches import cv2_imshow
haar_Cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img=cv2.imread('cricket1.jpeg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=np.array(gray,dtype="uint8")
face=haar_Cascade.detectMultiScale(gray,1.1,10)
for(x,y,w,h) in face:
  cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
cv2_imshow(img)
