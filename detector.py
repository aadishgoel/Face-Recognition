import cv2
import numpy as np      # 2,7,8 Archived
name={1:'Aadish',3:'Anjani',4:'Chhavi',69:'Vaibhav',5:'Chiran',6:'Aayush',9:'Jatin',10:'Pooja',11:'Prem',12:'Yash',13:'Harsh',14:'Tanay',15:'Sourav',16:'Kavish',17:'Namrata',18:'Sangeetha'} #,7:'Manish',8:'Nitu'}
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\trainingData.yml')
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        id,conf =rec.predict(gray[y:y+h,x:x+h])
        cv2.putText(img, name[id], (x,y+h),font, 2, (0,255,0), 2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
