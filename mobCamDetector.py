import urllib.request
import cv2
import numpy as np

rec = cv2.face.LBPHFaceRecognizer_create()

url = "http://192.168.43.155:8080/shot.jpg"
name={1:'Aadish',3:'Anjani',4:'Vaibhav',5:'Chiran',6:'Aayush'} #,7:'Manish',8:'Nitu'}
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec.read('recognizer\\trainingData.yml')

id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        id,conf =rec.predict(gray[y:y+h,x:x+h])
        cv2.putText(img, name[id], (x,y+h),font, 2, (0,255,0), 2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1)==ord('q')):
        break
    
cv2.destroyAllWindows()
