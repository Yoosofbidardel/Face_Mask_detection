import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import winsound

global ip
ip=0
my_model=load_model('C:/first_face_mask_detection_model.h5')


face_cascade=cv2.CascadeClassifier('C:/haarcascade_frontalface_default.xml')

color_dict={1:(0,0,255),0:((0,255,0))}
label_dict={0:'Mask',1:'No Mask'}
pros=['.','..','...']
cap=cv2.VideoCapture(0)

while True:


    ss,img=cap.read()
    img = cv2.flip(img, 1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.blur(gray, (5, 5))
    face_location = face_cascade.detectMultiScale(gray_filtered,2,1,minSize=(50,50))


    cv2.putText(img, 'processing video...', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    for (x,y,w,h) in face_location:
        roi=img[y:y+h,x:x+w,:]
        roi=cv2.resize(roi, (150, 150))
        roi_normal = roi/255.0
        roi_normal = roi_normal[..., ::-1].astype(np.float32)
        roi_normal=np.reshape(roi_normal,(1,150,150,3))

        result=my_model.predict(roi_normal)
        label=label_dict[int(result>.5)]
        color=color_dict[int(result>.5)]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)

        if result>.5 :

            cv2.putText(img,'accuracy ='+str(round(result.item(),5)),(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        else:
            cv2.putText(img,'accuracy ='+str(round(1-result.item(),6)),(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)








    cv2.imshow('Live', img)

    if cv2.waitKey(1) & 0XFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
