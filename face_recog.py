import face_recognition
import imutils
import pickle
import time
import cv2
import os
import csv
from datetime import datetime

faceCascade = cv2.CascadeClassifier(r"C:\\Users\\Dell\\Documents\\SmartAttendanceTracker\\haarcascade_frontalface_default.xml")

data = pickle.loads(open('face_enc', "rb").read())
filename=time.strftime('%Y-%m-%d__%H-%M',time.localtime())
def markAttendance(name):
        with open('Attendance'+filename+'.csv','a+') as f:
            myDataList=f.readlines()
            nameList=[]
            for line in myDataList:
                entry=line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now=datetime.now()
                dtString=now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    encodings = face_recognition.face_encodings(rgb)
    names = []
   
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        markAttendance(name)
  
     
    
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
          
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xff== ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
