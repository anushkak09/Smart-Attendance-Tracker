import face_recognition
import imutils
import pickle
import time
import cv2
import os
import csv
from datetime import datetime
from datetime import date

today = date.today()
total = int(input("\nTotal no of students in class: "))
Subject = input("\nSubject name: ")


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data = pickle.loads(open('face_enc', "rb").read())
filename=time.strftime('%Y-%m-%d__%H-%M',time.localtime())
open('Attendance'+filename+'.csv','a+')
with open('Attendance'+filename+'.csv','r+') as f:
    f.write('\t\t\t WATUMULL INSTITUTE OF TECHNOLOGY')
    f.write('\n\t\t\t BRANCH : Computer Science')
    f.write('\nDate : ')
    f.write(str(today))
    f.write('\nlecture name : ')
    f.write(Subject)
    f.write('\n\nName of Students,Time of Arrival')

count = 0
def markAttendance(name):

        with open('Attendance'+filename+'.csv','r+') as f:
            myDataList=f.readlines()
            nameList=[]
            for line in myDataList:
                entry=line.split(',')
                nameList.append(entry[0])

            if name not in nameList:
                global count
                now=datetime.now()
                dtString=now.strftime('%H:%M:%S')
                count += 1
                f.writelines(f'\n{name},{dtString}')


 
print("Streaming started")
video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1 , minNeighbors=5,minSize=(60, 60),flags= cv2.CASCADE_SCALE_IMAGE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    encodings = face_recognition.face_encodings(rgb)
    names = []
   
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "unknown"
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

absent = int(total-count)
with open('Attendance'+filename+'.csv','a+') as f:
    f.write('\n\n\n\n\n\t\t\t Total no of Students : ')
    f.write(str(total))
    f.write('\n\t\t\tStudents Present : ')
    f.write(str(count))
    f.write('\n\t\t\tStudents Absent : ')
    f.write(str(absent))

video_capture.release()
cv2.destroyAllWindows()

