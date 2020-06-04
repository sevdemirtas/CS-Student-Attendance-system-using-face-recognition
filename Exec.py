import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image
from datetime import datetime
import time
import os
from os import path
import csv
import cv2


AttndPage = Tk()
AttndPage.title("Attendance System")
AttndPage.config(background = 'dimgray')
Label(AttndPage, text = 'ATTENDANCE SYSTEM USING FACE RECOGNITION', fg = 'black', bg = 'dimgray',font = ('Times',38,'italic')).pack()


def warning():
    messagebox.showwarning("WARNING!", "Enter your Name and ID")


def newRegistry():                                                                        #  When we clicked the save button in new registration window, newRegistry() func will be called
    studentName=studentNameTxt.get()
    studentID=studentIDtxt.get()
    if studentName == '' or studentID == '':                                            # studentID and studentName are should be entered for new registry
        warning()
    else:
        cam = cv2.VideoCapture(0)                                                      # capture your first picture for data
        faceCascade = cv2.CascadeClassifier('C:\\Users\\Sevde\\Desktop\\SeniorProject\\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('C:\\Users\\Sevde\\Desktop\\SeniorProject\\haarcascade_eye.xml')
        i = 0
        
        while True:
            ret,img = cam.read()                                                                  # ret is not used
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4) 
            
                                 # returns x, y, w, h around face
            
            for(x, y, w, h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                #cv2.rectangle(img,(x,y),(x + w,y + h),(x+w,y+h),(255, 0, 0), 2)

                i += 1          
                # increment i
                eyes = eye_cascade.detectMultiScale(gray)  
            for (ex,ey,ew,eh) in eyes:   
                cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                i += 1          
                # increment i
                
                cv2.imwrite("Faces/id."+str(studentID)+"."+str(i)+".jpg",gray[y:y + h, x:x + w])   # saving the images to Faces Folder
         
                cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):                                               # Firstly waits,then if you press 'q' then quits
                break
            elif i > 50:                                                                    # If takes more than 50 images, then quits
                break
            
        cam.release()
        cv2.destroyAllWindows()
        
        row = [studentID,studentName]
        with open('StudentDetails.csv','a+',newline = '') as csvFile:                                  # writing the student StudentDetails to 'StudentDetails.csv'
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()

def regBtn():                                                                           # When new registry button is clicked, regBtn() func will be called
    regPage = Tk()
    regPage.title("New Student Registration")
    regPage.config(background = 'gray')
    regPage.geometry('520x330')

    if not path.exists('Faces'):
        os.mkdir('Faces')

    if not path.exists('StudentDetails.csv'):
        with open ('StudentDetails.csv','w', newline = '') as temp:
            tkWriter = csv.writer(temp)
    
    Label(regPage, text = 'STUDENT REGISTRATION', fg = 'white', bg = 'orange',font = ('Times',32,'italic')).pack()
    Label(regPage, text = 'Enter Your Name', fg = 'black', bg = 'gray',font = ('Times',19,'italic')).pack()
    
    global studentNameTxt
    studentNameTxt = tk.Entry(regPage, fg = 'white', bg = 'dimgray',font = ('Times',12,'italic'))
    studentNameTxt.pack()
    
    Label(regPage, text = 'Enter Your ID', fg = 'black', bg = 'gray',font = ('Times',19,'italic')).pack()
    
    global studentIDtxt
    studentIDtxt = tk.Entry(regPage, fg = 'white', bg = 'dimgray',font = ('Times',12,'italic'))
    studentIDtxt.pack()

    newRegBtn = tk.Button(regPage, text = 'SAVE', command = newRegistry, fg = 'black', bg = 'dimgray', font = ('Times',14,'italic')).pack()


def trainingClassifier():                                                                          # When train button is clicked, trainingClassifier() func will be called
    path = [os.path.join('Faces',f) for f in os.listdir('Faces')]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')                                                    # convert all images in the Faces directory to numpy arrays
        im = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split(".")[1])                                        # retrieve id of the image using split() funciton on the file names
        faces.append(im)
        ids.append(id)

    ids = np.array(ids)

    recognizer = cv2.face.LBPHFaceRecognizer_create()                                                  # create custom classifier and trains it with the custom faces and ids
    recognizer.train(faces, ids)
    recognizer.write("faceCascade.xml")


def checkAttnd():                                                                                       # When lesson window is clicked, checkAttnd() fun will be called
    studentName = lessonText.get()
    ts = time.time()                                                                                    #Converts timestamp to datetime objects  
    date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = 'AttendanceSheet/' + date + '/' + studentName + "_" + Hour + "-" + Minute + "-" + Second + ".csv"

    if not path.exists('AttendanceSheet'):
        os.mkdir('AttendanceSheet')

    if not path.exists('AttendanceSheet/' + date):
        os.mkdir('AttendanceSheet/' + date)
        
    with open (fileName,'w', newline = '') as temp:
        tkWriter = csv.writer(temp)
        tkWriter.writerow(['ID','Name','Attendance','Time'])
    
    with open('StudentDetails.csv') as csvFile:                                                                    #  StudentsDetails will be copied from StudentDetails.csv file, then students will be 'Absent' by default
        with open (fileName,'a') as temp:
            objReader = csv.DictReader(csvFile, fieldnames = ['ID','Name'])
            fieldnames = ['ID','Name','Attendance','Time']
            writer = csv.DictWriter(temp, fieldnames = fieldnames)
            for row in objReader:
                id1 = str(row['ID'])
                stud1 = str(row['Name'])
                writer.writerow({'ID' : id1, 'Name' : stud1, 'Attendance' : 'Absent','Time' : 'Absent'})


    faceCascade = cv2.CascadeClassifier('C:\\Users\\Sevde\\Desktop\\SeniorProject\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\Users\\Sevde\\Desktop\\SeniorProject\\haarcascade_eye.xml')
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("faceCascade.xml")                                                                                      # Reading custom faceCascade classifier

    cam = cv2.VideoCapture(0)

    while True:
        ret,img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        
        coords = []
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id, _ = clf.predict(gray[y:y+h, x:x+w])
        eyes = eye_cascade.detectMultiScale(gray) 
            
               
        for (ex,ey,ew,eh) in eyes:   
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(img, str(id), (x+h, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                       # Draws a rectangle around the face and recognized Student ID number appears in the corner
            cv2.imshow('img', img)

            
            with open (fileName,'r+') as temp:
                objReader = csv.DictReader(temp, fieldnames = ['ID','Name','Attendance','Time'])
                writer = csv.DictWriter(temp, fieldnames = ['ID','Name','Attendance','Time'])
                
                for row in objReader:                                                                              # Compares the ID of the face within the frame to each ID in the Attendance file and adds as "Present" in each ID match
                    id1 = str(row['ID'])                          
                    stud1 = str(row['Name'])
                    if str(id) == str(row['ID']):
                        if str(row['Attendance']) == 'Present':
                            continue
                        writer.writerow({'ID' : id1, 'Name' : stud1, 'Attendance' : 'Present','Time' : str(datetime.fromtimestamp(ts).strftime('%H:%M:%S'))})
        

        if cv2.waitKey(1) & 0xFF == ord('q'):                                                                    # q is the key button for stop recording video for detection
            break
    
    data = pd.read_csv(fileName)                                                                                 # Making data frame from csv file
    data.drop_duplicates(subset ='ID', keep = 'last', inplace = True)                                            # Dropping all duplicate values for students based on id
    data = data.sort_values('ID')                                                                                # Sorting by ID
    data.to_csv(fileName, index = False)
    
    cam.release()
    cv2.destroyAllWindows()


def checkAttndBtn():                                                                                             # When check attendance button is clicked, checkAttndBtn() func will be called and print attendance with excel file
    lessonWin = Tk()
    lessonWin.title("Lesson Name")
    lessonWin.config(background = 'orange')
    Label(lessonWin, text = 'Enter Your Lesson', fg = 'Black', bg = 'orange',font = ('Times',13,'italic')).pack()


    global lessonText
    lessonText = tk.Entry(lessonWin, fg = 'white', bg = 'orange', font = ('Times',16,'italic'))
    lessonText.pack()
    lessonBtn = tk.Button(lessonWin, text = 'SAVE', command = checkAttnd, fg = 'Black', bg = 'orange', font = ('Times',14,'italic')).pack()


registerBtn = tk.Button(AttndPage, text = 'NEW REGISTRY', command = regBtn, fg = 'White', bg = 'orange', font = ('Times',30,'italic'), height = 2, width = 22).pack()
attndBtn = tk.Button(AttndPage, text = 'CHECK ATTENDANCE', command = checkAttndBtn, fg = 'White', bg = 'orange',font = ('Times',30,'italic'), height = 2, width = 22).pack()
trainingBtn = tk.Button(AttndPage, text = 'TRAINING', command = trainingClassifier, fg = 'White', bg = 'orange',font = ('Times',15,'italic'), height = 2, width = 10).pack(side = LEFT)

AttndPage.mainloop()