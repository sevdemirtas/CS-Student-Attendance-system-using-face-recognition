

In this project, we’ll look at a simple way to get started with face recognition using Python and the open source library OpenCV.

Try to run the code. You must understand what the code does, not only to run it properly but also to troubleshoot it.
Make sure to use OpenCV cv2.
Have a working webcam so this system can work properly. 
Make sure change your file path based on the file on your computer.

BEFORE YOU START WHAT YOU NEED FOR YOUR SYSTEM

1. Python

Import the below requirements with using cmd or anaconda prompt(recommended cause easy to built new environment for python).

2. import OpenCV (pip install --user opencv-contrib-python)
3. import pandas (pip install pandas) 
4. import numpy  (pip install numpy)
5. import pillow (pip install pillow)
6. import tkinter(pip install tkinter)
7. import cv2(if neccessary , pip install cv2)


HOW TO USE ATTENDANCE SYSTEM

1)When you click register button you can enter your ID and name for register new student.
2)The webcam will open up and the program will take multiple images of the face of the student. These will be stored in the 'faces' directory in your computer.
3)'Studentdetails.csv' file will be created with the name and ID of the student after register.
4)Click on the train button to train the classifier with the faces of the student, taken from the faces directory.
5)Now click on check attendance button whenever you want to mark the attendance and give a name your attendance file. 
After, the webcam will open up, and the students who show their faces will be marked as present and the file will be stored in the attendance folder.
