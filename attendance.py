#import PIL.Image
import PIL.ImageTk
from tkinter import *
import cv2
import os
import csv
import numpy as np
import pandas as pd
import datetime
import time
import webbrowser


window = Tk()
window.geometry('1000x800')
window.configure(bg='grey15')
window.resizable(width=False, height=False)
window.title("Tech Giant Attendance System")
# window.configure(background='#D0D3D4')
image = PIL.Image.open("logo.png")
photo = PIL.ImageTk.PhotoImage(image)
lab = Label(image=photo, bg='grey15')
lab.pack()

fn = StringVar()
entry_name = Entry(window, textvar=fn, width=22, font=("roboto", 15))
entry_name.place(x=265, y=260)
ln = StringVar()
entry_id = Entry(window, textvar=ln, width=22, font=("roboto", 15))
entry_id.place(x=265, y=318)
em = StringVar()
enter_email = Entry(window, textvar=em, width=22, font=("roboto", 15))
enter_email.place(x=265, y=375)


def clear1():
    entry_name.delete(first=0, last=22)


def clear2():
    entry_id.delete(first=0, last=22)


def clear3():
    enter_email.delete(first=0, last=22)


def close():
    quit()


def detect():
    Id = ln.get()
    name = fn.get()
    email = em.get()
    cascade_face = cv2.CascadeClassifier(r"C:\Users\preya\PycharmProjects\Final\haarcascade_frontal.xml")
    cam = cv2.VideoCapture(0)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, Closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed

            img_name = "{}.jpg".format(name.lower() + "." + Id + '.' + str(img_counter))
            cv2.imwrite("TrainingImage\ " + img_name, roi_gray)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    row = [Id, name, email]
    with open(r"C:\Users\preya\PycharmProjects\Final\StudentDetails\StudentDetails.csv", 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


def ImagesAndNames(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # Loading the images in Training images and converting it to gray scale
        g_image = PIL.Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        image_ar = np.array(g_image, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(image_ar)
        Ids.append(Id)
    return faces, Ids


def train_image():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = ImagesAndNames("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"C:\Users\preya\PycharmProjects\Final\TrainingImage_yml\Trainner.yml")


def Math():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"C:\Users\preya\PycharmProjects\Final\TrainingImage_yml\Trainner.yml")

    faceCascade = cv2.CascadeClassifier(r"C:\Users\preya\PycharmProjects\Final\haarcascade_frontal.xml")
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                with open(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Math.csv", 'a') as f:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id) + "-" + aa
                    z = [Id, aa, date, timeStamp]
                    writer = csv.writer(f)
                    writer.writerow(z)
            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 65:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Math', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def physics():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"C:\Users\preya\PycharmProjects\Final\TrainingImage_yml\Trainner.yml")

    faceCascade = cv2.CascadeClassifier(r"C:\Users\preya\PycharmProjects\Final\haarcascade_frontal.xml")
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # col_names = ['Id', 'Name', 'Date', 'Time']
    # attendance = pd.DataFrame(columns=col_names)
    # fileName = "attendance.csv"
    # attendance.to_csv(fileName, index=False)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                with open(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Physics.csv", 'a') as f:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id) + "-" + aa
                    z = [Id, aa, date, timeStamp]
                    writer = csv.writer(f)
                    writer.writerow(z)

            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 65:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        # attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Physics', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def chemistry():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"C:\Users\preya\PycharmProjects\Final\TrainingImage_yml\Trainner.yml")

    faceCascade = cv2.CascadeClassifier(r"C:\Users\preya\PycharmProjects\Final\haarcascade_frontal.xml")
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # col_names = ['Id', 'Name', 'Date', 'Time']
    # attendance = pd.DataFrame(columns=col_names)
    # fileName = "attendance.csv"
    # attendance.to_csv(fileName, index=False)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                with open(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Chemistry.csv", 'a') as f:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    tt = str(Id) + "-" + aa
                    z = [Id, aa, date, timeStamp]
                    writer = csv.writer(f)
                    writer.writerow(z)

            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 65:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        # attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Chemistry', im)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def update():
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Math.csv")
    df.drop_duplicates(subset=['Date'], inplace=True, keep='first')
    df.to_csv(r'C:\Users\preya\PycharmProjects\Final\FinalAttendanceSheet\Math_final.csv', index=False)
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Physics.csv")
    df.drop_duplicates(subset=['Date'], inplace=True, keep='first')
    df.to_csv(r'C:\Users\preya\PycharmProjects\Final\FinalAttendanceSheet\Physics_final.csv', index=False)
    df = pd.read_csv(r"C:\Users\preya\PycharmProjects\Final\AttendanceSheet\Chemistry.csv")
    df.drop_duplicates(subset=['Date'], inplace=True, keep='first')
    df.to_csv(r'C:\Users\preya\PycharmProjects\Final\FinalAttendanceSheet\Chemistry_final.csv', index=False)




url = "https://public.tableau.com/profile/preyas2192#!/"
new = 1


def attendence_sheet():
    webbrowser.open(url, new=new)


label1 = Label(window, text="Create & Develope By Tech Giant", fg='DeepSkyBlue2', bg='grey15',
               font=("roboto", 20, 'bold')).place(x=280, y=150)
label2 = Label(window, text="New User", fg='#717D7E', bg='grey15', font=("roboto", 25, "bold")).place(x=20, y=200)
label3 = Label(window, text="Enter Name :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=260)
label4 = Label(window, text="Enter Roll Number :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=315)
label5 = Label(window, text="Enter Email address :", fg='black', bg='grey15', font=("roboto", 18)).place(x=20, y=370)
label6 = Label(window, text="Already a User?", fg='#717D7E', bg='grey15', font=("roboto", 25, "bold")).place(x=20,
                                                                                                             y=600)


button1 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear1)
button1.place(x=570, y=255)
button2 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear2)
button2.place(x=570, y=313)
button3 = Button(window, text="clear", fg='#000000', bg='red', relief=RAISED, font=("roboto", 15, "bold"),
                 command=clear3)
button3.place(x=570, y=370)
button4 = Button(window, text="Submit", width=5, fg='#000000', bg='dark green', relief=RAISED,
                 font=("roboto", 15, "bold"),
                 command=detect, height=1)
button4.place(x=20, y=450)
button5 = Button(window, text="Train Images", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=train_image)
button5.place(x=20, y=530)
button6 = Button(window, text="Maths", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=Math)
button6.place(x=20, y=660)
button7 = Button(window, text="Physics", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=physics)
button7.place(x=140, y=660)
button8 = Button(window, text="Chemistry", fg='#000000', bg='dark green', relief=RAISED, font=("roboto", 15, "bold"),
                 command=chemistry)
button8.place(x=280, y=660)
button9 = Button(window, text="Update", fg='#000000', bg='RoyalBlue1', relief=RAISED, font=("roboto", 15, "bold"),
                 command=update)
button9.place(x=20, y=740)
button10 = Button(window, text="Exit", width=5, fg='#000000', bg='Red', relief=RAISED, font=("roboto", 15, "bold"),
                  command=exit)
button10.place(x=870, y=740)
# button11 = Button(window, text="Check attendance", fg='#000000', bg='RoyalBlue1', relief=RAISED,
#                   font=("roboto", 15, "bold"),
#                   command=attendence_sheet)
# button11.place(x=140, y=740)
window.mainloop()
