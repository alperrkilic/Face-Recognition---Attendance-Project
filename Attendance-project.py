from datetime import datetime
import cv2
import numpy as np
import face_recognition
import os

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

print(myList)
# ['Bill-Gates.png', 'Elon-musk.png', 'Elon-test.png', 'Jack-Ma.png']

for cl in myList:  # cl is the name of our image for example Jack-Ma.png
    curImg = cv2.imread(f'{path}/{cl}')  # reading the current image
    images.append(curImg)  # append the current image to images array
    classNames.append(os.path.splitext(cl)[0])
    # only gives name part excludes .jpg and .png

print(classNames)  # ['Bill-Gates', 'Elon-musk', 'Elon-test', 'Jack-Ma']


# since opencv is working in terms of BGR we have to convert images to RGB.
# 'images' is an array of our attendance images, so in findEncodings we're looping through every image and changing their color to RGB
# and finding and their encodings into encode variable. After that storing these values in encodeList
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # since we are sending a single image every time in the loop, we need the first element therefore [0]
        # if we were sending more than one input image then we could use [1] ... [n]
        # in order to compare faces, we need face encodings and we are holding these values in an array (encodeList[])
        encodeList.append(encode)
    return encodeList


# in order to use datetime.now() we have to import 'from datetime import datetime'
# markAttendance(name) function takes a name, both in read and write modes opens the Attendance.csv (.csv means comma separated values)
# so if we call the function : markAttendance('Elon') , in Attendance.csv -> Elon,12:17:35 we see this
# Since we are checking the attendance of an existing list, if we find a match between faces we can call markAttendance function with name value

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(';')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name};{dtString}\n')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)  # for Webcam

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # resize the image by 1/4, since we're doing it in the real time reducing the size of our image will help us to speed the process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # color BGR to RGB

    facesCurFrame = face_recognition.face_locations(imgS)
    # facesCurFrame -> gives us the coordinates where the face is found
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # encodesCurFrame -> the first parameter of function face_encodings is the image that contains one or more faces, second parameter is the
    # locations of faces

    # function compare_faces() accepts two parameter, first one is the known encode lists, and the second one is the encodedFace that is to be
    # compared with the list. So comparing returns true or false, and we're storing this into matches, since we've 4 attendance pictures,
    # matches will store 4 boolean value.

    # At the below you can see zip(encodesCurFrame,facesCurFrame) what it says that in for loop, encodeFace will be looping through
    # encodesCurFrame and faceLoc will be looping through facesCurFrame inside the for loop so simply if we want 2 variables in loop,
    # we can use zip()

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        print(matches)  # [False, False, False, False]
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)  # [0.71132107 0.71845304 0.7306767  0.86115993]
        matchIndex = np.argmin(faceDis)
        print(matchIndex)
        # the index of the lowest number among face distances will be stored in the matchIndex

        # As I mentioned above, matches holds the boolean values so in the if statement, for example matchIndex is 2, it controls
        # matches[2] whether it's true or not, if true then condition is satisfied and the statements inside if statement will be executed

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            # faceLoc is coming from our facesCurFrame and gives us the locations and assigning faceLoc values into y1, x2, y2, x1
            # since the size was reduced by 1/4 we have to multiply y1, x2, y2, x1 in order to draw the accurate rectangle around the face
            # that is matched with our Attendance list
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
