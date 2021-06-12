import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


webcam=cv2.VideoCapture('TVSclass.mp4')
#webcam=cv2.VideoCapture(0)

while True:
    ###Read the current frame
    successful_frame_read,frame =webcam.read()

    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle-->cv2.rectangle(img,(x1,y1),(x1+x2,y1+y2),(R,G,B),thickness)
    # (x, y, w, h) = face_coordinates[1]
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)

    cv2.imshow('face detector', frame)

    key=cv2.waitKey(1)
    # stop if Q is pressed
    if key==81 or key==113:
        break

webcam.release()

print("code completed")


"""
#img=cv2.imread('friends.jpg')
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangle-->cv2.rectangle(img,(x1,y1),(x1+x2,y1+y2),(R,G,B),thickness)
#(x, y, w, h) = face_coordinates[1]
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),2)

#print(face_coordinates)

cv2.imshow('face detector',img)

cv2.waitKey()
"""

