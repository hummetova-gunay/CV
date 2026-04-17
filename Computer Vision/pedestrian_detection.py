import numpy as np
import cv2 as cv

fullbody_cascade = './cascades/haarcascade_fullbody.xml'
classifier = cv.CascadeClassifier(fullbody_cascade)

video = cv.VideoCapture('./videos/walking.avi')

while video.isOpened():
    flag, frame = video.read()
    if flag:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        people = classifier.detectMultiScale(frame_gray, 1.2, 3) # scale factor, minimum neighbors
        for (x, y, w, h) in people:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 2)
            cv.putText(frame, 'Person', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 2)
        cv.imshow('Person',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv.destroyAllWindows()