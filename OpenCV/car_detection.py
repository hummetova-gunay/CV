import numpy as np
import cv2 as cv

car_cascade = './cascades/haarcascade_car.xml'
classifier = cv.CascadeClassifier(car_cascade)

video = cv.VideoCapture('./videos/road_car_view.mp4')

while video.isOpened():
    flag, frame = video.read()
    if flag:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cars = classifier.detectMultiScale(frame_gray, 1.2, 3) # scale factor, minimum neighbors
        for (x, y, w, h) in cars:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 2)
            cv.putText(frame, 'Car', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 2)
        cv.imshow('Cars',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv.destroyAllWindows()