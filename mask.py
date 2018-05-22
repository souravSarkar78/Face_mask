import cv2
import numpy as np

face_c = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_mask = 'maska2.png'
face_mask = cv2.imread(face_mask)

h_mask, w_mask = face_mask.shape[:2]

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation= cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_c.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_rects:

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if h > 0 and w> 0:
            h, w = int(1.1*h), int(0.9 * w)

            y += int(0.005 * h)
            x -= int(0.01 * w)


            #print(y)

            frame_roi = frame[y:y+h, x:x+w]
            #print(frame[y:y+h, x:x+w])


            face_mask_small = cv2.resize(face_mask, (w, h), interpolation= cv2.INTER_AREA)

            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(gray_mask, 251, 255, cv2.THRESH_BINARY_INV)

            mask_inv = cv2.bitwise_not(mask)


            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask= mask_inv)


            frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)

    cv2.imshow('face_mask', frame)
    
    if cv2.waitKey(1) == 32:
        break
cv2.imwrite('face.jpg', frame)
cv2.destroyAllWindows()
cam.release()
