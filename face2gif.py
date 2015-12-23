# import numpy as np
import sys
import os.path

try:
    import cv2
except:
    print("Please install OpenCV")
    quit()

if (os.path.isfile("haarcascade_frontalface_default.xml")):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:
    sys.exit("haarcascade_frontalface_default.xml not found")

if (os.path.isfile("haarcascade_eye.xml")):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
else:
    sys.exit("haarcascade_eye.xml not found")


img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

if (len(faces) == 1):
    print("Found one face!")
else:
    print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

try:
    cv2.imshow('face2gif', img)
    cv2.waitKey(0)
except (KeyboardInterrupt):
    cv2.destroyAllWindows()
    print("User pressed Ctrl+C")

cv2.destroyAllWindows()

# if __name__ == '__main__':
