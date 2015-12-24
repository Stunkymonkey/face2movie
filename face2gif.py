# import numpy as np
import sys
import os.path

try:
    import cv2
except:
    print("Please install OpenCV")
    quit()

global VERBOSE
VERBOSE = False
FOLDER = os.path.join(os.path.abspath("."))


if (os.path.isfile("haarcascade_frontalface_default.xml")):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:
    sys.exit("haarcascade_frontalface_default.xml not found")

if (os.path.isfile("haarcascade_eye.xml")):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
else:
    sys.exit("haarcascade_eye.xml not found")


def dectectFace(gray):
    """detecting faces"""
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))


def detectEye(roi_gray):
    """detecting eyes"""
    return eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25))


def drawFaces(faces, img):
    """drawing faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)


def drawEyes(eyes, img):
    """drawing eyes"""
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)


def detect(img, gray):
    faces = dectectFace(gray)

    if VERBOSE:
        if (len(faces) == 1):
            print("Found one face!")
        else:
            print("Found {0} faces!".format(len(faces)))

    drawFaces(faces, img)

    data = dict()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = []
        eyes = detectEye(roi_gray)

        data[(x, y, w, h)] = eyes

        if VERBOSE:
            if (len(eyes) == 1):
                print("Found one eye!")
            else:
                print("Found {0} eyes!".format(len(eyes)))

        drawEyes(eyes, roi_color)
    return data


def calculatePicture(file):
    """gettings infos of the image"""
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_data = detect(img, gray)
    height, width, channels = img.shape

    """
    try:
        cv2.imshow('face2gif', img)
        cv2.waitKey(0)
    except (KeyboardInterrupt):
        cv2.destroyAllWindows()
        print("User pressed Ctrl+C")

    cv2.destroyAllWindows()
    """
    return [file, height, width, face_data]


def checkInput():
    """check input and return files"""
    files = []
    if not sys.argv[1]:
        print("No image given")
        quit()
    elif (sys.argv[1].endswith("/") and os.path.isdir(sys.argv[1])):
        onlyfiles = []
        for f in os.listdir(sys.argv[1]):
            if os.path.isfile(os.path.join(sys.argv[1], f)):
                onlyfiles.append(f)
        for file in onlyfiles:
            if os.path.isfile(sys.argv[1] + file):
                files.append(sys.argv[1] + file)
    else:
        for file in sys.argv[1:]:
            files.append(file)
    return files


if __name__ == '__main__':
    data = []
    files = checkInput()
    for file in files:
        data.append(calculatePicture(file))
    print(data)
