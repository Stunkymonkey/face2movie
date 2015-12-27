# import numpy as np
import sys
import os.path
from math import atan, pi
import numpy as np

try:
    import cv2
except:
    sys.exit("Please install OpenCV")

try:
    from images2gif import writeGif
except:
    sys.exit("Please install images2gif")

FOLDER = os.path.join(os.path.abspath("."))
DEST_DIR = os.path.join(os.path.abspath(".") + r"/tmp/")

if (os.path.isfile("haarcascade_frontalface_default.xml")):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:
    sys.exit("haarcascade_frontalface_default.xml not found")

if (os.path.isfile("haarcascade_eye.xml")):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
else:
    sys.exit("haarcascade_eye.xml not found")

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)


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

    # for making sure only having one face
    if len(faces) == 0:
        return None, None

    # drawFaces(faces, img)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        eyes = detectEye(roi_gray)

        if len(eyes) != 2:
            return None, None

        # drawEyes(eyes, roi_color)
        return faces, eyes
    # return faces, eyes


def matrixPicture(face, eyes, height, width):
    """calculation of rotation and movement of the image"""
    center = tuple((face[0] + (face[2] / 2), face[1] + (face[3] / 2)))
    scale = 1.0

    M1 = np.float32([[1, 0, (width / 2) - center[0]],
                     [0, 1, (height / 2) - center[1]]])

    eye1 = tuple((eyes[0][0] + (eyes[0][2] / 2),
                  eyes[0][1] + (eyes[0][3] / 2)))
    eye2 = tuple((eyes[1][0] + (eyes[1][2] / 2),
                  eyes[1][1] + (eyes[1][3] / 2)))

    angle = atan((float(eye2[1]) - float(eye1[1])) /
                 (float(eye2[0]) - float(eye1[0]))) * 180 / pi

    M2 = cv2.getRotationMatrix2D(center, angle, scale)

    # Matrix = np.dot(M1, M2)
    # Matrix = cv2.getAffineTransform(M1, M2)
    # print(Matrix)
    return M1, M2


def calculatePicture(file):
    """gettings infos of the image and applie the matrixes"""
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, eyes = detect(img, gray)
    height, width, channels = img.shape

    if faces is None or eyes is None:
        return

    face = faces[0]
    eye = [eyes[0], eyes[1]]

    moveMatrix, rotMatrix = matrixPicture(face, eye, height, width)

    dst = cv2.warpAffine(img, moveMatrix, (width, height))
    dst = cv2.warpAffine(dst, rotMatrix, (width, height))

    """
    try:
        cv2.imshow('face2gif', dst)
        cv2.waitKey(0)
    except (KeyboardInterrupt):
        cv2.destroyAllWindows()
        print("User pressed Ctrl+C")

    cv2.destroyAllWindows()
    """
    # cv2.imwrite(DEST_DIR + os.path.basename(file), dst)
    # if faces is not None and eyes is not None:
    # if animation.isOpened():
    animation.write(dst)
    # else:
    #     print("Skipped picture")

    # return dst


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
    files = checkInput()
    # x264
    fourcc = cv2.cv.CV_FOURCC(*"XVID")
    fps = 24.0
    height, width, channel = cv2.imread(files[0]).shape
    global animation
    animation = cv2.VideoWriter("animation.mov", fourcc, fps, (height, width))
    while animation.isOpened:
        for file in files:
            dst = calculatePicture(file)
            animation.write(dst)
        cv2.destroyAllWindows()
        break
    animation.release()
    print("just do: 'convert -delay 10 -loop 0 tmp/*.jpeg animation.gif'")
