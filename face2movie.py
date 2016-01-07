#!/usr/bin/env python2.7
import sys
import os.path
from math import atan, pi
from optparse import OptionParser

try:
    import numpy as np
except:
    sys.exit("Please install numpy")

try:
    import cv2
except:
    sys.exit("Please install OpenCV")

# Parser
parser = OptionParser()
parser.add_option("-i", "--imagefolder", type="string", dest="imagefolder",
                  help="Path of images")
parser.add_option("-s", "--facescale", type="string", dest="facescale",
                  help="scale of the face (default is 1/3)")
parser.add_option("-f", "--fps", type="string", dest="fps",
                  help="fps of the resulting file (default is 24)")
parser.add_option("-n", "--nameoftargetfile", type="string", dest="outputfile",
                  help="name of the output file")
parser.add_option("-w", "--write", action="store_true", dest="write",
                  default=False, help="to write every single image to file")
parser.add_option("-r", "--reverse", action="store_true", dest="reverse",
                  default=False, help="iterate the files reversed")
parser.add_option("-q", "--quiet", action="store_false", dest="quiet",
                  default=True, help="the output should be hidden")
parser.add_option("-m", "--multiplerender", action="store_true",
                  dest="multiplerender", default=False,
                  help="render the images multiple times")

# parsing the input
(options, args) = parser.parse_args()
imagefolder = options.imagefolder
if imagefolder is None:
    sys.exit("No images given")
facescale = options.facescale
if facescale is None:
    facescale = float(1.0 / 3)
else:
    facescale = float(facescale)
fps = float(options.fps)
if fps is None:
    fps = 24
outputfile = options.outputfile
if outputfile is None:
    outputfile = "animation"
write = bool(options.write)
reverse = bool(options.reverse)
quiet = bool(options.quiet)
multiplerender = bool(options.multiplerender)

# OpenCV files
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
    if multiplerender:
        for i in np.arange(1.05, 1.65, 0.05)[::-1]:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=i, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 1:
                return faces
            elif len(faces) > 1:
                return None
            # print(str(i) + "- useless calc:" + str(faces))
        # print("no face found")
        return None
    else:
        return face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))


def detectEye(roi_gray):
    """detecting eyes"""
    if multiplerender:
        for i in np.arange(1.01, 1.10, 0.01)[::-1]:
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=i, minNeighbors=5, minSize=(25, 25))
            if len(eyes) == 2:
                return eyes
            elif len(eyes) > 2:
                return None
            # print(str(i) + "- useless calc:" + str(eyes))
        # print("no eyes found")
        return None
    else:
        return eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(25, 25))


def drawFaces(faces, img):
    """drawing faces (for debug)"""
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)


def drawEyes(eyes, img):
    """drawing eyes (for debug)"""
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)


def detect(img, gray):
    """getting the image and returns the face and eyes"""
    faces = dectectFace(gray)
    # for making sure only having one face
    if faces is None or len(faces) != 1:
        return None, None
    # drawFaces(faces, img)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        eyes = detectEye(roi_gray)

        # making sure only having two eyes
        if eyes is None or len(eyes) != 2:
            return None, None
        # drawEyes(eyes, roi_color)
    return faces, eyes


def matrixPicture(face, eyes, height, width):
    """calculation of rotation and movement of the image"""
    center = tuple((face[0] + (face[2] / 2), face[1] + (face[3] / 2)))

    moveMatrix = np.float32([[1, 0, (width / 2) - center[0]],
                             [0, 1, (height / 2) - center[1]]])

    scale = float(min(height, width)) / float(face[2]) * facescale

    eye1 = tuple((eyes[0][0] + (eyes[0][2] / 2),
                  eyes[0][1] + (eyes[0][3] / 2)))
    eye2 = tuple((eyes[1][0] + (eyes[1][2] / 2),
                  eyes[1][1] + (eyes[1][3] / 2)))

    x = (float(eye2[0]) - float(eye1[0]))
    y = (float(eye2[1]) - float(eye1[1]))

    if x == 0:
        angle = 0
    else:
        angle = atan(y / x) * 180 / pi

    rotMatrix = cv2.getRotationMatrix2D(center, angle, scale)

    return moveMatrix, rotMatrix


def calculatePicture(file):
    """gettings infos of the image and applie the matrixes"""
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, eyes = detect(img, gray)
    # print("faces: " + str(faces) + " # eyes:" + str(eyes))
    height, width, channels = img.shape

    if faces is None or eyes is None:
        return None

    face = faces[0]
    eye = [eyes[0], eyes[1]]

    moveMatrix, rotMatrix = matrixPicture(face, eye, height, width)

    dst = cv2.warpAffine(img, moveMatrix, (width, height))
    dst = cv2.warpAffine(dst, rotMatrix, (width, height))

    return dst


def checkInput():
    """ check input and return files """
    files = []
    if imagefolder:
        for file in os.listdir(imagefolder):
            if os.path.isfile(os.path.join(imagefolder, file)):
                files.append(imagefolder + file)
    if files is [] or not imagefolder.endswith("/"):
        sys.exit("No files found")
    if reverse:
        files.sort(reverse=True)
    else:
        files.sort()
    return files


def toMovie():
    """ iterating the files and save them to movie-file """
    files = checkInput()
    codecs = cv2.cv.CV_FOURCC(*'MP4V')
    height, width, channel = cv2.imread(files[0]).shape

    video = cv2.VideoWriter(outputfile + ".mkv", codecs,
                            fps, (width, height), True)
    if not video.isOpened():
        sys.exit("Error when writing video file")
    images = 0
    found = 0
    for file in files:
        dst = calculatePicture(file)
        images = images + 1
        if quiet:
            sys.stdout.flush()
            sys.stdout.write("\rimages: " + str(images) + "/" +
                             str(len(files)) + " and " + str(found) +
                             " added to movie")
        if dst is not None and video.isOpened():
            found = found + 1
            video.write(dst)
    video.release()
    if quiet:
        print
        print("saved to " + outputfile + ".mkv")


def toFile():
    """ iterating files and save them seperately """
    destdir = os.path.join(os.path.abspath(".") + r"/tmp/")
    import subprocess
    files = checkInput()
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    for file in files:
        dst = calculatePicture(file)
        if dst is not None:
            """
            try:
                cv2.imshow('face2gif', dst)
                cv2.waitKey(0)
            except (KeyboardInterrupt):
                cv2.destroyAllWindows()
            cv2.destroyAllWindows()
            """
            cv2.imwrite(destdir + os.path.basename(file), dst)
    if quiet:
        print("all files are safed in: " + str(destdir))
        print("now generating gif ...")
        print(subprocess.call(["convert", "-delay", fps,
                               "-loop", "0", "tmp/*.jpeg", outputfile + ".gif"]))
    else:
        subprocess.call(["convert", "-delay", fps,
                         "-loop", "0", "tmp/*.jpeg", outputfile + ".gif"])


if __name__ == '__main__':
    if write:
        toFile()
    else:
        toMovie()
