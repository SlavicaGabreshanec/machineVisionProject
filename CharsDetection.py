import os

import cv2
import numpy as np
import math
import random

import Main

import Char


kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80


MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100
#for preprocess
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(originalPhoto):
    height, width, numberOfChannels = originalPhoto.shape

    photoHSV = np.zeros((height, width, 3), np.uint8)
    photoHSV = cv2.cvtColor(originalPhoto, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(photoHSV)
    #grayscale
    photoGrayscale = value

    height, width = photoGrayscale.shape

    photoTopHat = np.zeros((height, width, 1), np.uint8)
    photoBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    photoTopHat = cv2.morphologyEx(photoGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    photoBlackHat = cv2.morphologyEx(photoGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    photoGrayscalePlusTopHat = cv2.add(photoGrayscale, photoTopHat)
    photoGrayscalePlusTopHatMinusBlackHat = cv2.subtract(photoGrayscalePlusTopHat, photoBlackHat)

    #max contrast
    photoMaxContrastGrayscale = photoGrayscalePlusTopHatMinusBlackHat

    height, width = photoGrayscale.shape

    photoBlurred = np.zeros((height, width, 1), np.uint8)

    photoBlurred = cv2.GaussianBlur(photoMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    photoThresh = cv2.adaptiveThreshold(photoBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return photoGrayscale, photoThresh

def loadDataAndTrainKNN():
    # read in training classifications
    try:
        classifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("Error: unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False
        # read in training images
    try:
        trainingImages = np.loadtxt("training_images.txt", np.float32)
    except:
        print("Error: unable to open training_images.txt, exiting program\n")
        os.system("pause")
        return False

    classifications = classifications.reshape((classifications.size, 1))
    kNearest.setDefaultK(1)
    # train KNN object
    kNearest.train(trainingImages, cv2.ml.ROW_SAMPLE, classifications)

    return True

def detectCharsInPlates(possiblePlates):
    intPlateCounter = 0
    contours = []

    if len(possiblePlates) == 0:
        return possiblePlates

    for possiblePlate in possiblePlates:
        possiblePlate.imgGrayscale, possiblePlate.photoThresh = preprocess(possiblePlate.imgPlate)

        # increase size of plate image for easier viewing and char detection
        possiblePlate.photoThresh = cv2.resize(possiblePlate.photoThresh, (0, 0), fx = 1.6, fy = 1.6)

         # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.photoThresh = cv2.threshold(possiblePlate.photoThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find all possible chars in the plate
        possibleCharsPlate = findPossibleCharsInPlate(possiblePlate.photoThresh)

        height, width, numChannels = possiblePlate.imgPlate.shape
        photoContours = np.zeros((height, width, 3), np.uint8)
        del contours[:]

        for possibleChar in possibleCharsPlate:
            contours.append(possibleChar.contour)

        # given a list of all possible chars, find groups of matching chars within the plate
        listsOfMatchingCharsPlate = findListOfListsOfMatchingChars(possibleCharsPlate)


        imgContours = np.zeros((height, width, 3), np.uint8)
        del contours[:]

        for listOfMatchingChars in listsOfMatchingCharsPlate:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

        if (len(listsOfMatchingCharsPlate) == 0):
            intPlateCounter = intPlateCounter + 1
            cv2.destroyWindow("8")
            cv2.destroyWindow("9")
            cv2.destroyWindow("10")
            cv2.waitKey(0)

            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listsOfMatchingCharsPlate)):
            listsOfMatchingCharsPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)
    #tukaa
        for listOfMatchingChars in listsOfMatchingCharsPlate:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            del contours[:]

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listsOfMatchingCharsPlate)):
            if len(listsOfMatchingCharsPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listsOfMatchingCharsPlate[i])
                intIndexOfLongestListOfChars = i


        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listsOfMatchingCharsPlate[intIndexOfLongestListOfChars]

        imgContours = np.zeros((height, width, 3), np.uint8)
        del contours[:]

        for matchingChar in longestListOfMatchingCharsInPlate:
            contours.append(matchingChar.contour)

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.photoThresh, longestListOfMatchingCharsInPlate)

        intPlateCounter = intPlateCounter + 1
        cv2.waitKey(0)
    return possiblePlates

def findPossibleCharsInPlate(photoThresh):
    listOfPossibleChars = []
    photoThreshCopy = photoThresh.copy()

    # find all contours in plate
    contours, hierarchy = cv2.findContours(photoThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = Char.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars

#a rough check on a contour to see if it could be a char
def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
         possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

#a list of lists of matching chars
def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    return listOfListsOfMatchingChars

#given a possible char and a big list of possible chars
def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue

        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    adj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    opp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
    # check to make sure we do not divide by zero if the center X positions are equal
    if adj != 0.0:
        angleInRad = math.atan(opp / adj)
    else:
        angleInRad = 1.5708

    angleInDeg = angleInRad * (180.0 / math.pi)

    return angleInDeg

# this is where we apply the actual char recognition
def recognizeCharsInPlate(photoThresh, listOfMatchingChars):
    strChars = ""

    height, width = photoThresh.shape

    photoThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    cv2.cvtColor(photoThresh, cv2.COLOR_GRAY2BGR, photoThreshColor)

    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(photoThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)
        # crop char out of threshold image
        photo = photoThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        photoResized = cv2.resize(photo, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
        resized = photoResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))

        resized = np.float32(resized)
        retval, results, neigh_resp, dists = kNearest.findNearest(resized, k = 1)

        strCurrentChar = str(chr(int(results[0][0])))

        strChars = strChars + strCurrentChar
    return strChars









