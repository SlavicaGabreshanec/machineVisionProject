import cv2
import numpy as np
import math
import Main
import random


import CharsDetection
import Plate
import Char


PLATE_WIDTH_PADDING_FACTOR = 1.5
PLATE_HEIGHT_PADDING_FACTOR = 1.9

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

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

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


def detectPlatesInPhoto(originalPhoto):
    listOfPossiblePlates = []

    height, width, numChannels = originalPhoto.shape

    photoGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    photoThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    # preprocess to get grayscale and threshold images
    photoGrayscaleScene, photoThresh = preprocess(originalPhoto)

    # find all possible chars in the scene,
    listPossibleCharsPlate = findPossibleCharsInPhoto(photoThresh)


    imgContours = np.zeros((height, width, 3), np.uint8)

    contours = []

    for possibleChar in listPossibleCharsPlate:
        contours.append(possibleChar.contour)

    # given a list of all possible chars, find groups of matching chars
    listOfListsOfMatchingCharsInScene = CharsDetection.findListOfListsOfMatchingChars(listPossibleCharsPlate)

    imgContours = np.zeros((height, width, 3), np.uint8)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        intRandomBlue = random.randint(0, 255)
        intRandomGreen = random.randint(0, 255)
        intRandomRed = random.randint(0, 255)

        contours = []

        for matchingChar in listOfMatchingChars:
            contours.append(matchingChar.contour)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = getPlate(originalPhoto, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)

    for i in range(0, len(listOfPossiblePlates)):
        p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

        cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
        cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
        cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
        cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

        cv2.waitKey(0)

    return listOfPossiblePlates

def findPossibleCharsInPhoto(imgThresh):
    listOfPossibleChars = []

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):
        possibleChar = Char.PossibleChar(contours[i])

        if CharsDetection.checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars

def getPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = Plate.Plate()

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight


    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = CharsDetection.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped

    return possiblePlate