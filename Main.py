import numpy as np
import cv2
import os

import CharsDetection
import PlatesDetection


SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)

showSteps = False

def main():

    #load and train data with K-Nearest Neighbours algoritam
    KNNTrainingSucc = CharsDetection.loadDataAndTrainKNN()
    if KNNTrainingSucc == False:
        print("Error: KNN training was not successful!")
        return

    originalPhoto = cv2.imread("LicencePlatePhotos/2.png")
    if originalPhoto is None:
        print("Image not read from file \n")
        os.system("pause")
        return

    possiblePlates = PlatesDetection.detectPlatesInPhoto(originalPhoto)
    possiblePlates = CharsDetection.detectCharsInPlates(possiblePlates)

    cv2.imshow("originalPhoto", originalPhoto)

    if len(possiblePlates) == 0:
        print("No license plates were detected\n")
    else:
        #sort the list of possible plates in descending order
        possiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        licencePlate = possiblePlates[0]
        #show the crop of plate
        cv2.imshow("LicencePlate", licencePlate.imgPlate)

        if len(licencePlate.strChars) == 0:
            print("No characters were detected\n")
            return

        drawRectangleAroundPlate(originalPhoto, licencePlate)
        cv2.imshow("originalPhoto", originalPhoto)
        cv2.imwrite("originalPhoto.png", originalPhoto)
    cv2.waitKey(0)

    return

def drawRectangleAroundPlate(originalPhoto, plate):

    p2fRectPoints = cv2.boxPoints(plate.rrLocationOfPlateInScene)

    cv2.line(originalPhoto, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(originalPhoto, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(originalPhoto, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(originalPhoto, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

if __name__ == "__main__":
    main()


















