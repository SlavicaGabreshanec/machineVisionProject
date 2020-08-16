# Plate.py

import cv2
import numpy as np

class Plate:
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.photoThresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars = ""




