import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def readImage(img):
    img = cv.imread(img, 0)
    img = cv.resize(img, (500, 500))
    return img


def laplacianFilter(img):
    kernel = np.matrix([[0.0, -1.0, 0.0],
                        [-1.0, 5.0, -1.0],
                        [0.0, -1.0, 0.0]])
    filteredImage = cv.filter2D(img, -1, kernel)
    stacked = np.hstack((img, filteredImage))
    cv.imshow('stacked', stacked)
    return img


def horizontalSobel(img):
    kernel = np.matrix([[1.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [-1.0, -2.0, -1.0]])
    filteredImage = cv.filter2D(img, -1, kernel)
    stacked = np.hstack((img, filteredImage))
    cv.imshow('stacked', stacked)
    return img


def verticalSobel(img):
    kernel = np.matrix([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]])
    filteredImage = cv.filter2D(img, -1, kernel)
    stacked = np.hstack((img, filteredImage))
    cv.imshow('stacked', stacked)
    return img


def cleanImage(img, kernelSize):
    # Using histogram equalization
    equ = cv.equalizeHist(img)

    # Using a high pass mask
    highPassKernel = np.matrix([[-1.0, -1.0, -1.0],
                                [-1.0, 9.0, -1.0],
                                [-1.0, -1.0, -1.0]])*1.0
    highPassPicture = cv.filter2D(equ, -1, highPassKernel)
    # Using a laplacian filter
    laplacianKernel = np.matrix([[0.0, -1.0, 0.0],
                                 [-1.0, 5.0, -1.0],
                                 [0.0, -1.0, 0.0]])
    laplacianPicture = cv.filter2D(highPassPicture, -1, laplacianKernel)
    # Remove the "salt and pepper"
    filteredImage = cv.medianBlur(laplacianPicture, kernelSize)

    # Put the picture next to the original one for comparison
    stacked = np.hstack((img, filteredImage))
    cv.imshow('stacked', stacked)
    return img


img = readImage('city.jpg')
laplacianFilter(img)

while 1:
    key = cv.waitKey(0)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    if key == ord('1'):
        laplacianFilter(img)
    if key == ord('2'):
        horizontalSobel(img)
    if key == ord('3'):
        verticalSobel(img)
    if key == ord('4'):
        img = readImage('face.jpg')
        img = cleanImage(img, 7)
