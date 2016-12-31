import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def main():
    # Load an color image in grayscale
    img = cv2.imread("circ32.tif",0)
    
    inputImg = np.asarray(img, dtype=np.uint8)
    inputImg_V = inputImg[0::2,0::2]
    
    DFT2DImg = np.empty([inputImg.shape[0], inputImg.shape[1]], dtype=np.complex)
    IDFT2DImg = np.empty([DFT2DImg.shape[0], DFT2DImg.shape[1]], dtype=np.complex)
    
    DFT2DImg = DFT2D(inputImg,True)
    IDFT2DImg = DFT2D(DFT2DImg,False) 
    
    DFTImage_1 = ScaleSpectrum(DFT2DImg, True)
    IDFTImage_1 = ScaleSpectrum(IDFT2DImg, False)
    
    MSEValue = MSE(inputImg, IDFTImage_1)
    print 'MSE : %f' %MSEValue
    
    plt.subplot(131),plt.imshow(inputImg, cmap = 'gray')
    plt.title('Input Image'), plt.xticks(), plt.yticks()
    plt.subplot(132),plt.imshow(DFTImage_1, cmap = 'gray')
    plt.title('DFT Image'), plt.xticks(), plt.yticks()
    plt.subplot(133),plt.imshow(IDFTImage_1, cmap = 'gray')
    plt.title('IDFT Image'), plt.xticks(), plt.yticks()
        
    plt.show()
    
def DFT2D(img,isDFT):
    height = img.shape[0]
    width = img.shape[1]
    
    pmat = np.empty([height, width], dtype=np.complex)
    DFT2DImg = np.empty([height, width], dtype=np.complex)
    
    if(isDFT):
        twoPiH = (-2j * math.pi)/height
        twoPiW = (-2j * math.pi)/width
        factorH = float(1)
        factorW = float(1)
    else:
        twoPiH = (2j * math.pi)/height
        twoPiW = (2j * math.pi)/width
        factorH = float(1)/(height)
        factorW = float(1)/(width)
    
    for k in xrange(height):
        for b in xrange(width):
            f = 0 + 0j
            m = 0 + 0j
            for a in xrange(height):
                m = np.exp(twoPiH * k * a)
                f = f + img[a][b] * m 
            pmat[k][b] = f*factorH
    
    for k in xrange(height):
        for l in xrange(width):
            f = 0 + 0j
            m = 0 + 0j
            for b in xrange(width):
                m = np.exp(twoPiW * l * b)
                f = f + pmat[k][b] * m
            DFT2DImg[k][l] = f*factorW
    
    return DFT2DImg
    
def ScaleSpectrum(inputImg, isShift):
    
    magnitude = np.empty([inputImg.shape[0], inputImg.shape[1]], dtype=np.float)
    
    for i in xrange(inputImg.shape[0]):
        for j in xrange(inputImg.shape[1]):
            magnitude[i][j] = math.sqrt((inputImg[i][j].real**2) + (inputImg[i][j].imag**2))

    if isShift:
        LogSpectrum = np.log(magnitude)
        Shift(LogSpectrum, LogSpectrum)
    else:
        LogSpectrum = magnitude
        
    return LogSpectrum
    

def Shift(inputImg, outputImg):

    temp = np.empty(inputImg.shape, inputImg.dtype)

    height, width = inputImg.shape[:2]

    cx1 = cx2 = width/2
    cy1 = cy2 = height/2

    if width % 2 != 0:
        cx2 += 1
    if height % 2 != 0:
        cy2 += 1

    # swap q1 and q3
    temp[height-cy1:, width-cx1:] = inputImg[0:cy1 , 0:cx1 ]
    temp[0:cy2 , 0:cx2 ] = inputImg[height-cy2:, width-cx2:]

    # swap q2 and q4
    temp[0:cy2 , width-cx2:]  = inputImg[height-cy2:, 0:cx2 ] 
    temp[height-cy1:, 0:cx1 ] = inputImg[0:cy1 , width-cx1:]

    outputImg[:,:] = temp

    return outputImg
    
def MSE(inputImg, outputImg):
    mse = 0
    for i in xrange(inputImg.shape[0]):
        for j in xrange(inputImg.shape[1]):
            mse += math.pow((inputImg[i][j] - outputImg[i][j]), 2)
            
    return mse
    
main()



















