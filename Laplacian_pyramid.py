import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def main():
    
    kernel = 1.0/16 *np.array([[1, 2, 1],
                                [1, 4, 2],
                                [1, 2, 1]])
    
    img = cv2.imread("circ32.tif",0)
    img = np.asarray(img, dtype=np.uint8)    
    G0 = convolve(img,kernel)
    G1 = convolve(downSample(G0),kernel)
    G2 = convolve(downSample(G1),kernel)
    G3 = convolve(downSample(G2),kernel)
    G4 = convolve(downSample(G3),kernel)
    G5 = convolve(downSample(G4),kernel)
    
    L4 = G4-upSample(G5)
    L3 = G3-upSample(G4)
    L2 = G2-upSample(G3)
    L1 = G1-upSample(G2)
    L0 = G0-upSample(G1)
    
    R0 = L0+upSample(G1)
    
    MSEValue = MSE(img, R0)
    print 'MSE : %f' %MSEValue
    
    plt.subplot(171),plt.imshow(img, cmap = 'gray')
    plt.title('Original'), plt.xticks(), plt.yticks()
    plt.subplot(172),plt.imshow(L0, cmap = 'gray')
    plt.title('L0'), plt.xticks(), plt.yticks()
    plt.subplot(173),plt.imshow(L1, cmap = 'gray')
    plt.title('L1'), plt.xticks(), plt.yticks()
    plt.subplot(174),plt.imshow(L2, cmap = 'gray')
    plt.title('L2'), plt.xticks(), plt.yticks()
    plt.subplot(175),plt.imshow(L3, cmap = 'gray')
    plt.title('L3'), plt.xticks(), plt.yticks()
    plt.subplot(176),plt.imshow(L4, cmap = 'gray')
    plt.title('L4'), plt.xticks(), plt.yticks()
    plt.subplot(177),plt.imshow(R0, cmap = 'gray')
    plt.title('Recovered'), plt.xticks(), plt.yticks()
    
    plt.show()
    
def convolve(img, kernel):
    
    rows = img.shape[0]
    cols = img.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]
    
    kCenterX = kCols / 2;
    kCenterY = kRows / 2;
    
    output = np.zeros((img.shape[0], img.shape[1]))

    for i in range(rows):           
        for j in range(cols):       
            for m in range(kRows):    
                mm = kRows - 1 - m;      
                for n in range(kCols):
                    nn = kCols - 1 - n
                    ii = i + (m - kCenterY)
                    jj = j + (n - kCenterX)
                    if( ii >= 0 and ii < rows and jj >= 0 and jj < cols ):
                        output[i][j] += img[ii][jj] * kernel[mm][nn]
    return output

def downSample(img):
    return img[::2, ::2]


def upSample(img):
    rows = img.shape[0]
    cols = img.shape[1]
    for i in xrange(rows):
        img = np.insert(img, 2*i+1,np.array(img[2*i,:]), 0) 

    for j in xrange(cols):
        img = np.insert(img, 2*j+1, np.array(img[:,2*j]), 1) 
    
    return img

def MSE(inputImg, outputImg):
    mse = 0
    for i in xrange(inputImg.shape[0]):
        for j in xrange(inputImg.shape[1]):
            mse += math.pow((inputImg[i][j] - outputImg[i][j]), 2)
            
    return mse

main()



















