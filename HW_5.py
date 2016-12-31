import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def problemA(img,kernel):
    output=np.zeros(img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]
    
    kCenterX = kCols / 2;
    kCenterY = kRows / 2;
    
    for i in range(rows):           
        for j in range(cols):       
            max = 0
            for m in range(i-kCenterX,i+kCenterX+1):       
                for n in range(j-kCenterY,j+kCenterY+1):
                    if(m>=0 and m<rows and n>=0 and n<cols):
                        if(img[m][n]>max):
                            max = img[m][n]
            output[i][j]=max
    return output

def problemB(img,kernel):
    output=np.zeros(img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]
    
    kCenterX = kCols / 2;
    kCenterY = kRows / 2;
    
    for i in range(rows):           
        for j in range(cols):       
            min = 10
            for m in range(i-kCenterX,i+kCenterX+1):       
                for n in range(j-kCenterY,j+kCenterY+1):
                    if(m>=0 and m<rows and n>=0 and n<cols):
                        if(img[m][n]<min):
                            min = img[m][n]
            output[i][j]=min
    return output
    
def problemC(img,kernel):
    output=np.zeros(img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    kRows = kernel.shape[0]
    kCols = kernel.shape[1]
    
    kCenterX = kCols / 2;
    kCenterY = kRows / 2;
    
    for i in range(kCenterX,rows-kCenterX):           
        for j in range(kCenterY,cols-kCenterY):       
            match = True
            for m in range(0,kRows):       
                for n in range(0,kCols):
                    if(kernel[m][n]==9):
                        continue
                    if(kernel[m][n]!=img[i-kCenterX+m][j-kCenterY+n]):
                        match = False
                        break
                if(match==False):
                    break
            if(match):
                output[i][j]=1
            else:
                output[i][j]=0
                
    return output

def main():
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    img = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0],
                    [0,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0],
                    [0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0],
                    [0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0],
                    [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                    [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
                    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                    [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
                    
    kernel1 = np.array([[9,1,9],
                        [0,1,1],
                        [0,0,9]])
    kernel2 = np.array([[9,1,9],
                        [1,1,0],
                        [9,0,0]])
    kernel3 = np.array([[9,0,0],
                        [1,1,0],
                        [9,1,9]])
    kernel4 = np.array([[0,0,9],
                        [0,1,1],
                        [9,1,9]])

    
    outputA = problemA(img,kernel)
    outputB = problemB(img,kernel)
    outputC = problemC(img,kernel1)
    outputC = np.logical_or(outputC,problemC(img,kernel2))*1
    outputC = np.logical_or(outputC,problemC(img,kernel3))*1
    outputC = np.logical_or(outputC,problemC(img,kernel4))*1
    
    print "Solution A"
    print outputA
    print "Solution B"
    print outputB
    print "Solution C"
    print outputC

main()






























