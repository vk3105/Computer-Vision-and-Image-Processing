import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

range_fn = lambda start, end: range(start, end+1)

def zeroCrossingFunc(img):
    zcImg = np.zeros(img.shape)
    for i in range(1,img.shape[0]-1):
	for j in range(1,img.shape[1]-1):
	   neg = 0
	   pos = 0
	   for a in range_fn(-1, 1):
	       for b in range_fn(-1,1):
	           if(a != 0 and b != 0):
			if(img[i+a,j+b] < 0):
	                   neg += 1
			elif(img[i+a,j+b] > 0):
	                   pos += 1
	   z_c = ( (neg > 0) and (pos > 0) )
	   if(z_c):
	       zcImg[i,j] = 0
	   else:
	       zcImg[i,j] = 255
    return zcImg

def convolve(img, kernel,threshold,applyThreshold):
    
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
            if(applyThreshold):
                if(output[i][j] >= threshold):
                    output[i][j] = 1
                else:
                    output[i][j] = 0
    return output
    
def zeroCrossing(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    output = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(rows):
        for j in range(cols-1):
            if(img[i][j]*img[i][j+1]<0):
                output[i][j+1] = 1
    
    for i in range(rows-1):
        for j in range(cols):
            if(img[i][j]*img[i+1][j]<0):
                output[i+1][j] = 1
    
    return output

def gradient(x,y):
    x1 = x**2
    y1 = y**2
    res = x1+y1
    resfinal = np.sqrt(res)
    
    return resfinal

def main():
    img = cv2.imread("ub.jpeg",0)
    #img = cv2.imread("circ32.tif",0)
    img = np.asarray(img, dtype=np.uint8)
    
    kernel = np.array([[0 , 0,-1,-1,-1, 0, 0],
                       [0 ,-2,-3,-3,-3,-2, 0],
                       [-1,-3, 5, 5, 5,-3,-1],
                       [-1,-3, 5,16, 5,-3,-1],
                       [-1,-3, 5, 5, 5,-3,-1],
                       [0 ,-2,-3,-3,-3,-2, 0],
                       [0 , 0,-1,-1,-1, 0, 0]])
    
    sobelx = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
    sobely = np.array([[1 , 2, 1],
                       [0 , 0, 0],
                       [-1,-2,-1]])
    
    DoG = convolve(img,kernel,0.0,False)
    
    zeroCrossingImg = zeroCrossingFunc(DoG)
    zeroCrossingImg1 = zeroCrossing(DoG)
    
    sobelxCon = convolve(zeroCrossingImg1,sobelx,0.0,False)
    
    sobelyCon = convolve(zeroCrossingImg1,sobely,0.0,False)
    
    gradValue = gradient(sobelxCon,sobelyCon)
    
    maxi = np.max(gradValue)
    mean = np.mean(gradValue)
    std = np.std(gradValue)
    print maxi
    print mean
    print std
    print mean-std
    print mean+std
    print np.median(gradValue)
    
    grad = np.where(gradValue == maxi , 1 , 0)
    print grad
    
    res1 = np.where((zeroCrossingImg1-grad)==0,1,0)
    
    res = np.logical_not(np.logical_and(grad, zeroCrossingImg1))*1
    
    #plt.subplot(131),plt.imshow(img, cmap = 'gray')
    #plt.title('Original'), plt.xticks(), plt.yticks()
    #plt.subplot(132),plt.imshow(DoG, cmap = 'gray')
    #plt.title('DoG'), plt.xticks(), plt.yticks()
    plt.subplot(121),plt.imshow(res1, cmap = 'gray')
    plt.title('ZC'), plt.xticks(), plt.yticks()
    plt.subplot(122),plt.imshow(res, cmap = 'gray')
    plt.title('Grad'), plt.xticks(), plt.yticks()
    plt.show()
    
main()
