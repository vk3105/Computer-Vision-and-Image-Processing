import numpy as np
from matplotlib import pyplot as plt

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

def problem2():
    kernelLowPass = 1.0/5 *np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]])
        
    signal =  np.array([[0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,0,0,0,0],
                        [0,0,0,1,1,1,1,0,0,0],
                        [0,0,1,1,1,1,1,1,0,0],
                        [0,0,1,1,1,1,1,1,1,0],
                        [0,0,1,1,1,1,1,1,1,0],
                        [0,0,0,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,1,0,0,0]])
    
    threshold = 0.5
    
    output1 = convolve(signal,kernelLowPass,threshold,False)
    print "Problem 2 without Threshold"
    print output1
    
    output2 = convolve(signal,kernelLowPass,threshold,True)
    print "Problem 2 with Threshold"
    print output2
    
    #plt.subplot(131),plt.imshow(signal, cmap = 'gray')
    #plt.title('Original'), plt.xticks(), plt.yticks()
    #plt.subplot(132),plt.imshow(output1, cmap = 'gray')
    #plt.title('Outpu1'), plt.xticks(), plt.yticks()
    #plt.subplot(133),plt.imshow(output2, cmap = 'gray')
    #plt.title('Output2'), plt.xticks(), plt.yticks()
    #plt.show()

def problem3():
    kernelLaplace = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
        
    signal =  np.array([[0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,1,1,0,0],
                        [0,0,0,0,0,1,1,1,0,0],
                        [0,0,0,0,1,1,1,1,0,0],
                        [0,0,0,1,1,1,1,1,0,0],
                        [0,0,1,1,1,1,1,1,0,0],
                        [0,1,1,1,1,1,1,1,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]])
    
    threshold = 0.0
    
    output1 = convolve(signal,kernelLaplace,threshold,False)
    print "Problem 3"
    print output1
    
    #plt.subplot(121),plt.imshow(signal, cmap = 'gray')
    #plt.title('Original'), plt.xticks(), plt.yticks()
    #plt.subplot(122),plt.imshow(output1, cmap = 'gray')
    #plt.title('Outpu1'), plt.xticks(), plt.yticks()
    #plt.show()

problem2()
problem3()













