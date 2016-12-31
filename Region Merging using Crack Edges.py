import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def addCrackEdges(img,threshold):
    rows = img.shape[0]
    cols = img.shape[1]
    crackEdgeImg = np.zeros((2*rows+1, 2*cols+1))
    
    crackEdgeImg.setflags(write=True)
    
    for i in range(0,rows):  
        for j in range(1,cols):
            if(img[i][j-1]>=img[i][j]):
                diff = img[i][j-1]-img[i][j]
            else:
                diff = img[i][j]-img[i][j-1]
            if(diff<threshold):
                crackEdgeImg[2*i+1][2*j] = 0
            else:
                crackEdgeImg[2*i+1][2*j] = 1
    
    for j in range(0,cols):  
        for i in range(1,rows):
            if(img[i-1][j]>=img[i][j]):
                diff = img[i-1][j]-img[i][j]
            else:
                diff = img[i][j]-img[i-1][j]
            if(diff<threshold):
                crackEdgeImg[2*i][2*j+1] = 0
            else:
                crackEdgeImg[2*i][2*j+1] = 1
    
    for i in range(0,rows):  
        for j in range(0,cols):
            crackEdgeImg[2*i+1][2*j+1] = CONST
    
    return crackEdgeImg

def minimumValue(x,y):
    if(x>y):
        return y
    else:
        return x

def mergeRegionsSecond(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols,T2):
    width = commonRegionPerimeter.shape[0]
    height = commonRegionPerimeter.shape[1]
    
    count = 0
    
    for i in range(1,width):  
        for j in range(1,height):
            if(commonRegionPerimeter[i][j]!=0):
                minVal = 1.0*minimumValue(regionsPerimeter[i],regionsPerimeter[j])
                if(minVal>0):
                    weight = commonRegionPerimeter[i][j]/minVal
                    if(weight>=T2):
                        count += 1
                        labeledRegionMat[labeledRegionMat == i] = j
                        regionsPerimeter[j]=regionsPerimeter[i]+regionsPerimeter[j]-2*commonRegionPerimeter[i][j]
                        regionsPerimeter[i] = 0
                        val = commonRegionPerimeter[i][j]
                        commonRegionPerimeter[i][j]=commonRegionPerimeter[j][i]=0
                        for k in range(1,height):
                            if(commonRegionPerimeter[i][k]!=0):
                                commonRegionPerimeter[j][k] = commonRegionPerimeter[k][j] = commonRegionPerimeter[j][k]+val 
                                commonRegionPerimeter[i][k] = commonRegionPerimeter[k][i] = 0
                    
    return count
    
def mergeRegionsThird(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols,T3):
    width = commonRegionPerimeter.shape[0]
    height = commonRegionPerimeter.shape[1]
    
    count =0 
    
    for i in range(1,width):  
        for j in range(1,height):
            if(commonRegionPerimeter[i][j]!=0):
                minVal = 1.0*minimumValue(regionsPerimeter[i],regionsPerimeter[j])
                if(minVal>0):
                    weight = commonRegionPerimeter[i][j]/minVal
                    if(weight>=T3):
                        count += 1
                        #print count
                        labeledRegionMat[labeledRegionMat == i] = j
                        regionsPerimeter[j]=regionsPerimeter[i]+regionsPerimeter[j]-2*commonRegionPerimeter[i][j]
                        regionsPerimeter[i] = 0
                        val = commonRegionPerimeter[i][j]
                        commonRegionPerimeter[i][j]=commonRegionPerimeter[j][i]=0
                        for k in range(1,height):
                            if(commonRegionPerimeter[i][k]!=0):
                                commonRegionPerimeter[j][k] = commonRegionPerimeter[k][j] = commonRegionPerimeter[j][k]+val 
                                commonRegionPerimeter[i][k] = commonRegionPerimeter[k][i] = 0
                    
    return count
    
    
def mapToOriginalImage(labeledRegionMat,img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    width = labeledRegionMat.shape[0]
    height = labeledRegionMat.shape[1]
    
    for i in range(0,rows):  
        for j in range(0,cols):
            if(isValid(2*i+3,2*j+3,width,height)):
                if(labeledRegionMat[2*i+1][2*j+1]!=labeledRegionMat[2*i+3][2*j+3]):
                    img[i][j]=255
    
    return img


def isValid(x,y,rows,cols):
    if(x>0 and x<rows and y>0 and y<cols):
        return True
    else:
        return False
        
def fill(crackEdgeImg,state,rows,cols,x,y,label):
    width = crackEdgeImg.shape[0]
    height = crackEdgeImg.shape[1]
    
    stack = set(((x, y),))
    
    while stack:
        a, b = stack.pop()
        
        if(state[a][b]==0 and isValid(a,b+2,width,height) and isValid(a,b+1,width,height)):
            if(crackEdgeImg[a][b] == crackEdgeImg[a][b+2] and crackEdgeImg[a][b+1]!=1):
                stack.add((a, b+2))

        if(state[a][b]==0 and isValid(a,b-2,width,height) and isValid(a,b-1,width,height)): 
            if(crackEdgeImg[a][b] == crackEdgeImg[a][b-2] and crackEdgeImg[a][b-1]!=1):
                stack.add((a, b-2))
                
        if(state[a][b]==0 and isValid(a+2,b,width,height) and isValid(a+1,b,width,height)): 
            if(crackEdgeImg[a][b] == crackEdgeImg[a+2][b] and crackEdgeImg[a+1][b]!=1):
                stack.add((a+2, b))
                
        if(state[a][b]==0 and isValid(a-2,b,width,height) and isValid(a-1,b,width,height)): 
            if(crackEdgeImg[a][b] == crackEdgeImg[a-2][b] and crackEdgeImg[a-1][b]!=1):
                stack.add((a-2, b))

        state[a][b] = label  
        
def findRegions(crackEdgeImg,state,rows,cols):
    rowsEdge = crackEdgeImg.shape[0]
    colsEdge = crackEdgeImg.shape[1]
    
    label = 1
    
    for i in range(0,rows):  
        for j in range(0,cols):
            if(state[2*i+1][2*j+1]==0):
                fill(crackEdgeImg,state,rowsEdge,colsEdge,2*i+1,2*j+1,label)
                label = label+1
    return label
    
def findPerimeter(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols):
    width = labeledRegionMat.shape[0]
    height = labeledRegionMat.shape[1]
    
    for i in range(0,rows):  
        for j in range(0,cols):
            x = 2*i+1
            y = 2*j+1
            if(isValid(x+2,y,width,height)):
                r1=labeledRegionMat[x][y]
                r2=labeledRegionMat[x+2][y]
                
                if(labeledRegionMat[x][y]!=labeledRegionMat[x+2][y]):
                    commonRegionPerimeter[r1][r2]= commonRegionPerimeter[r1][r2]+1
                    commonRegionPerimeter[r2][r1]= commonRegionPerimeter[r2][r1]+1
                    regionsPerimeter[r1] = regionsPerimeter[r1]+1
            else:
                 regionsPerimeter[r1] = regionsPerimeter[r1]+1       
                    
            if(isValid(x-2,y,width,height)):
                r1=labeledRegionMat[x][y]
                r2=labeledRegionMat[x-2][y]
                
                if(labeledRegionMat[x][y]!=labeledRegionMat[x-2][y]):
                    commonRegionPerimeter[r1][r2]= commonRegionPerimeter[r1][r2]+1
                    commonRegionPerimeter[r2][r1]= commonRegionPerimeter[r2][r1]+1
                    regionsPerimeter[r1] = regionsPerimeter[r1]+1
            else:
                 regionsPerimeter[r1] = regionsPerimeter[r1]+1 
                  
            if(isValid(x,y-2,width,height)): 
                r1=labeledRegionMat[x][y]
                r2=labeledRegionMat[x][y-2]
                
                if(labeledRegionMat[x][y]!=labeledRegionMat[x][y-2]):
                    commonRegionPerimeter[r1][r2]= commonRegionPerimeter[r1][r2]+1
                    commonRegionPerimeter[r2][r1]= commonRegionPerimeter[r2][r1]+1
                    regionsPerimeter[r1] = regionsPerimeter[r1]+1
            else:
                 regionsPerimeter[r1] = regionsPerimeter[r1]+1 
                 
            if(isValid(x,y+2,width,height)):    
                r1=labeledRegionMat[x][y]
                r2=labeledRegionMat[x][y+2]
                
                if(labeledRegionMat[x][y]!=labeledRegionMat[x][y+2]):
                    commonRegionPerimeter[r1][r2]= commonRegionPerimeter[r1][r2]+1
                    commonRegionPerimeter[r2][r1]= commonRegionPerimeter[r2][r1]+1
                    regionsPerimeter[r1] = regionsPerimeter[r1]+1
            else:
                 regionsPerimeter[r1] = regionsPerimeter[r1]+1
                                 
def main(): 
        
    img = cv2.imread("Mixed Vegetables.jpeg",0)
    img = np.asarray(img, dtype=np.uint8)  
              
    rows = img.shape[0]
    cols = img.shape[1]
    
    T1 = 13
    T2 = 0.50
    T3 = 0.25
                       
    crackEdgeImg = addCrackEdges(img,T1)    
    
    labeledRegionMat = np.zeros(crackEdgeImg.shape)
   
    labelCount = findRegions(crackEdgeImg,labeledRegionMat,rows,cols)
      
    regionsPerimeter = np.zeros(labelCount+1)
    
    commonRegionPerimeter = np.zeros((labelCount+1,labelCount+1))
    
    findPerimeter(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols)
        
    commonRegionPerimeter = commonRegionPerimeter/float(2)    
    
    t = 0
    
    while(True):
        count = mergeRegionsSecond(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols,T2)
        if(count==0):
            break
    
    count = 0
    while(True):
        mergeRegionsThird(labeledRegionMat,regionsPerimeter,commonRegionPerimeter,rows,cols,T3)
        if(count==0):
            break
    
    result=mapToOriginalImage(labeledRegionMat,img)
    
    cv2.imwrite("crackEdgeImg.png",labeledRegionMat)    
    
    plt.subplot(111),plt.imshow(result, cmap = 'gray')
    plt.title('result'), plt.xticks(), plt.yticks()
    plt.show()
    
CONST = 255
main()
