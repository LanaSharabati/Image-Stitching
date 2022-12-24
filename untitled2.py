import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

class stitching:
    def __init__(self,img1,img2):
        
        self.image1 = img1
        self.image2 = img2
    
    
    def edges_detect(self,img):
        """
        detect the image edges right , left ,up ,down

        """
        up =img[0]
        down = img[len(img)-1]
        left =[]
        right=[]
        for i in range(len(img)):
            left.append(img[i][0])
            right.append(img[i][len(img[i])-1])
        return(up,down, np.array(right), np.array(left))


    def  mean_square_error(self,list1,list2):
        """
        calculate the mean square erro to find 
        the best place for image stitching
        the up edges for first image with down edge foe second image viavesr
        the left edge for first with righr with second
        
        """
        mqe = sum((list1 - list2)**2)/len(list1)
        return mqe


    def image_stitching(self,lis,img1,img2):
        """
        compute the minumum value of mean square error for all edges
        to detect the side nees to stitching
        """
        
        ind = lis.index(min(lis))
        
        if  ind == 2:
            first = img1
            second = img2
            n =1
        elif ind == 0 :
            first = img2
            second = img1
            n =0
        elif ind == 1 :
            first = img1
            second = img2
            n =0
        elif ind == 3 :
            first = img2
            second = img1
            n =1
        con = np.concatenate((first, second ), axis=n)
        cv.imwrite('out.png', con)
        plt.imshow(con)
        plt.figure() 
        
        
def image_list():
    """
    Return a list images in test file
    """
    # os.chdir(folder_name) 
    images = glob.glob('*.jpg')
    return images


def read(image):
    '''
    take list of images name and read them

    '''
    img = cv.imread(image)
    
    return img


    
images = image_list()

for i in range(0,len(images),2):
    
    img1 = read(images[i])#read the first half
    img2 = read(images[i+1])#read the second half
    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)#conver images to grayscale
    gray2 =cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    class1 = stitching(img1, img2)
    edges1 =  class1.edges_detect(gray1)
    edges2 =  class1.edges_detect(gray2)
    mean_values =[]
    for i in range(0,len(edges1),2):
        if len(edges1[i]) == len(edges2[i]):
            error = class1.mean_square_error(edges1[i],edges2[i+1])
            mean_values.append(error)
            error = class1.mean_square_error(edges1[i+1],edges2[i])
            mean_values.append(error)
        else:
            error = 1000000
            mean_values.append(error)
    print(mean_values)
    class1.image_stitching(mean_values,img1,img2)
        
            
    
   
    
    
    
    
    
    
