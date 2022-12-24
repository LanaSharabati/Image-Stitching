import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
# img = cv.imread('Fr01.JPG')
# img2 = cv.imread('Fr02.JPG')
img = cv.imread('eifel.JPG')
img2 = cv.imread('eifel02.JPG')
# img = cv.imread('Jerusalem01.JPG')
# img2 = cv.imread('Jerusalem02.JPG')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

# plt.imshow(gray)
# plt.imshow(gray2)
up1 =gray[0]
down1 = gray[len(gray)-1]
left1 =[]
right1=[]
up2 =gray2[0]
down2 = gray2[len(gray2)-1]

left2 =[]
right2=[]

for i in range(len(gray)):
    left1.append(gray[i][0])
    right1.append(gray[i][len(gray[i])-1])
    
for i in range(len(gray2)):
    left2.append(gray2[i][0])
    right2.append(gray2[i][len(gray2[i])-1])



# cosine = np.dot(up1,down2)/(norm(up1)*norm(down2))
# cosine2 = np.dot(up2,down1)/(norm(up2)*norm(down1)) 
# cosine3 = np.dot(np.array(right1),np.array(left2))/(norm(np.array(right1))*norm(np.array(left2))) 
# cosine4 = np.dot(np.array(left1),np.array(right2))/(norm(np.array(right2))*norm(np.array(left1)))     

# cos = np.dot(up1,down2) / (np.sqrt(np.dot(up1,up1)) * np.sqrt(np.dot(down2,down2)))
# cos2 = np.dot(up2,down1) / (np.sqrt(np.dot(up2,up2)) * np.sqrt(np.dot(down1,down1)))
# cos3 = np.dot(np.array(right2),np.array(left1)) / (np.sqrt(np.dot(np.array(right2),np.array(right2))) * np.sqrt(np.dot(np.array(left1),np.array(left1))))
# cos4 = np.dot(np.array(right1),np.array(left2)) / (np.sqrt(np.dot(np.array(right1),np.array(right1))) * np.sqrt(np.dot(np.array(left2),np.array(left2))))

cosine = up1 - down2
cosine2 = up2-down1 
# cosine3 = np.array(right1) - np.array(left2)
# cosine4 = np.array(right2) - np.array(left1) 
cosine3 = 0
cosine4 = 0
f = np.where(cosine == 0)[0]
f2 = np.where(cosine2 == 0)[0]
f3 = np.where(cosine3 == 0)[0]
f4 = np.where(cosine4 == 0)[0]

co = sum((up1 - down2)**2)/len(up1)
co2 = sum((up2 - down1)**2)/len(up2)
# co3 = sum((np.array(right1) - np.array(left2))**2)/len(right1)
# co4 = sum((np.array(right2) - np.array(left1))**2)/len(right2) 


e =[len(f),len(f2),len(f3),len(f4)] 

ind = e.index( max(e)) 
if  ind == 2: ##jur $ efil
    first = img
    second = img2
    n =1
elif ind == 0 :
    first = img2
    second = img
    n =0
elif ind == 1 :
    first = img
    second = img2
    n =0
elif ind == 3 :
    first = img2
    second = img
    n =1
  




vis = np.concatenate((first, second ), axis=n)
cv.imwrite('out.png', vis)
