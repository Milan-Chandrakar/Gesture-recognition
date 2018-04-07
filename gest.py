import cv2
import numpy as np

lower = np.array([50,80,40])
upper = np.array([90,255,255])

cam = cv2.VideoCapture(0)
kernelOpen = np.ones((5,5))
kernelClose = np.ones((5,5))

while True :
    ret,img = cam.read()
    img = cv2.resize(img,(320,244))

    #convert BGR to HSV
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #create a mask 
    mask = cv2.inRange(imgHSV,lower,upper)
    #morphology
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal = maskClose
    conts = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    #ctr = np.array(conts).reshape((-1,1,2)).astype(np.int32)
    #cv2.drawContours(mask, [ctr], 0, (0, 255, 0), -1)
    #cv2.drawContours(img,conts,0,(0,255,0),3)
    
    #for i in range(len([ctr])):
        
    (x,y,w,h) = cv2.boundingRect(conts)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        #cv2.imshow('mask',mask)
        #cv2.imshow('maskopen',maskOpen)
    cv2.imshow('maskClose',maskClose)
    cv2.imshow('img',img)
    

    key = cv2.waitKey(5)
    if key==255: key=-1 
    if key >= 0:
            break
    print('Closing the camera')
 
cam.release()
cv2.destroyAllWindows()
print('bye bye!')
quit()     
