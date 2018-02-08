# This program reads the camera image
# converts camera image to an eyepair masked image
# loads the X numpy array saved eyeContactClassifier program, 
# transforms the eyepair masked image to a n_component PCA vector
# Loads the classifier created by the eyeContactClassifier program
# predicts the label of each eyepair image

import math
import time

import cv2
import dlib
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera_num = 1
# import glob
# import os
n_components = 300
n_components_lda = 5

# path='/Users/ag6031/Dropbox/IOTAP/EyeContactPrj/data/eyepair/eye_contact/'

predictor_path =  "shape_predictor_68_face_landmarks.dat"
# faces_folder_path = "faces"

face_num = 0
scale=.4
n = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img2 = cv2.imread('mask.png') #logo
rows,cols,channels = img2.shape
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
clf = joblib.load('eyeContactcls.pkl')
pca = joblib.load('pca.pkl')
lda_30 = joblib.load('lda_30.pkl')
lda_15 = joblib.load('lda_15.pkl')
lda0 = joblib.load('lda0.pkl')
lda15 = joblib.load('lda15.pkl')
lda30 = joblib.load('lda30.pkl')

flag = True
jump = False
def lda(X):
    x_15 = lda_15.transform(X)
    x_30 = lda_30.transform(X)
    x0 = lda0.transform(X)
    x15 = lda15.transform(X)
    x30 = lda30.transform(X)
    list = [[x_30[0][0],x_15[0][0],x0[0][0],x15[0][0],x30[0][0]]]
    return np.array(list)

def FaceDetection(imag):
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    return faces

def findEye(leftX,leftY,rightX,rightY,image):
        # calculate slope of the eye line
        slope=math.degrees(np.arctan(float(leftY-rightY)/float(leftX-rightX)))

        # calculate center of the image
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
         
        # rotate the image by slope degrees around the center
        M = cv2.getRotationMatrix2D(center, slope, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        # points
        points = np.array([[rightX,  rightY],
                           [leftX, leftY]])
        # add ones
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])        
        # transform points
        new_points = M.dot(points_ones.T).T 
        mar = int((new_points[0,0]-new_points[1,0])/2)
        cropped = rotated[int(new_points[0,1])-mar:int(new_points[0,1])+mar, int(new_points[1,0]):int(new_points[0,0])]
        (wi,hei) = cropped.shape[:2]
        if (hei <> 0) & (wi <> 0) :
            resized_image = cv2.resize(cropped, (48, 36)) 
        else: 
            resized_image = np.zeros((36,48,3), np.uint8)
            jump = True
        return resized_image

# Start capturing images from camera
# cv2.namedWindow("capture",1)
capture = cv2.VideoCapture(camera_num)
ret, img = capture.read()
while (capture.isOpened ()):
    start = time.time()
    ret, img = capture.read()
    if ret:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
#         imgraw = img   
        faces1 = FaceDetection(img) 
        k = len(faces1)
        for (x, y, w, h) in faces1:  
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  
            shape = predictor(img, dlib_rect)
            leftEye = findEye(shape.part(36).x,shape.part(36).y,shape.part(39).x,shape.part(39).y, img)
            rightEye = findEye(shape.part(42).x,shape.part(42).y,shape.part(45).x,shape.part(45).y, img)
            eyes = np.concatenate((leftEye,rightEye),axis = 1)
            if not jump:
                jump = False
                # apply mask
                roi = eyes[0:rows, 0:cols ]
                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)
                eyes[0:rows, 0:cols ] = dst
                eyes=cv2.cvtColor(eyes,cv2.COLOR_BGR2GRAY)
                # histogram
                eyes=cv2.equalizeHist(eyes)
                featurevector=np.array(eyes,dtype="float64")
#                 featurevector_scaled = preprocessing.scale(featurevector)
                featurevector = featurevector.flatten()
                featurevector=featurevector.reshape(1,-1)
                featurevector= pca.transform(featurevector)
#                 print featurevector

                featurevector= lda(featurevector)
#                 print featurevector

                eyeContact = clf.predict(featurevector)
        #         print eyeContact[0]
                if eyeContact[0] == 1:
#                 if featurevector[0] > 4:
                    print ("Person %d looked at the camera!" %(k))
        #             os.system('say "hi"')
        
        #         cv2.imwrite(path+'sample_neg/'+ str(n) + '.jpg', eyes)
        #         n=n+1
                
                cv2.imshow("Pre_processed: "+str(k),eyes)
                print time.time()-start
#                 cv2.imshow("Normalized: "+str(k+1),featurevector_scaled)
#                 cv2.imshow("Raw_image: "+str(k+1),imgraw)

#                 print time.time()-start
#             cv2.imshow("Eyes_Person: "+str(k+1),featurevector_1)

    ch = cv2.waitKey(1)
    if ch == 32:
        break
capture.release()
cv2.destroyAllWindows()

