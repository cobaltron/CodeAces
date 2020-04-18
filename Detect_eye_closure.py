import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from threading import Thread
import threading
from playsound import playsound

class Frame:
    img=[]
    img=np.array(img)
    def __init__(self,frame):
        self.img=frame

    def res(self):
        #scaling down the frame to 50% to speed up processing
        scale_percent = 50
        #calculating the new height and width
        width = int(self.img.shape[1] * scale_percent / 100) 
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(self.img, dim, interpolation = cv2.INTER_AREA)
        #flip the image
        image=cv2.flip(resized,1)
        #converting to grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return gray

class FaceDetectionManager:
    faceimg=[]
    faceimg=np.array(faceimg)
    landmarks=[]
    landmarks=np.array(landmarks)
    gray_face=[]
    gray_face=np.array(gray_face)
    detector = dlib.get_frontal_face_detector()#inbuilt face detector in dlib
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#loading the pretrained model for facial landmark detection
    def shape_to_np(self,shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        #converting the predicted coordinated to a numpy array of shape(68,2)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    def getFace(self,img):
        self.faceimg = self.detector(img, 0)#face detection
        #dummy array filled with zeroes
        dummy=np.zeros((68,2))
        ret=1
        #when no face is detected
        if(len(self.faceimg)==0):
            ret=-1
            return ret,dummy
        #when more than one face is detected
        elif(len(self.faceimg)>1):
            ret=-1
            return ret,dummy
        else:
            for (i, self.faceimg) in enumerate(self.faceimg): # All 68 facial landmarks are being detected
                self.landmarks= self.predictor(img, self.faceimg)
                self.landmarks = self.shape_to_np(self.landmarks)
            return ret,self.landmarks
