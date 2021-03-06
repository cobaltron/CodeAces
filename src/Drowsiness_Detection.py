import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from threading import Thread
import threading
from playsound import playsound
from sys import platform
import os

#Author-Sheersendu Ghosh
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
#Author-Sheersendu Ghosh
class FaceDetectionManager:
    faceimg=[]
    faceimg=np.array(faceimg)
    landmarks=[]
    landmarks=np.array(landmarks)
    gray_face=[]
    gray_face=np.array(gray_face)
    detector = dlib.get_frontal_face_detector()#inbuilt face detector in dlib
    predictor = dlib.shape_predictor('../resources/shape_predictor_68_face_landmarks.dat')#loading the pretrained model for facial landmark detection
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

#Author-Sheersendu Ghosh
class EyeDetectionManager:
    left_eye=[]
    right_eye=[]
    #array of indexes to be extracted from the array of coordinates
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    def getEye(self,landmarks): # landmarks for left and right eye is being returned
        self.left_eye = landmarks[self.LEFT_EYE_POINTS]
        self.right_eye = landmarks[self.RIGHT_EYE_POINTS]
        return self.left_eye,self.right_eye
#Author- Sheersendu Ghosh
class EyeClosureManager:
    EYE_AR_THRESH = 0.25
    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear
    def Drowsiness_Detected(self,left_eye,right_eye):
        try:
            #calculate and store the Eye Aspect Ratios for the left and right eye
            ear_left = self.eye_aspect_ratio(left_eye)
            ear_right = self.eye_aspect_ratio(right_eye)
            #find the average EAR
            ear = (ear_left + ear_right) / 2.0
            if ear < self.EYE_AR_THRESH:
                return 1
            else:
                return 0
        except:
            return -1
#Author-Souvik Dey
class MouthDetectionManager:
    upper_lip=[]
    lower_lip=[]
    #array of indexes to be extracted from the array of coordinates
    LOWER_LIP_POINTS = list(range(65, 68))
    UPPER_LIP_POINTS = list(range(61, 64))
    def getMouth(self,landmarks):#landmarks for upper and lower lip is being returned
        self.upper_lip=landmarks[self.UPPER_LIP_POINTS]
        self.lower_lip=landmarks[self.LOWER_LIP_POINTS]
        return self.upper_lip,self.lower_lip
#Author-Souvik Dey
class YawningManager:
    YAWN_THRESH = 8
    def yawn_detection(self,upper_lip,lower_lip):#Distance between the upper and lower lip is being measured
        top_mean = np.mean(upper_lip, axis=0)
        low_mean = np.mean(lower_lip, axis=0)
        dist = abs(top_mean[1] - low_mean[1])
        #print(dist)
        return dist
    def Drowsiness_Detected(self,upper_lip,lower_lip): #Comparing the calculated distance with threshold and drowsiness is detected

        d=self.yawn_detection(upper_lip,lower_lip)
        if d > self.YAWN_THRESH:
            return 1
        else:
            return 0
#Author-Rajarshi Lahiri
class BuzzerAPI:

    def alarm(self):
        if ps:
            if platform == "win32":
                playsound('../resources/alarm-buzzer.mp3')# Activated when drowsiness is detected
            else:
                os.system('mpg123 "../resources/alarm-buzzer.mp3"')# Activated when drowsiness is detected
    def alert(self):
        if platform=="win32":
            playsound('../resources/alert.mp3')# Activated when no or more than one face is detected
        else:
            os.system('mpg123 "../resources/alert.mp3"')# Activated when no or more than one face is detected
#Author-Rajarshi Lahiri
class MainManager:

    global ps #global flag variable used to keep playing the buzzer
    ps=True
    EYE_AR_CONSEC_FRAMES = 25
    YAWN_CONSEC_FRAMES = 5
    EYE_COUNTER=0
    YAWN_COUNTER=0
    ALARM_ON=False
    def main(self):
        video_capture = cv2.VideoCapture(0) #Video is being captured from webcam
        c=0
        while(True):
            ret, frame = video_capture.read() #Video is being captured from webcam
            if(ret==True):
                f=Frame(frame)
                cv2.imshow("out",f.res())
                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break
                #Objects of classes used for calling methods
                fm=FaceDetectionManager()
                em=EyeDetectionManager()
                ecm=EyeClosureManager()
                mm=MouthDetectionManager()
                ym=YawningManager() 
                ba=BuzzerAPI()

                ret,land=fm.getFace(f.res()) #Read landmark coordinates if present and return value for ret
                leye,reye=em.getEye(land) #Landmark coordibates for left and right eye
                upper,lower=mm.getMouth(land) #landmark coordinates for mouth
                if(ecm.Drowsiness_Detected(leye,reye)==1):
                    self.EYE_COUNTER += 1
                    if self.EYE_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        ps=True
                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            t = Thread(target=ba.alarm)#A deamon thread is created for activating the alarm if drowsiness is detected
                            t.deamon = True
                            t.start()
                elif(ecm.Drowsiness_Detected(leye,reye)==-1):
                    c=c+1
                    if(c%45==0): #2.2.2 :Bug fixed echoing of alert when no face or more than one face is detected
                        t1 = Thread(name='Thread-a',target=ba.alert)#A deamon thread is created for activating the alarm if drowsiness is detected
                        t1.setDaemon(True)
                        t1.start()
                else:
                    self.EYE_COUNTER = 0
                    self.ALARM_ON = False
                    ps=False    

                if(ym.Drowsiness_Detected(upper,lower)==1):
                    self.YAWN_COUNTER += 1
                    if self.YAWN_COUNTER >= self.YAWN_CONSEC_FRAMES:
                        ps=True
                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            t = Thread(target=ba.alarm)#A deamon thread is created for activating the alarm if drowsiness is detected
                            t.deamon = True
                            t.start()
                else:
                    self.YAWN_COUNTER = 0
                    self.ALARM_ON = False
                    ps=False 
            else:
                print("Camera not connected.Check connection.")
                break
        video_capture.release()
        cv2.destroyAllWindows()
ob=MainManager() # main manager object is being created
ob.main() #method for main manager is being called
