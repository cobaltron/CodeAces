import sys
sys.path.append('../src/')
from Drowsiness_Detection import Frame,FaceDetectionManager,EyeDetectionManager,EyeClosureManager,MouthDetectionManager,YawningManager
import unittest
import cv2
import dlib
import numpy as np
class Test_simple(unittest.TestCase):

    l=["face.jpg","eyes_closed.jpg","no_face.jpg"]
    for i in l:
        
        def test_res(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            expected_img_dimension=2
            self.assertEqual(img.ndim,expected_img_dimension)
        
        def test_shape_to_np(self):
            faceimg=[]
            faceimg=np.array(faceimg)
            landmarks=[]
            landmarks=np.array(landmarks)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('../resources/shape_predictor_68_face_landmarks.dat')
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            faceimg = detector(img,0)
            for (i, faceimg) in enumerate(faceimg): # All 68 facial landmarks are being detected
                landmarks= predictor(img,faceimg)
                landmarks = fd.shape_to_np(landmarks)
            if(self.i=="no_face.jpg"):
                pass
            else:
                expected_landmarks_shape=(68,2)
                self.assertEqual(landmarks.shape,expected_landmarks_shape)



        def test_getface(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ret,lands=fd.getFace(img)
            shap=lands.shape
            if(self.i=="no_face.jpg"):
                expected_noface=-1
                self.assertEqual(expected_noface,ret)
            else:
                expected_face=1
                self.assertEqual(expected_face,ret)
                expected_shape=(68,2)
                self.assertEqual(shap,expected_shape)
        
        def test_getEye(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            gray=f.res()
            fd=FaceDetectionManager()
            ed=EyeDetectionManager()
            ret,lands=fd.getFace(gray)
            left,right=ed.getEye(lands)
            if(self.i=="no_face.jpg"):
                pass
            else:
                expected_eye_shape=(6,2)
                self.assertEqual(left.shape,expected_eye_shape)
                self.assertEqual(right.shape,expected_eye_shape)
        
        def test_Drowsiness_Detected_eye(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ret,lands=fd.getFace(img)
            ed=EyeDetectionManager()
            left,right=ed.getEye(lands)
            lshap=left.shape
            rshap=right.shape
            e=EyeClosureManager()
            t=e.Drowsiness_Detected(left,right)
            if(self.i =="eyes_closed.jpg"):
                expected_eye=1
                self.assertEqual(t,expected_eye)
            elif(self.i=="face.jpg"):
                expected_no_eye=0
                self.assertEqual(t,expected_no_eye)
            else:
                pass
                    
        
        
        def test_eye_aspect_ratio(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            gray=f.res()
            fd=FaceDetectionManager()
            ed=EyeDetectionManager()
            ec=EyeClosureManager()
            if(self.i=="no_face.jpg"):
                pass
            else:   
                ret,lands=fd.getFace(gray)
                left,right=ed.getEye(lands)
                lef=ec.eye_aspect_ratio(left)
                rig=ec.eye_aspect_ratio(right)
                t=str(lef)
                s=str(rig)
                self.assertFalse(t.strip().isdigit())
                self.assertFalse(s.strip().isdigit())


    l=["face.jpg","no_face.jpg","yawn.jpg"]
    for i in l:

        def test_Drowsiness_Detected_yawn(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ret,lands=fd.getFace(img)
            md=MouthDetectionManager()
            upper,lower=md.getMouth(lands)
            ym =YawningManager()
            t=ym.Drowsiness_Detected(upper,lower)
            if(self.i =="yawn.jpg"):
                expected_yawn=1
                self.assertEqual(t,expected_yawn)
            elif(self.i=="no_face.jpg"):
                pass
            else:
                expected_no_yawn=0
                self.assertEqual(t,expected_no_yawn)
        def test_getMouth(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ret,lands=fd.getFace(img)
            if(self.i=="no_face.jpg"):
                pass
            else:
                ed=MouthDetectionManager()
                upper,lower=ed.getMouth(lands)
                expected_lip_shape=(3,2)
                self.assertEqual(upper.shape,expected_lip_shape)
                self.assertEqual(lower.shape,expected_lip_shape)

        def test_yawn_detection(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ed=MouthDetectionManager()
            ym=YawningManager()
            ret,lands=fd.getFace(img)
            upper,lower=ed.getMouth(lands)
            r=ym.yawn_detection(upper,lower)
            r=str(r)
            self.assertFalse(r.strip().isdigit())
            
    
if __name__ == '__main__': 
    unittest.main() 
