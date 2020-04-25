from Detect_eye_closure import Frame,FaceDetectionManager,EyeDetectionManager,EyeClosureManager
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
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#loading the pretrained model for facial landmark detection
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
                expected_landmarks_shape=(68,2)#expected value
                self.assertEqual(landmarks.shape,expected_landmarks_shape)



        def test_getface(self):
            im=cv2.imread(self.i)
            f=Frame(im)
            img=f.res()
            fd=FaceDetectionManager()
            ret,lands=fd.getFace(img)
            shap=lands.shape
            if(self.i=="no_face.jpg"):
                expected_noface=-1 #expected value
                self.assertEqual(expected_noface,ret)
            else:
                expected_face=1#expected value
                self.assertEqual(expected_face,ret)
                expected_shape=(68,2)#expected value
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
                expected_eye_shape=(6,2)#expected value
                self.assertEqual(left.shape,expected_eye_shape)
                self.assertEqual(right.shape,expected_eye_shape)
        
        def test_Drowsiness_Detected(self):
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
                expected_yawn=1#expected value
                self.assertEqual(t,expected_yawn)
            elif(self.i=="no_face.jpg"):
                pass
            else:
                expected_no_yawn=0#expected value
                self.assertEqual(t,expected_no_yawn)    
        
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
            
    
if __name__ == '__main__': 
    unittest.main() 
