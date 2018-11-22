import sys
import dlib
from skimage import io
import numpy as np
from scipy.spatial import distance
import cv2

# Developers : S.Sai Hemanth, M. Vihari, V. Vitru Varenya
# email : saihemanth.s@outlook.com

status = "Not sleeping"
predictor_Path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_Path)
Capture = cv2.VideoCapture(0)

def compute_Ear(vec):
    a = distance.euclidean(vec[1], vec[5])
    b = distance.euclidean(vec[2], vec[4])
    c = distance.euclidean(vec[0], vec[3])
    ear = (a + b)/(2 * c)
    return ear

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def Main():
    while(True):
        if(not Capture.isOpened()):
            print("Unable to connect to camera")
            sys.exit()
        ret, img = Capture.read()
        img = cv2.resize(img,(800,600))
        #cv2.imshow('test',img)
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        vec = np.empty([68,2],dtype =int)
        dets = detector(img,1)
        for (k, det) in enumerate(dets):
            shape = predictor(img,det)
            face = predictor(img,det)
            face = shape_to_np(shape)
            (f,g,w,h) = rect_to_bb(det)
            for b in range(68):
                vec[b][0] = shape.part(b).x
                vec[b][1] = shape.part(b).y
                right_ear = compute_Ear(vec[42:48])
                left_ear = compute_Ear(vec[36:42])
                
                if(right_ear+left_ear)/2 <0.2:
                    status = "sleeping"
                    cv2.rectangle(img, (f, g), (f + w, g + h), (0, 0, 255), 2)
                    cv2.putText(img, "Sleeping", (f - 10, g - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    statue = "Not sleeping"
                    cv2.rectangle(img, (f, g), (f + w, g + h), (0, 255, 0), 2)
            for (f,g) in face:
                cv2.circle(img,(f,g),1,(0,0,255),-1)
        cv2.imshow("Output",img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
##                win.add_overlay(shape)
##        win.add_overlay(dets)
##        win.set_title(status)
Main()
