# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --folder /path/to/image/folder

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--folder", required=True, nargs='+',
    help="full path to folder containing images")
args = vars(ap.parse_args())

extensions = ('.png','.jpg','.jpeg')

try:
    images = [file for file in os.listdir(args['folder']) if file.lower().endswith(extensions)]
except:
    sys.exit("Invalid folder name")

if len(images) == 0:
    sys.exit("No images found in folder")

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
for image in images:
    print("processing:", image)
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image)
    image = imutils.resize(image, width=800)
    # pad the border, in case the face is right on the edge of the image
    image = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)
            
	import uuid
	f = str(uuid.uuid4())
	cv2.imwrite("aligned/" + f + ".png", faceAligned)

