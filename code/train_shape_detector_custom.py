#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make an object
#   detector for things like object, pedestrians, and any other semi-rigid
#   object.  In particular, we go though the steps to train the kind of sliding
#   window object detector first published by Dalal and Triggs in 2005 in the
#   paper Histograms of Oriented Gradients for Human Detection.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import os
import sys
import glob
import cv2
import dlib
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.path.abspath(os.curdir)
print(current_dir)
# In this example we are going to train a face detector based on the small
# object dataset in the examples/object directory.  This means you need to supply
# the path to this object folder as a command line argument so we will know
# where it is.
if len(sys.argv) < 2:
    print(
        "Give the path to the examples/object directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/object")
    exit()
ip_folder = sys.argv[1]
random_check_folder = sys.argv[2]

options = dlib.shape_predictor_training_options()
# Now make the object responsible for training the model.
# This algorithm has a bunch of parameters you can mess with.  The
# documentation for the shape_predictor_trainer explains all of them.
# You should also read Kazemi's paper which explains all the parameters
# in great detail.  However, here I'm just setting three of them
# differently than their default values.  I'm doing this because we
# have a very small dataset.  In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
options.oversampling_amount = 300
# I'm also reducing the capacity of the model by explicitly increasing
# the regularization (making nu smaller) and by using trees with
# smaller depths.
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

print(ip_folder)
ip_folder = os.path.abspath(ip_folder)
training_xml_path = os.path.abspath(os.path.join(ip_folder, "training","training.xml"))
testing_xml_path = os.path.abspath(os.path.join(ip_folder, "testing","testing.xml"))
print(training_xml_path,testing_xml_path)
# This function does the actual training.  It will save the final detector to
# detector.dat.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
if not os.path.exists(os.path.join(ip_folder,"detector.dat")):
    os.chdir(os.path.dirname(training_xml_path))

    dlib.train_shape_predictor("training.xml", os.path.join(ip_folder,"detector.dat"), options)



# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
print("")  # Print blank line to create gap from previous output
os.chdir(os.path.dirname(training_xml_path))
train_test = dlib.test_shape_predictor("training.xml", os.path.join(ip_folder,"detector.dat").replace("\\","/"))
print("Training accuracy: {}".format(train_test))
# However, to get an idea if it really worked without overfitting we need to
# run it on images it wasn't trained on.  The next line does this.  Happily, we
# see that the object detector works perfectly on the testing images.
#print("Testing accuracy: {}".format(dlib.test_simple_object_detector(testing_xml_path, os.path.join(current_dir,"detector.dat")))
os.chdir(os.path.dirname(testing_xml_path))
true_test = dlib.test_shape_predictor("testing.xml", os.path.join(ip_folder,"detector.dat").replace("\\","/"))
print("Testing accuracy: {}".format(true_test))

os.chdir(current_dir)
# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
predictor = dlib.shape_predictor(os.path.join(ip_folder,"detector.dat").replace("\\","/"))
detector = dlib.simple_object_detector(os.path.join(ip_folder,"detector.svm").replace("\\","/"))

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
#win_det = dlib.image_window()
#win_det.set_image(detector)

# Now let's run the detector over the images in the object folder and display the
# results.
#win = dlib.image_window()
print("Showing detections on the images in the object folder...")
verify_files = glob.glob(os.path.join(random_check_folder, "*.JPG"))
tot_verify_files = len(verify_files)
positive_verify = 0
for f in verify_files:

    print("Processing file: {}".format(f))
    img = io.imread(f)
    orig_img = img.copy()
    dets = detector(img)
    print("Number of object detected: {}".format(len(dets)))
    #win.clear_overlay()
    #win.set_image(img)
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()

    if len(dets)>0:
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            xmin = d.left()
            ymin = d.top()
            xmax = d.right()
            ymax = d.bottom()

            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),5)
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                        shape.part(1)))
            print(shape)
            if len(shape.parts())>0:
                parts = []
                positive_verify = positive_verify + 1
                for part in enumerate(shape.parts()):
                    print(part)
                    parts.append([part[1].x,part[1].y])
                nparts = np.array(parts, np.int32)
                print(nparts)
                cv2.polylines(img, np.int32([nparts]),True,(0,255,0))

    fig = plt.figure(figsize = (15,15))
    ax1 = fig.add_subplot(211)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Original Image')
    ax1.imshow(orig_img)
    #plt.show()

    ax2 = fig.add_subplot(212)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Image with Detections')
    ax2.imshow(img)
    plt.show()
print("% of Image Shapes Detected:",int(positive_verify*100/tot_verify_files))
