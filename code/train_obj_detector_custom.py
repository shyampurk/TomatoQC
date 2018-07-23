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

# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since object are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

print(ip_folder)
ip_folder = os.path.abspath(ip_folder)
training_xml_path = os.path.abspath(os.path.join(ip_folder, "training","training.xml"))
testing_xml_path = os.path.abspath(os.path.join(ip_folder, "testing","testing.xml"))
print(training_xml_path,testing_xml_path)
# This function does the actual training.  It will save the final detector to
# detector.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
if not os.path.exists(os.path.join(ip_folder,"detector.svm")):
    os.chdir(os.path.dirname(training_xml_path))

    dlib.train_simple_object_detector("training.xml", os.path.join(ip_folder,"detector.svm"), options)



# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
print("")  # Print blank line to create gap from previous output
os.chdir(os.path.dirname(training_xml_path))
train_test = dlib.test_simple_object_detector("training.xml", os.path.join(ip_folder,"detector.svm").replace("\\","/"))
print("Training accuracy: {}".format(train_test))
# However, to get an idea if it really worked without overfitting we need to
# run it on images it wasn't trained on.  The next line does this.  Happily, we
# see that the object detector works perfectly on the testing images.
#print("Testing accuracy: {}".format(dlib.test_simple_object_detector(testing_xml_path, os.path.join(current_dir,"detector.svm")))
os.chdir(os.path.dirname(testing_xml_path))
true_test = dlib.test_simple_object_detector("testing.xml", os.path.join(ip_folder,"detector.svm").replace("\\","/"))
print("Testing accuracy: {}".format(true_test))

os.chdir(current_dir)
# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
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
        positive_verify = positive_verify + 1
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            xmin = d.left()
            ymin = d.top()
            xmax = d.right()
            ymax = d.bottom()

            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),5)

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
print("% of Images Detected:",int(positive_verify*100/tot_verify_files))
