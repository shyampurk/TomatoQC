# TomatoQC
A concept application for visual inspection of vegetables(tomatoes)  using Machine Learning via the DLib machine learning toolkit.

This code repository is part of this blog post that examines the possibility of using the Dlib's object detection and recognition
capabalities to scan tomatoes for performing automated visual inspection of their quality.

## Prerequisites

This code is tested on Ubntu 16.4 system with Python 3. Following are the list of external python package dependencies required for running the demo code.

1. Anaconda3
2. Dlib
3. OpenCV (Python bindings)

Prior knowledge of working with Dlib and it's Imglib tool will be helpful. 
For more details on building and using ImgLab, refer to this [Github link](https://github.com/davisking/dlib/tree/master/tools/imglab)

## Sample Images

For testing this demo, we have used a set of sample tomato images 

1. [Self Images](sample/selfImages/test_feature_Defect1_Rej/) - This directory contains the previousy available training images of a tomatoes, having a specific type of defect. 

2. [Client Images](sample/clientImages/Defect1_Reject/) - This directory contains the actual images which are tested for the specific type of defects.

For simplifying the demo, we have used multiple images of a fe tomatoes from different angles. 


## Building the sample dataset of images

Imglab is the tool for annotating and labeling the images and storing their attributes in an XML file. before performing the below steps, make sure that you compile and bulild the Imglab tool.

### Step 1 - Partition the  self images into training and testing

You can see the two directories inside the [Self Images](sample/selfImages/test_feature_Defect1_Rej/) directory, namely training and testing. We have already split the images randomly but you can define your ouw split as well.

### Step 2 - Create the XML database for training 

Clone the repo and run the imglab tool for training images (Make sure imglab is in your UNIX PATH)

    imglab -c training.xml sample/selfImages/test_feature_Defect1_Rej/training

Move the generated training.xml to the sample/selfImages/test_feature_Defect1_Rej/training directory.

### Step 3 - Load the XML database for annonating each image

Run the imglab tool again from within the training directory

    imglab --parts "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14" training.xml

### Step 4 - Label the images

The previous imglab command will open up a UI and the we can mark 14 annotations on each tomato image to highlight the surface defects. You can choose to have less number of annotations as well.

When the image is loaded, you can press shift and right key to make a blue coloured rectangle box. Inside the box you can lable the parts which have defects. To mark the partitions, right click on mouse and select Add 00.

<image>

In the similar way mark other 13 points.

<image>

Do it for rest of the images and finally go to file->save. The xml file will get updated.

### Step 5 - Repeat teh above steps for the testing directory under [Self Images](sample/selfImages/test_feature_Defect1_Rej/) 

Make sure that you name the xml file as testing.xml. 



## Run the demo

There are two stages of running this program.

1. Objection detection - To test whether the tomato can be detected within an image. 

2. Shape Detection - To test whether the the shape of surface defect on the tomate image can be detected.


  
### Object Detector

The [train_obj_detector_custom.py](code/train_obj_detector_custom.py) is the python script modified from the original Dlib object detection sample code. It takes two parameters/arguments as input

1) the path where its own data is stored. (path to own data)

2) the path where the client’s data is stored.  (path to client data)

The command to execute this script is

        python train_obj_detector_custom.py <path to own data> <path to client data>

The training.xml file was built through imglab is fed to detector.svm (which is created during the execution of the program). The detector file will be tested against its own data to train itself and check for the accuracy. If it works well, then train with more images to improve its efficiency for the final output.

Then finally the client’s data is tested through detector.svm. The objects are detected when the images are converted to grey scale (through imread function) and the boundary is formed through pixel’s left, right, top, bottom positions and enclosed within a rectangle shape. The object detection is done through dlib libraries and the labelled images in xml file dataset. SVM file generated will further be used for shape detection in the next stage.

This script will call the following Dlib APIs

train_simple_object_detector() – It trains simple object detector based on labelled image image in XML file dataset training.xml. It returns trained object detector in svm file(detector.svm).

test_simple_object_detector( ) – This function runs detector.svm file against 2 datasets – training.xml (known dataset) and testing.xml (unknown dataset). It returns the precision, average precision and accuracy of the detector.

simple_object_detector( ) - This function represents sliding window histogram-of-oriented-gradients based object detector. It is the final detector (detector.svm) which is used against real world images to detect the object bounds, in this case the tomatoes.

<console out>
    
<UI out>
   

### Shape Detector

[train_shape_detector_custom.py](code/train_shape_detector_custom.py) is the python script modified from the original Dlib object detection sample code. It takes the same two parameters/arguments as inputs.

1) the path where its own data is stored. (path to own data)

2) the path where the client’s data is stored.  (path to client data)

The command to execute this script is

        python train_shape_detector_custom.py <path to own data> <path to client data>

The concept behind detecting shape is HOG filter (Histogram of oriented gradients). It counts the occurrence of gradient orientation in localised portion of images. Images are further divided into small connected regions called cells and for pixels within each cell, histogram of gradient is compiled. Classifier detects the objects through sliding window and if there is any large probability observed in sliding window, it will record the bounding box of window. These will be highlighted with green colour polygon shape.

This script will call the following Dlib APIs

train_shape_predictor() - Uses Dlib’s shape_predictor_trainer to train a shape predictor based on the labelled images in the XML file training.xml and the provided options. This function assumes the file training.xml is in the XML format produced by imglab tool. The trained shape predictor is serialized to the file detector.dat.

test_shape_predictor() - This function tests the predictor(detector.dat) against the dataset and returns the mean average error of the detector. This error denotes the average distance between object output given by detector.dat and where it should be according to truth data.

shape predictor() - It is a tool that takes in detector.dat file and outputs a set of point locations that defines the defect of the object. This function is expected to show the defected area of tomatoes.


<console out>
    
<UI out>
    
    
