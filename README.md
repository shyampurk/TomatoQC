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


  





