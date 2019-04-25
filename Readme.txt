The single code file "Glaucoma_detection" contains all the code for the term project.

The code works perfectly and a snapshot of the results run on our laptop has been attached.

The link to dataset in given is - http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php

Copy the dataset into the folder of the code.

****The code has many places where the folder of the corresponding files named in the comments has to be properly mentioned****

The code has 5 major functions:
1.To extract images and cdr values from each folder of the image in the Dhristi datset.
2.To segment the cup and disc region from each of the fundus image.
3.Calculate the cdr values from these segmentations.
4.Main function where segmented images are fed to these function and corresponding cdr values calculated and are stored in as csv file.
5.Training of classification model and the output of results.

The a,b,c,d csv files are real cdr values of each expert.
x,y values are cdr values and its label of training dataset
x1,y1 are the calculated cdr values for test image and it's real label.


The rest of the details are explained in the comments of the code itself.

An adaptive threshold based algorithm for optic disc and cup segmentation in fundus images - https://ieeexplore.ieee.org/document/7095384
