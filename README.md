# Real-world Anomaly Detection and classification in Surveillance Videos

This Repo contains implementation for real world anomaly detection.

## Abstract 

This project aims to detect anomalies and classify those anomalies.
Used pre trained C3D network. C3D used for feature extraction.
Generated features from C3D are fed to Anomaly detection NN(ADNN) and trained using multiple instance learning to classify anomolus video segments.

After training Anomaly detection NN (ADNN) I marked which frame contains anomaly using same trained ADNN.
So These marked frames further fed to simple Classification neural network to classify anomaly type(Fighting/Accident)

### Used Facebooks C3D to ectract features 
Feature extractor directory contains all the details about extraction process.
https://github.com/facebook/C3D

### PREPROCESSING:
Resize each video frame to 240*320 pixels and fix
frame rate at 30fps.

### FEATURE EXTRACTION:
C3D features for every 16-frame video clip
followed by l2 normalization. To obtain features
for a video segment, we take the average of all
16-frame clip features within that segment.

### Used Tools
* Tensorflow
* Python
* Caffe(To extract features)
* Colab

### Dataset
https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0

### Used Papers
* http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf



