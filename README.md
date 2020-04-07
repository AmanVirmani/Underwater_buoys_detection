# Underwater_buoys_detection
This project does detection and segmentation of underwater buoys based on color information using Gaussian Mixture 
Models(GMM) and Expectation-Maximization(EM) algorithm.

A buoy is a distinctly shaped, colored floating device which has many practical purposes.It can be anchored to the 
bottom,for designating moorings, navigable channels, or obstructions in a water body.

Buoys can also be useful for underwater markings for navigation. For this project we have given an under water video
sequence which shows three different color buoys such as orange, yellow and green.

## Build Instructions

Run the following bash command to detect a single color 
```bash
python3 GMM_EM.py -k <number of clusters> -train <path to train images> -test <path of the video to test>
```

To do detection of all 3 colored buoys, enter the following command.
```bash
python3 main.py 
```

## Output

A video file "output.avi" is generated as output that is the original video with detections as circular 
roi's around the detected buoys.

## Dependencies

The following dependencies must be installed.

1. python3.5 or above 
2. numpy 
3. opencv 3.4 or above
4. argparse
5. scikit-learn
6. scipy 

Enter the given commands in bash terminal to install the dependencies.
```bash
sudo apt-get install python3
pip3 install numpy opencv-python argparse scikit-learn scipy
```