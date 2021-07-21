# mmWave-gesture-signal-segmentation
## Introduction 
This library can help us quickly implement mmWave gesture signal segmentation, and can align with CSI data to segment the gesture from the CSI signal.  

The basic steps are  
  1. mmWave signal preprocessing: including background removal, OTSU thresholding  
  2. Cutting the silent area: Determine the position of the gesture signal through the Edge dectection principle of computer vision.  
  3. By calculating the duration of the gesture, excluding extreme values, improve the cutting accuracy  
  4. Output results  

Versions:
  There are two versions, one allow us to segment the gesture data mannually. The other one can do the segmentation automatically.

## Run the code 
0. Dependencies
    Please install the following libraries:
      - numpy
      - scipy


2. Run main.py  
    `main.py input_directory output_directory`
4. example of output

6. ..


## Performance report  
The performance metrics are calculate based on the manual segmentated data (ground truth).


1. PA
2. MPA
3. IU / IoU（Intersection over Union）
      For the two regions R and R’, the overlap is calculated as follows:
      The covering index is calculated as:


 
