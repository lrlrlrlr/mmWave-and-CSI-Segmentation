# mmWave-gesture-signal-segmentation
## Introduction 
This library can help us quickly implement mmWave gesture signal segmentation, and can align with CSI data to segment the gesture from the CSI signal.  
____________________________________________
  Original mmwave:
    <img src="https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/original_mmwave.png" width="300" height="300">  Original WiFi:
  <img src="https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/original_wifi.png" width="300" height="300">
____________________________________________
### The basic steps 
  1. mmWave signal preprocessing: including background removal, OTSU thresholding  
  ![bg removal and OTSU thresholding](https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/out2.png)
  3. Cutting the silent area: Determine the position of the gesture signal through the Edge dectection principle of computer vision. 
  ![edge dectection](https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/edge_dectection.png) 
  5. By calculating the duration of the gesture, excluding extreme values, improve the cutting accuracy  
  ![extreme value removal](https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/filter.png)
  7. Output results  
  ![result](https://github.com/lrlrlrlr/mmWave-and-CSI-Segmentation/blob/main/doc/result.png)




_________________________________________
## Example gesture dataset download
1. to be upload


_________________________________________

## Run the code 
0. Dependencies
    Please install the following libraries:
      - numpy
      - scipy
1. Prepare your dataset
  - the format of mmWave file: txt
  - the format of pushpull file: dat

2. Run main.py  
    `main.py input_directory output_directory`
    
4. example of output


### Versions 
  There are two versions, one allow us to segment the gesture data mannually. The other one can do the segmentation automatically.

## Performance report  
The performance metrics are calculate based on the manual segmentated data (ground truth).


1. Pixel Accuracy (PA)
2. Mean Pixel Accuracy (MPA)
3. Intersection over Union (IU / IoU)
      For the two regions R and R’, the overlap is calculated as follows:
      The covering index is calculated as:
____________________________________________
### Reference 
 [A Review on Deep Learning Techniques Applied to Semantic Segmentation](https://arxiv.org/abs/1704.06857)
___________________________________________
