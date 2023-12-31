# Improving Computer Tomography COVID-19 Lesion Segmentation Using Reproducible Pathways and Data Augmentation Techniques on Multi-demographic Cohorts

# Abstract:
The COVID-19 pandemic has put immense pressure on healthcare systems around the world. To release some of this pressure, scientists attempted to include machine learning techniques in the infection testing process. This was done with the help of medical imaging and was intended as a solution to the lack of fast, reliable testing capabilities. However, reproducing and comparing different possible solutions is difficult since there is no baseline open-source model or data to compare the results with. For this reason, the current thesis aims to evaluate data augmentation techniques to improve automatic COVID-19 lesion segmentation by using reproducible techniques. These techniques come in the form of public data and open-source frameworks for preprocessing images and for creating the models. One such open-source framework is MONAI, a ”freely available, community-supported, PyTorch-based framework for deep learning in healthcare imaging” [1]. Two deep learning models, U-Net and Dynamic U-Net (nnU-Net), implemented using MONAI, were employed to segment COVID-19 lesions. These models were trained and tested on 329 computed tomography scans which were publicly available. In order to improve the performance of the two models, data augmentation techniques in the form of intensity projections, noise filters and sliding windows were used. Overall, regardless of which model was tested, using sliding windows and not adding any noise lead to the most optimal results (AUC = 0.81 or AUC = 0.77, depending on which testing dataset was used). Intensity projection did not seem to improve the results in any situation. The goal of this thesis was to use open-source data and frameworks to improve lesion segmentation models. However, this approach presents certain challenges in the form of limited, low-quality data. Therefore, the aim for future research would be to increase the amount of high-quality data with the help of professional radiologists in order to avoid some of the limitations present in this thesis.


### **Keywords**: COVID-19, lesion segmentation, U-Net, Dynamic U-Net

# Models
## U-Net
![alt text](https://github.com/palfibogdan/3D-covid-segmentation/blob/3bca19177bc3ca140ae1b20e15e6f211dccc9745/Visualization/UNET.png)

## nnU-Net
![alt text](https://github.com/palfibogdan/3D-covid-segmentation/blob/3bca19177bc3ca140ae1b20e15e6f211dccc9745/Visualization/nnUnet.png)

## Visualization
### Original CT scan

![alt text](https://github.com/palfibogdan/3D-covid-segmentation/blob/617317b97fe5073c31707208660e0112a8d2db52/Visualization/notpreprocessed%20(1).jpg)

### Processed CT scan

![alt text](https://github.com/palfibogdan/3D-covid-segmentation/blob/617317b97fe5073c31707208660e0112a8d2db52/Visualization/original_crop.png)

### Prediction
![alt text](https://github.com/palfibogdan/3D-covid-segmentation/blob/617317b97fe5073c31707208660e0112a8d2db52/Visualization/prediction1.png)

