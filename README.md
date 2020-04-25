# msds19043_COVID19_DLSpring2020
This repository contains code and results for COVID-19 classification assignment by Deep Learning Spring 2020 course offered at Information Technology University, Lahore, Pakistan. This assignment is only for learning purposes and is not intended to be used for clinical purposes.


# Contents

[INTRODUCTION: 2](#_Toc38741651)

[EXPERIMENTAL SETUP: 2](#_Toc38741652)

[TASK 1: 2](#_Toc38741653)

[I. VGG16 FCLayer530, learning rate 0.001 2](#_Toc38741654)

[II. VGG16 FCLayer530, learning rate 0.00001 4](#_Toc38741655)

[III. VGG16 FCLayer 2000, learning rate 0.001 6](#_Toc38741656)

[IV. VGG16 FCLayer: 2000 , learning rate 0.00001 8](#_Toc38741657)

[V. RESNET FCLayer530, learning rate 0.001 10](#_Toc38741658)

[VI. RESNET FCLayer530, learning rate 0.00001 12](#_Toc38741659)

[VII. RESNET FCLayer 2000, learning rate 0.00001 14](#_Toc38741660)

[TASK 2 16](#_Toc38741661)

[I. VGG Net, learning rate 0.001 16](#_Toc38741662)

[II. VGG Net 16, convolutional layers FREEZED, learning rate 0.001 18](#_Toc38741663)

[III. VGG Net 16, convolutional layers freezed [0,2,5,10,24,28], learning rate 0.001 20](#_Toc38741664)

[IV. Res Net 18, No Freezing learning rate 0.001 22](#_Toc38741665)

[V. Res Net 18, FC layers freeze, learning rate 0.001 24](#_Toc38741666)

[VI. Res Net 18,convolution 1 and 2 layers freeze, learning rate 0.001 26](#_Toc38741667)

[COMPARISON AND ANALYSIS 28](#_Toc38741668)

#


Dataset:

[https://drive.google.com/drive/u/1/folders/1-FzZhQO9oHIT9SNOWYoKsuz7fe447vtR](https://drive.google.com/drive/u/1/folders/1-FzZhQO9oHIT9SNOWYoKsuz7fe447vtR)

GitHub link:

[https://github.com/Basir-mahmood/msds19043\_COVID19\_DLSpring2020](https://github.com/Basir-mahmood/msds19043_COVID19_DLSpring2020)

# INTRODUCTION:

The assignment includes the classification of X-Ray Images and the classes are for the COVID-19 infected or normal. VGG16 and RESNET-18 are used for the detection and classification of the images. There are several analyses which are given in the following. The most significant problem faced was the time constraint, time to complete and submit within deadline is too near, whereas, each epoch takes around 6 minutes approximately. So, in the whole a lot of time is required for analysis.

# EXPERIMENTAL SETUP:

## TASK 1:

1.
### VGG16 FCLayer530, learning rate 0.001

In the first task, the vgg16 network&#39;s FC layer was changed, where the first layer&#39;s neuron was changed to 530, which is equal to = (roll number\*10 )+100, and there is a second layer with two neurons i.e., number of classes . The accuracy over the epochs can be seen in the following image.

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_2a94e903d6f44981.gif) ![](RackMultipart20200425-4-4lmrq_html_4ffc30398ba9616b.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_65a0413f49bcf5b9.png) ![](RackMultipart20200425-4-4lmrq_html_8da7e6b2e2d96a5e.png)

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 534 | 81 |
| --- | --- | --- |
| **Actual Normal** | 25 | 860 |

#### F1 Score is : 0.941

#### Accuracy : 92.93

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_50a82f949e1f8bff.png) ![](RackMultipart20200425-4-4lmrq_html_1b1df3086281c78d.png)

1.
### VGG16 FCLayer530, learning rate 0.00001

In this part, learning rate is changed to 0.00001, and the other parameters are same.

![](RackMultipart20200425-4-4lmrq_html_cdd28d00f44b03cd.gif) ![](RackMultipart20200425-4-4lmrq_html_8966fb64f4eeaa9a.gif)

Following are the Accuracy, F1 Measure and Confusion Matrices.

![](RackMultipart20200425-4-4lmrq_html_2cbb111495e266c7.png) ![](RackMultipart20200425-4-4lmrq_html_9acf2ee1116e7a33.png)

![](RackMultipart20200425-4-4lmrq_html_f6ae539d675e1321.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 544 | 71 |
| --- | --- | --- |
| **Actual Normal** | 25 | 860 |

#### F1 Score is : 0.947

#### Accuracy : 93.60

![](RackMultipart20200425-4-4lmrq_html_ed7e37af5f916f92.png) ![](RackMultipart20200425-4-4lmrq_html_f6fb275861cf995a.png)

Above given are the 2 images from each result class predictions

1.
### VGG16 FCLayer 2000, learning rate 0.001

Following are the loss and accuracy curves obtained over epochs.

![](RackMultipart20200425-4-4lmrq_html_38c9652f23908068.gif) ![](RackMultipart20200425-4-4lmrq_html_65cae98788b15fa9.gif)

F1- measure, Accuracy and Confusion Matrices are as follows,

![](RackMultipart20200425-4-4lmrq_html_934e43cfbf674475.png) ![](RackMultipart20200425-4-4lmrq_html_b767d276eb3bb7ad.png)

![](RackMultipart20200425-4-4lmrq_html_6a4cd7b9fbded418.png)

![](RackMultipart20200425-4-4lmrq_html_d6320248c95fcd8.png) ![](RackMultipart20200425-4-4lmrq_html_28014379ac78b831.png)

Above are the 2 case examples for each resultant of class.

#####

1.
### VGG16 FCLayer: 2000 , learning rate 0.00001

FC Layers neurons are changed from 530 to 2000. Following results are obtained.

![](RackMultipart20200425-4-4lmrq_html_1f6ac7aaa8b0c577.gif) ![](RackMultipart20200425-4-4lmrq_html_133bf42bd06bbb9e.gif)

Accuracy, F1- Measure and Confusion Matrices are following

![](RackMultipart20200425-4-4lmrq_html_b7e9e1fa4831218f.png) ![](RackMultipart20200425-4-4lmrq_html_ac596327fde0873e.png)

![](RackMultipart20200425-4-4lmrq_html_1c723827721dd8c8.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 517 | 98 |
| --- | --- | --- |
| **Actual Normal** | 37 | 848 |

#### F1 Score is : 0.926

#### Accuracy : 91.0

![](RackMultipart20200425-4-4lmrq_html_13bbda83ab12a61.png) ![](RackMultipart20200425-4-4lmrq_html_34541cf57f9ac196.png)

Above given are the 2 pictures for each case.

1.
### RESNET FCLayer530, learning rate 0.001

In this task, the ResNet network&#39;s FC layer was changed, where the first layer&#39;s neuron was changed to 530, which is equal to = (roll number\*10 )+100, and there is a second layer with two neurons i.e., number of classes . The accuracy over the epochs can be seen in the following image.

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_85e59b5fe36f388f.gif) ![](RackMultipart20200425-4-4lmrq_html_817c4a365abaf083.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_f6665684b1a4efca.png) ![](RackMultipart20200425-4-4lmrq_html_d71e265e3707193f.png)

![](RackMultipart20200425-4-4lmrq_html_4c9d19042023c3ed.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 562 | 53 |
| --- | --- | --- |
| **Actual Normal** | 171 | 714 |

#### F1 Score is : 0.864

#### Accuracy : 85.06

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_9cc78fded826678d.png) ![](RackMultipart20200425-4-4lmrq_html_5ea55fe28ae16ad1.png)

1.
### RESNET FCLayer530, learning rate 0.00001

In this task, the resNet network&#39;s FC layer was changed, where the first layer&#39;s neuron was changed to 530, which is equal to = (roll number\*10 )+100, and there is a second layer with two neurons i.e., number of classes . The accuracy over the epochs can be seen in the following image.

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_f065da6cc82449ab.gif) ![](RackMultipart20200425-4-4lmrq_html_a32044e8b2610354.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_8174f8bc2d457b0c.png) ![](RackMultipart20200425-4-4lmrq_html_1682afc515d3c75.png)

![](RackMultipart20200425-4-4lmrq_html_79718f35158a203f.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 539 | 76 |
| --- | --- | --- |
| **Actual Normal** | 121 | 764 |

#### F1 Score is : 0.885

#### Accuracy : 86.86

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_557761798f0c086e.png) ![](RackMultipart20200425-4-4lmrq_html_1c77ce7e13672888.png)

1.
### RESNET FCLayer 2000, learning rate 0.00001

In this task, the resNet network&#39;s FC layer was changed, where the first layer&#39;s neuron was changed to 200, and there is a second layer with two neurons i.e., number of classes . The accuracy over the epochs can be seen in the following image.

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_2043c3ecedced27a.gif) ![](RackMultipart20200425-4-4lmrq_html_72273778607ad19f.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_39bdc1b0a1c5140f.png) ![](RackMultipart20200425-4-4lmrq_html_6001e8074ab47811.png)

![](RackMultipart20200425-4-4lmrq_html_c9c071bc8072e03b.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 483 | 132 |
| --- | --- | --- |
| **Actual Normal** | 89 | 796 |

#### F1 Score is : 0.878

#### Accuracy : 85.26

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_fa2a2fb03c42ede8.png) ![](RackMultipart20200425-4-4lmrq_html_318bb1f7b74eca0d.png)

## TASK 2

1.
### VGG Net, learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_24e5aefb40cd58fd.gif) ![](RackMultipart20200425-4-4lmrq_html_eeb7178e31104721.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_78ca81e87a1067ba.png) ![](RackMultipart20200425-4-4lmrq_html_f1c0972a880e4e2c.png)

![](RackMultipart20200425-4-4lmrq_html_96c2d7f41fc23efe.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 584 | 31 |
| --- | --- | --- |
| **Actual Normal** | 29 | 856 |

#### F1 Score is : 0.966

#### Accuracy : 96.0

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_b5aaa67ce583cc58.png) ![](RackMultipart20200425-4-4lmrq_html_810e23a37d701f77.png)

1.
### VGG Net 16, convolutional layers FREEZED, learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_1ec0045effdd6b27.gif) ![](RackMultipart20200425-4-4lmrq_html_1eeb83fcf45f7821.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_6f8a2e63cde9b504.gif) ![](RackMultipart20200425-4-4lmrq_html_924285ce5f4dbc2.gif)

![](RackMultipart20200425-4-4lmrq_html_cf2a8515547e3984.gif)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 573 | 42 |
| --- | --- | --- |
| **Actual Normal** | 9 | 876 |

#### F1 Score is : 0.971

#### Accuracy : 96.6

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_9fe98a29c0887923.png) ![](RackMultipart20200425-4-4lmrq_html_829fc92e473e4dcd.png)

1.
### VGG Net 16, convolutional layers freezed [0,2,5,10,24,28], learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_7a0084040dc9cd36.gif) ![](RackMultipart20200425-4-4lmrq_html_608e13c9b0027b00.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_b81061919469f8c2.png) ![](RackMultipart20200425-4-4lmrq_html_14801aa54b9255d4.png)

![](RackMultipart20200425-4-4lmrq_html_890cc3878a270426.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 589 | 26 |
| --- | --- | --- |
| **Actual Normal** | 17 | 868 |

#### F1 Score is : 0.975

#### Accuracy : 97.13

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_50a7115865ebbf4f.png) ![](RackMultipart20200425-4-4lmrq_html_881abb595c3ed4c7.png)

1.
### Res Net 18, No Freezing learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_f0cff303f94efacc.gif) ![](RackMultipart20200425-4-4lmrq_html_cb5a0c5fc47f57c6.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_dad3e5ab946523eb.png) ![](RackMultipart20200425-4-4lmrq_html_1abb8599d5404fc4.png)

![](RackMultipart20200425-4-4lmrq_html_cffd50bf239f1a3f.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 595 | 20 |
| --- | --- | --- |
| **Actual Normal** | 30 | 855 |

#### F1 Score is : 0.971

#### Accuracy : 96.66

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_2ceb79c363b47735.png) ![](RackMultipart20200425-4-4lmrq_html_a871b782449f5ffd.png)

1.
### Res Net 18, FC layers freeze, learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_c722b61260dfa279.gif) ![](RackMultipart20200425-4-4lmrq_html_8cde56d836e0f5a5.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_46e401170e8f6f7f.png) ![](RackMultipart20200425-4-4lmrq_html_370123cdaacbea21.png)

![](RackMultipart20200425-4-4lmrq_html_edee871b0640905.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 581 | 34 |
| --- | --- | --- |
| **Actual Normal** | 14 | 871 |

#### F1 Score is : 0.926

#### Accuracy : 91.0

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_1056aa27e6d894dc.png) ![](RackMultipart20200425-4-4lmrq_html_ca6c622b7ebfda1c.png)

1.
### Res Net 18,convolution 1 and 2 layers freeze, learning rate 0.001

In this task, the VGG network&#39;s is trained on the data,

Following graphs, describe the losses and training curves over epochs.

![](RackMultipart20200425-4-4lmrq_html_4bc8f711170c20b6.gif) ![](RackMultipart20200425-4-4lmrq_html_e24115c82f5b9c2c.gif)

Accuracy, F1 measure and confusion matrix is as follows,

![](RackMultipart20200425-4-4lmrq_html_ae7e19038deafca.png) ![](RackMultipart20200425-4-4lmrq_html_4244dad63a5418c.png)

![](RackMultipart20200425-4-4lmrq_html_a1ec7922be00ea3f.png)

### Testing

#### Confusion Matrix

|
 | **Predicted Infected** | **Predicted Normal** |
| --- | --- | --- |
| **Actual Infected** | 582 | 33 |
| --- | --- | --- |
| **Actual Normal** | 38 | 847 |

#### F1 Score is : 0.959

#### Accuracy : 95.26

Following are the two sample images from each class results.

![](RackMultipart20200425-4-4lmrq_html_2e98cc73ea4ad3e4.png) ![](RackMultipart20200425-4-4lmrq_html_fcc4a214f0f8a421.png)

# COMPARISON AND ANALYSIS

Comparison Table

|
 | Test Accuracy (%) | Test F-1 Score |
| --- | --- | --- |
| TASK 1 |
| [VGG16 FCLayer530, learning rate 0.001](#_Toc38740212) | 90.06 | 0.921 |
| [VGG16 FCLayer530, learning rate 0.00001](#_Toc38740213) | 93.60 | 0.947 |
| [VGG16 FCLayer 2000, learning rate 0.001](#_Toc38740214) | 93.46 | 0.944 |
| [VGG16 FCLayer : 2000 , learning rate 0.00001](#_Toc38740215) | 91.0 | 0.926 |
| [RESNET FCLayer530, learning rate 0.001](#_Toc38740216) | 85.06 | 0.864 |
| [RESNET FCLayer530, learning rate 0.00001](#_Toc38740217) | 86.86 | 0.885 |
| [RESNET FCLayer 2000, learning rate 0.00001](#_Toc38741447) | 85.26 | 0.870 |
| TASK 2 |
| [VGG Net, learning rate 0.001](#_Toc38741662) | 96.0 | 0.966 |
| [VGG Net 16, convolutional layers FREEZED, learning rate 0.001](#_Toc38741663) | 96.6 | 0.971 |
| [VGG Net 16, convolutional layers freezed [0,2,5,10,24,28], learning rate 0.001](#_Toc38741664) | 97.13 | 0.975 |
| [Res Net 18, No Freezing learning rate 0.001](#_Toc38741665) | 96.66 | 0.971 |
| [Res Net 18, FC layers freeze, learning rate 0.001](#_Toc38741666) | 96.8 | 0.973 |
| [Res Net 18,convolution 1 and 2 layers freeze, learning rate 0.001](#_Toc38741667) | 95.26 | 0.976 |

It could be compared from the graphs over epochs that decreasing the learning rate, smooth the curve and the curve goes towards better accuracy and loss with smoothness, whereas, high learning rate tends to make abrupt changes, but it is also evident from the graphs that decreasing the learning rate makes the learning slow.

From Task 1, It could be illustrated that increasing the number of neurons also increase the accuracy and also the F1-score. Therefore, as the number of neurons is very less i.e., 530 and corresponding accuracy is 90%, whereas, increasing the number of neurons to 2000, the accuracy increased to 93.4. Similarly, in task 2 it is increased to 96%.

From Task 2, for some convolutional layers freeze, we observe that the accuracy is increased to 97.13 %. Whereas, whole unfreeze network gives 96.6% accuracy. Therefore, it could be said that the intermediate layers frozen, which could learn several parameters can give better results.

From the images of wrong identification, there are images which have contrast problem. The network is predicting them wrong; it is supposed that such networks are learning brightness at particular places rather than texture or behavior.

Some of the X-Rays also have tubes or medical apparatus, it could be seen that such images are also predicted wrong.

 Overall, the ResNet predict CoVID19 cases better than VGG, but with the fine tuning it is find that VGG16 is slightly better performing on the dataset, this behavior is mainly due to the higher number of convolution layers and the dataset properties.
