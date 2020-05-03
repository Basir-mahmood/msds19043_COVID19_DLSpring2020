### Dataset:
https://drive.google.com/file/d/1eytbwaLQBv12psV8I-aMkIli9N3bf8nO/view?usp=sharing

### Experiment 1
VGG16 Focal Loss with SGD Optimizer learning rate= 1e-2, gamma = 2, alpha = 0.25
 
VGG-16 each, end-to-end with focal loss


Results:
![Capture](https://user-images.githubusercontent.com/64276129/80923053-89869500-8d9a-11ea-8057-b030500f9654.JPG)
![Capture](https://user-images.githubusercontent.com/64276129/80923094-c5215f00-8d9a-11ea-8ae6-1866021d1f06.JPG)


### Experiment 4 
ResNet FLoss  with SGD Optimizer learning rate= 1e-2, gamma = 2, alpha = 0.25

ResNet each, end-to-end with focal loss

Results :
![Capture](https://user-images.githubusercontent.com/64276129/80923187-70caaf00-8d9b-11ea-9ffb-c9107e1f541f.JPG)
![Capture](https://user-images.githubusercontent.com/64276129/80923192-8049f800-8d9b-11ea-895a-a276381f72be.JPG)




### Experiment 2
ResNet18 BCE Loss with SGD Optimizer learning rate= 1e-2, gamma = 2, alpha = 0.25


ResNet18 each, end-to-end without focal loss


Results:
![Capture](https://user-images.githubusercontent.com/64276129/80923120-f4d06700-8d9a-11ea-8d1c-49e8089d5493.JPG)
![Capture](https://user-images.githubusercontent.com/64276129/80923143-17628000-8d9b-11ea-86a9-1dc1840893d1.JPG)

### Experiment 3

VGG BCE Loss with SGD Optimizer learning rate= 1e-2, gamma = 2, alpha = 0.25

 VGG-16 each, end-to-end without focal loss


Results :

![Capture](https://user-images.githubusercontent.com/64276129/80923160-3f51e380-8d9b-11ea-8bcc-8627907132a1.JPG)
![Capture](https://user-images.githubusercontent.com/64276129/80923172-52fd4a00-8d9b-11ea-91d0-b0c999339a96.JPG)


### Best Experiments Results:
Res18 Fine tuning BCELoss with all freezed layers except for last 2

![Capture](https://user-images.githubusercontent.com/64276129/80923204-a66f9800-8d9b-11ea-9cba-55f38a2788cf.JPG)
![Capture](https://user-images.githubusercontent.com/64276129/80923221-bdae8580-8d9b-11ea-9ffd-747d824c2a6a.JPG)


### Comparison 
![Capture](https://user-images.githubusercontent.com/64276129/80923343-8ee4df00-8d9c-11ea-9f85-adf4b38b13af.JPG)

From Comparing the confusion matrices for the experiments and looking at the most significant class is covid-19. Where the most important classification is False Negative. Moreover, True Positive is also very important class. And the validation data set is more significant the training dataset. By looking at the experiments results. We can see that the best results are obtained from the experiment 15. Where the False negative is minimum i.e.,8 and true positive are highest i.e.,20.

Overall, experiment number 15 has the best results.

### Notebook is provided where all the results are present.
###Weights link:
https://drive.google.com/open?id=1jJ3VmSKyff-24oVgjyaM88usNoLMlcvo

###Dataset link:
https://drive.google.com/file/d/1eytbwaLQBv12psV8I-aMkIli9N3bf8nO/view?usp=sharing

