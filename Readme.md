# Project
# Dataset
You can read about the dataset here: http://armbench.s3-website-us-east-1.amazonaws.com/index.html
You can download it here (18gb): https://armbench-dataset.s3.amazonaws.com/segmentation/armbench-segmentation-0.1.tar.gz

Download Salmon dataset by running: 
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="qQ1P4jK2ADK2BzncoQrq")
project = rf.workspace("myworkspace-f9vbo").project("salmon-qniqe")
version = project.version(8)
dataset = version.download("coco-segmentation")
                

## Process
### Documentation
- [ ] Presentation of task 
- [ ] Presentation of dataset
- [ ] Presentation of chosen network structure
- [ ] Presentation of training strategy
### Implementation
- [ ] Split a small section of the data, so we can try thing on our own machines 
- [x] Preproccesing: Convert images from ARMBENCH and COCO to grayscale, since Salmon images are only grayscale. Training on only grayscale can be more efficient. 
- [x] Create a dataset
- [x] Plot the few of the datas and undestand them
- [x] Spilt the dataset into train, validation, and test
- [x] Create a dataloaders for each train, validation, and test
- [x] Implement data augmentation techniques
    - [x] Basic Implementation of augmentations
    - [ ] MOSAIC augmentation
    - [ ] Comprenhensive data augmentation techniques
- [x] Visualize augmentations with mask, bb and class
- [x] Create a Model 
  - [x] Create model with pretrained COCO weights
  - [x] Create model with random weights
  - [ ] Modify model to be more efficient with grayscale images
- [x] Print the summary of the model, this check if it can run
- [x] Find the loss function 
- [x] Find the optimizer
- [x] Implement early stopping
- [x] Train and validate the model
  - [x] Create a histogram
    - [x] Lross
    - [x] Learning Rate
  - [x] Train and validate for amount of epochs 
    - [x] Early stop if validation loss is not going down
    - [x] Save the best model weights
  - [x] Print Values
- [x] Plot the training and validation 
  - [x] Recall
  - [x] Precision
  - [ ] F1 score
- [x] Save configurations for a run
- [x] Add gradient clipping
- [x] Add optimizer
  - [x] SGD
  - [x] AdamW
- [x] Add learning rate scheduler
  - [x] STEPLR
  - [x] Onecycle
  - [x] Cosine
- [x] Test the model
  - [x] Get the Loss 
  - [x] Get the Accuracy
  - [x] Run inference on samples from test dataset to see results
- [ ] Generate plots
  - [ ] Recall
  - [ ] Precision
  - [ ] Average Precision
  - [ ] Mean Average Precision
  - [ ] Inference speed on test set
- [x] Apply gradient clipping
  