# Automatically quantifying knee osteoarthritis severity from X-ray images using a faster R-CNN model #

*Authors: Suhas Suresha, Eni Halilaj and Łukasz Kidziński*
### Introduction

This code respoitory, implemented in Python, can be used to automatically quantify knee osteoarthritis severity (using the Kellgren & Lawrence grading system) from X-ray images using a trained faster region convolutional neural network (R-CNN) model. The model was trained on a dataset consisting of 4214 X-ray images of the knees (including left and right leg) obtained from the Osteoarthritis Initiative (OAI) dataset, courtesy of the Mobilize Center at Stanford University. 

### License

The project 'Knee OA staging' is released under the MIT License (refer to the LICENSE file for details).

### Methodology

We train a **faster region convolutional neural network** (*faster R-CNN*) with our dataset to achieve the following two objectives: 1. **extract the knee joint regions** from the X-ray images and 2. **classify the extracted knee-joint regions** based on KL score. Faster R-CNN achieves both of these objectives together, where it first learns to identify potential knee-joint regions using the region proposal network (RPN) and then classifies the knee-joint regions according to KL grades using a fast R-CNN object classification network.

This approach combines RPN, which predicts knee-joint region and object classification network, which predicts KL grades, into a single network by sharing their convolutional features to improve prediction speed. For our faster R-CNN model architecture, we explore the VGG-16 model, which has 13 shareable convolutional layers. 

For every image, faster R-CNN predicts 300 knee-joint region proposals with probability estimates (confidence scores) for all classes (KL scores from 0-4) for each of the proposals. Hence for each image, we create a confidence score matrix with rows corresponding to the knee-joint proposals and the columns corresponding to the probability estimates for each label. In order to predict the label for a given image from the faster R-CNN predictions, we implemented the following methodologies:

1.) **Maximum average confidence score :** We choose the label with the highest average confidence score over all the predicted knee-joint region proposals.

2.) **Support vector machine classifier :** We train a support vector machine (SVM) classifier on the training images with their confidence score matrix as input feature.

3.) **Random forest classifier :** We train a random forest classifier on the training images with their confidence score matrix as input feature. 

4.) **Multi-layer perceptron classifier :** We train a multi-layer perceptron classifier on the training images with their confidence score matrix as input feature. 

### Using the code repository to qunatify knee OA severity 

#### Requirements
1.) Install CUDA on your system. *Note:* Even if using a CPU, CUDA libraries are necesary to run Caffe.  

2.) Python packages needed : 
      •	cython 
      •	numpy
      •	opencv-python
      •	easydict
      •	matplotlib
      •	scikit-image
      •	protobuf
      •	scikit-learn 
      •	pyyaml

#### Steps to Follow
1.) Clone the repository
```Shell
git clone --recursive https://github.com/suhas1992/knee_OA_staging.git
```
2.) Build the Cython Modules: 
```Shell
cd knee_OA_staging/lib
make
```

3.) Build Caffe and Pycaffe:
```Shell
cd knee_OA_staging/caffe-fast-rcnn
make -j8 && make pycaffe
```

4.) Download trained faster R-CNN model to quantify knee OA severity:
```Shell
cd knee_OA_staging/data
mkdir faster_rcnn_models
cd faster_rcnn_models
wget https://transfer.sh/Czqow/VGG16_faster_rcnn_final.caffemodel
```

5.) Demo images are available in knee_OA_staging/data/test_images. To test the model on the demo images, do the following:
```Shell
cd knee_OA_staging
python tools/demo_all_predictions.py [GPU_ID] [CPU] [model] 
# GPU_ID : GPU you want to train on
# CPU : Use CPU mode (overrides --gpu)
# model : Options from {avg, SVM, Random_forest, MLP}. Used to predict the label for a given image from the faster R-CNN predictions. 
# Ex: python tools/demo_all_predictions.py --cpu --model MLP 
```
The output images are available in knee_OA_staging/data/output_test_images.

6.) To test the model on any new X-ray image (only works on X-ray image with one knee), add the image in knee_OA_staging/data/test_images and repeat Step 5 above. Output will be avilable in knee_OA_staging/data/output_test_images. 

###  Reference

A significant portion of the code repsoitory is borrowed from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn) by Shaoqing Ren, Kaiming He, Ross Girshick and Jian Sun. 
