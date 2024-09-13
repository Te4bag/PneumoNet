# PneumoPred
A Deep CNN image classifier that takes chest x-rays as input and predicts whether or not the patient has pneumonia.<br><br>
![Prediction Plot](https://github.com/Te4bag/PneumoNet/blob/main/Images/Screenshot%202024-09-13%20151008.png)
# Contents
1. Dataset<br/>
2. Tools and libraries<br/>
3. Data Preprcessing<br/>
4. Model Architecture<br/>
5. Training and evaluation<br/>

## 1. Dataset
Dataset is available on Kaggle. [Click Here to go to Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
## 2. Tools and libraries
- Tensorflow
- PIL
- Scikit-Learn
- NumPy
- Matplotlib
## 3. Data Preprocessing
Images were resized to 256px x 256px and normalized between [0,1]
## 4. Model Architecture
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.
![CNN](https://github.com/Te4bag/PneumoNet/blob/main/Images/cnn.gif)
## 5. Training and Evaluation
- The model was trained for 10 epochs with a batch size of 32.
- Loss function used was BinaryCrossentropy
- Adam was used as optimizer with learning rate of 0.0001
- We recieved a validation accuracy of **68.75%** and testing accuracy of **91.03%** 
