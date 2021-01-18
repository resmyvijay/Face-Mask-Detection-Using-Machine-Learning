# Machine-Learning-Face-Mask-Detection
The Covid=19 virus has affected large number of people all around the world due to its highly infectious nature and no cure available. The Covid-19 virus spreads aerially when the infected person coughs, sneezes, talks or sings. One way of preventing the spread of the virus is through wearing facemasks that prevent spread of the virus from the infected person to a non-infected person. It has become mandatory in many countries to wear face masks.
Through use of machine learning, we can detect whether a person is wearing a face mask or not by analyzing the image of a person or a live video stream. This project is on creating a python program for detecting face masks in images. It can determine to a high level of accuracy if a person seen through an image or a video stream is wearing a face mask or not. This can be used at entrances to public places in order to enforce the regulation of wearing masks.
This project uses Python with Tensorflow/Keras along with OpenCV Classifier to train a face mask detector and detect facemasks in images.
1.1	### Objectives 
To build a face mask detector with computer vision and deep learning using Python, Open CV and TensorFlow/Keras.
The project consists of two phases:
Phase 1: Train Face Mask Detector
Phase 2: Apply Face Mask Detector

1.1.1 	Tools
The tools used for developing the two phases are:
1)	Google Colab
2)	Python
3)	Keras
4)	Tensor Flow

1.1.1.1	 Data Source
The dataset for the project is a face mask detection dataset from Kaggle. It consists of 6024 images containing images of people with face masks and without face masks. 
A CSV file contains information about each image and classifies each image based on whether the person is wearing a face mask or not.
 
 ### ANALYSIS AND DESIGN
There are two phases to the development of the face mask detector.
1)	Training
In the training phase, the fask mask detection dataset is downloaded from the disk, a model is trained using this dataset, and then saved to be used during the deployment phase.
2)	Deployment
3)	Once the face mask detector is trained, it is loaded onto the mask detector, which is then used to detect if a face image has a mask or no mask.

 


2.1.1.1 ALGORITHMS 
### Phase 1: Train Face Mask Detector
In this phase, we are going to make a classifier that can differentiate between faces with masks and faces without masks. 
In order to create this classifier, we need data in the form of images. This is obtained from the dataset which contains images of faces with mask and without masks.
As the images are very less in number, we cannot train the neural network from scratch. Instead, we finetune a pre-trained network called MobileNetV2 which is trained on the ImageNet dataset. 
The pre-trained model is customized according to our needs. To do this, we remove the top layers of the pre-trained model and add few layers of our own. 
The dataset is then split into training and testing sets to evaluate them. The next step is data augmentation which significantly increases the diversity of data, without actually collecting new data. Data augmentation techniques such as cropping, shearing, rotation and horizontal flipping are used.
The model is then compiled and trained on the augmented data.
After the model is trained, it is saved for later use.

 ### Phase 2: Apply Facemask Detector
With the trained model, we can now apply it to images and detect is the image has a face mask or not. 
First, we need images of faces. This is obtained by opening the webcam through OpenCV code. Next, we get the frames from the webcam stream using the read() function. 
In actual application, this would be done in infinite look to get all the frames till the time we want to close the stream. But, for the sake of project, we use Google Colab code snippet to capture an image and send it for detection.
The face is detected using the OpenCV classifier. Open CV Classifier has several trained Haar Cascade models which are saved as XML files. The “haarcascade_frontalface_alt2.xml” is used in this project. 
For the classifier to work, the frame is converted to grayscale. Then, only the face is detected and a rectangle is drawn over it. Next, the face is fed into the model for prediction.
The facecascade model contains all faces which are stored in a list called the faces_list. The preds list is used to store the predictions made by the mask detector model.
When the face is fed to the facecascade model, a prediction is made whether the face contains a mask or not. If the prediction for face mask is higher than the prediction for no face mask, the label “Mask” is outputted, else “No Mask” is outputted.
