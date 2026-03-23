**ANN:** THe (ANN) model is built to predict whether a passenger survived or not at the tragedy of titanic, based on features like age, gender, fare, and passenger class. The output is binary, where 1 represents survival and 0 represents not survived.The model started with low accuracy but improved as training continued. After some epochs, it learned the patterns in the data and reached around 80% accuracy. The training and validation results stayed close, which means the model is not overfitting much.

On test data, the model achieved about 78% accuracy, which shows it works well on unseen data. The confusion matrix shows that it predicts non-survivors better than survivors, but overall the performance is good.

In simple words, the ANN learned from the data properly and is able to make reasonable predictions.

**CNN:** The model is built for image classification using the CIFAR-10 publicly available dataset. The images were first normalized so that pixel values fall between 0 and 1. The model consists of convolution layers, max pooling layers, a flatten layer, and fully connected dense layers. The convolution layers extract image features, while the pooling layers reduce dimensionality. The dense layers perform the final classification into 10 classes. The model was trained using the Adam optimizer and sparse categorical crossentropy loss function. Its performance was evaluated using accuracy, precision, recall, and F1-score. Accuracy shows the overall correctness, precision shows how many predicted classes were correct, recall shows how many actual classes were identified, and F1-score gives the balance between precision and recall.

**CNN Based Image Recognition System using Flask: ** I developed an image recognition system using a Convolutional Neural Network (CNN) and Flask. I collected a dataset of three different persons, each having 400 training images and 100 testing images with different angles, lighting, and expressions. The images were resized and normalized before training.

The CNN model consists of convolution layers, pooling layers, and dense layers, which help in extracting features and classifying the images. After training, the model achieved good accuracy and was able to correctly identify different persons.

The trained model was then integrated into a Flask web application. The system allows the user to upload an image, and the model predicts the name of the person. The uploaded image and predicted label are displayed on the webpage.

Overall, the system works effectively and demonstrates how deep learning can be used for real-world image recognition tasks.
