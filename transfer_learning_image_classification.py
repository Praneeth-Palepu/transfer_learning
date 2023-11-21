'''
Image classification for custom datasets using transfer learning.
'''
import os
import joblib
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications import imagenet_utils, VGG16, VGG19, MobileNet, InceptionV3, ResNet50
from keras.models import Sequential
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ImageClassifier:
    '''
    The ImageClassifier class contains methods to 
    classify an image dataset using pretrained models. 
    '''
    def __init__(self, feature_extractor = 'vgg16', target_size=(224, 224)):
        '''
        Description
        -----------
        This method instantiates the ImageClassifier class.

        Arguments
        ---------
        feature_extractor: optional: str
        The feature_extractor is the pretrained model to be used for feature extraction\
            from the provided images. Acceptable values are \
                'vgg16', 'vgg19', 'mobilenet', 'inception', 'resnet'
        
        target_size: optional: tuple
        The target size of each image. Each image in the dataset will be resized to this size.

        Return Type
        -----------
        None
        '''
        self.target_size = target_size
        self.feature_extractor = feature_extractor

    def dataset_loader(self, img_base_path, split_size=0.2):
        '''
        Description
        -----------
        The dataset_loader method loads images from the images base path.
        The images must be organized in unique folders (Each class in one folder\
            with the name of the class as the folder name).
        Only files with bmp, jpg, jpeg, png and tiff formats are accepted.
        All images are resized to a target size specified during initialization.
        All images are preprocessed using keras imagenet_utils.
        The images are split into training and testing with a test size as specified\
            in the argument (split_size).
        The target class labels are not label encoded and hence cannot be converted to\
            categorical for training neural networks directly. 

        Arguments
        ---------
        img_base_path: path
        Path where all the folders for each class containing the images are stored.

        split_size: float
        The test size used to split the entire dataset into training and testing.

        Return Type
        -----------
        Two tuples
        First tuple: train_images_array, label_train_images
        Second tuple: test_images_array, label_test_images
        '''
        images, labels = [], []
        for label in os.listdir(img_base_path):
            label_path = os.path.join(img_base_path, label)
            if os.path.isdir(label_path):
                for img in os.listdir(label_path):
                    _, extension = os.path.splitext(img)
                    if extension.lower() in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']:
                        img_path = os.path.join(label_path, img)
                        image = load_img(img_path, target_size=self.target_size)
                        img_arr = img_to_array(image)
                        img_arr_processed = imagenet_utils.preprocess_input(img_arr)
                        images.append(img_arr_processed)
                        labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        xtr, xte, ytr, yte = train_test_split(images, labels, test_size=split_size, random_state=0)
        return (xtr, ytr), (xte, yte)

    def vgg16_feature_extractor(self, image_array):
        '''
        Description
        -----------
        This methods extracts features from an image using VGG16 pretrained model.
        The image must be preprocessed using imagenet_utils. \
            (Automatically handled if the images are loaded using the dataset_loader).
        
        All images must be colored images.
        The extracted features are flattened into a single array.

        Arguments
        ---------
        The preprocessed image_array for feature extraction.

        Return Type
        -----------
        Array of extracted features using VGG16.
        '''
        shape = (self.target_size[0], self.target_size[1], 3)
        vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=shape)
        model = Sequential([
            vgg_model,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features

    def vgg19_feature_extractor(self, image_array):
        '''
        Description
        -----------
        This methods extracts features from an image using VGG19 pretrained model.
        The image must be preprocessed using imagenet_utils. \
            (Automatically handled if the images are loaded using the dataset_loader).
        
        All images must be colored images.
        The extracted features are flattened into a single array.

        Arguments
        ---------
        The preprocessed image_array for feature extraction.

        Return Type
        -----------
        Array of extracted features using VGG19.
        '''
        shape = (self.target_size[0], self.target_size[1], 3)
        vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=shape)
        model = Sequential([
            vgg_model,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features

    def mobilenet_feature_extractor(self, image_array):
        '''
        Description
        -----------
        This methods extracts features from an image using mobilenet pretrained model.
        The image must be preprocessed using imagenet_utils. \
            (Automatically handled if the images are loaded using the dataset_loader).
        
        All images must be colored images.
        The extracted features are flattened into a single array.

        Arguments
        ---------
        The preprocessed image_array for feature extraction.

        Return Type
        -----------
        Array of extracted features using mobilenet.
        '''
        shape = (self.target_size[0], self.target_size[1], 3)
        vgg_model = MobileNet(include_top=False, weights='imagenet', input_shape=shape)
        model = Sequential([
            vgg_model,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features

    def inception_feature_extractor(self, image_array):
        '''
        Description
        -----------
        This methods extracts features from an image using inception pretrained model.
        The image must be preprocessed using imagenet_utils. \
            (Automatically handled if the images are loaded using the dataset_loader).
        
        All images must be colored images.
        The extracted features are flattened into a single array.

        Arguments
        ---------
        The preprocessed image_array for feature extraction.

        Return Type
        -----------
        Array of extracted features using inception.
        '''
        shape = (self.target_size[0], self.target_size[1], 3)
        vgg_model = InceptionV3(include_top=False, weights='imagenet', input_shape=shape)
        model = Sequential([
            vgg_model,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features

    def resnet_feature_extractor(self, image_array):
        '''
        Description
        -----------
        This methods extracts features from an image using resnet pretrained model.
        The image must be preprocessed using imagenet_utils. \
            (Automatically handled if the images are loaded using the dataset_loader).
        All images must be colored images.
        The extracted features are flattened into a single array.

        Arguments
        ---------
        The preprocessed image_array for feature extraction.

        Return Type
        -----------
        Array of extracted features using resnet.
        '''
        shape = (self.target_size[0], self.target_size[1], 3)
        vgg_model = ResNet50(include_top=False, weights='imagenet', input_shape=shape)
        model = Sequential([
            vgg_model,
            Flatten()
        ])
        image_features = model.predict(image_array)
        return image_features

    def classifier(self, model, dataset_path, split_size, model_name='classifier'):
        '''
        Description
        -----------
        The classifier methods loads an image dataset from the given path.\
            Divides the dataset into training and testing sets and extracts \
                features (using a pretrained model specified during initialization) \
                    from the image array.
        Finally it uses the initialized model (passed as parameter) to fit the training data\
            and make a prediction on the test data. The trained model is saved in the default\
                working directory.
        
        Arguments
        ---------
        model: Machine Learning model
        The model used for classification. The model must have a fit method to \
            fit x & y . x = training_input and y = training_output
        
        dataset_path: Path
        The path to the dataset that contains class labels as folder_name and \
            images of each class in the respective folder.

        split_size: float
        The percentage of test data size based on which the dataset is \
            divided into training and testing groups.

        model_name: optional: str
        The name on which the machine learning model is stored in the working directory.\
            If no name is provided the default name is classifier.
        
        Return Type: float
        -----------
        The accuracy of the model on test data.
        '''
        (xtr, ytr), (xte, yte) = self.dataset_loader(img_base_path = dataset_path,
        split_size=split_size)
        if self.feature_extractor == 'vgg16':
            tr_features = self.vgg16_feature_extractor(xtr)
            te_features = self.vgg16_feature_extractor(xte)
        elif self.feature_extractor == 'vgg19':
            tr_features = self.vgg19_feature_extractor(xtr)
            te_features = self.vgg19_feature_extractor(xte)
        elif self.feature_extractor == 'mobilenet':
            tr_features = self.mobilenet_feature_extractor(xtr)
            te_features = self.mobilenet_feature_extractor(xte)
        elif self.feature_extractor == 'inception':
            tr_features = self.inception_feature_extractor(xtr)
            te_features = self.inception_feature_extractor(xte)
        elif self.feature_extractor == 'resnet':
            tr_features = self.resnet_feature_extractor(xtr)
            te_features = self.resnet_feature_extractor(xte)

        model.fit(tr_features, ytr)
        prediction = model.predict(te_features)
        accuracy = accuracy_score(yte, prediction)
        joblib.dump(model, f"{model_name}.joblib")
        return round(accuracy*100, 2)

    def prediction(self, base_path, classifier_model):
        '''
        Description
        -----------
        The function takes a test dataset as an input and predicts the \
            class of each input image.
        The function loads the images from the test folder, preprocesses the \
            images, extracts features and uses the trained model to make a prediction\
                of the class of each of the image.
        
        Arguments
        ---------
        base_path: path
        The directory path where all the test images are stored.

        classifier_model: joblib: model_file
        The trained ML model joblib file that classifies the features extracted from \
            the pretrained model to appropriate classes.

        Return Type
        -----------
        result_dict: dict
        The result dictionary where the image name from the test folder is the key \
            and the predicted class of the image is the value.
        '''
        images_test, labels = [], []
        for file_name in os.listdir(base_path):
            file_path = os.path.join(base_path, file_name)
            _, extension = os.path.splitext(file_path)
            if extension.lower() in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff']:
                img = load_img(file_path, target_size=self.target_size)
                img_array = img_to_array(img)
                img_array_processed = imagenet_utils.preprocess_input(img_array)
                images_test.append(img_array_processed)
                labels.append(file_name)
        images_arr_test = np.array(images_test)
        if self.feature_extractor == 'vgg16':
            img_features = self.vgg16_feature_extractor(images_arr_test)
        elif self.feature_extractor == 'vgg19':
            img_features = self.vgg19_feature_extractor(images_arr_test)
        elif self.feature_extractor == 'mobilenet':
            img_features = self.mobilenet_feature_extractor(images_arr_test)
        elif self.feature_extractor == 'inception':
            img_features = self.inception_feature_extractor(images_arr_test)
        elif self.feature_extractor == 'resnet':
            img_features = self.resnet_feature_extractor(images_arr_test)
        prediction = classifier_model.predict(img_features)
        result_dict = dict(zip(labels, prediction))
        return result_dict
