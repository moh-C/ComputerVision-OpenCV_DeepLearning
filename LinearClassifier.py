import keras
import numpy as np
import cv2

class LinearClassifier():
    def __init__(self, Rogue, Hover):
        self.epochs = 200
        self.hover = self.preprocess_data(Hover)
        self.rogue = self.preprocess_data(Rogue)
        
        self.model = model = keras.models.Sequential([
            keras.layers.InputLayer((80,80,3)),
            keras.layers.Conv2D(64, 3, padding= 'same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, padding= 'same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation= 'sigmoid')
        ])
    
    @staticmethod
    def preprocess_data(image):
        image_toresize = image.copy()
        height, width = image_toresize.shape[:2]
        blank_image = np.zeros((80, 80, 3), np.uint8)
        blank_image[:,:] = (255,255,255)
        l_img = blank_image.copy()
        x_offset = 40 - int(width/2)
        y_offset = 40 - int(height/2)
        l_img[y_offset:y_offset+height, x_offset:x_offset+width] = image_toresize.copy()
        return np.array(l_img[np.newaxis, :, :, :])
    
    def prep_data(self):
        _hover = self.hover
        _rogue = self.rogue
        X_train = np.vstack([_hover, _rogue])/255.0
        y_train = np.array([1.0,0.0])
        return X_train, y_train
        
    def train_classifier(self):
        X_train, y_train = self.prep_data()
        self.model.compile(optimizer= keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0), loss= keras.losses.binary_crossentropy)
        self.model.fit(X_train, y_train, epochs= self.epochs, verbose=0)
        return self.model
