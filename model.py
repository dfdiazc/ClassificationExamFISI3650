import numpy as np
import tensorflow as tf


class Model():
    
    def __init__(self):
        self.model = tf.keras.models.load_model('model.h5')
        
    def predict(self, file_path):
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(50, 50))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = self.model.predict(img_array)
        prediction = np.argmax(predictions)
        
        return prediction
        
    def train_model(self):
        data_dir = 'Rice_Image_Dataset/Train'
        batch_size = 64
        img_height = 50
        img_width = 50
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2,
            subset = 'training',
            seed = 42,
            image_size = (img_height, img_width),
            batch_size = batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split = 0.2,
            subset = 'validation',
            seed = 42,
            image_size = (img_height, img_width),
            batch_size = batch_size
        )
        
        class_names = train_ds.class_names
        num_classes = len(class_names)
        
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics = ['accuracy']
        )
        
        model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = 10
        )
        
        model.save('model.h5')