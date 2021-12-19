from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation,BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
ImgH, ImgW = 224, 224
batch_size = 32
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/CNN_3l/model.h5',
    save_freq='epoch')


train_datagen = ImageDataGenerator(
    rotation_range=60,
    brightness_range=(0.75,1.25),
    rescale=1./255,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
  directory=r"./dataset/face_images/train/",
  target_size=(ImgH, ImgW),
  color_mode="rgb",
  batch_size=batch_size,
  class_mode="categorical",
  shuffle=True,
  seed=42
)


test_generator = test_datagen.flow_from_directory(
  directory=r"./dataset/face_images/test/",
  target_size=(ImgH, ImgW),
  color_mode="rgb",
  batch_size=batch_size,
  class_mode="categorical",
  shuffle=True,
  seed=42
)

classes_list = train_generator.class_indices
print(classes_list,len(classes_list))
N = len(classes_list)

inputs = Input(shape=(ImgH,ImgW,3))                 # input layer
x = Conv2D(32, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(128, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(256, (3, 3), padding="same")(x)
x = Conv2D(512, (3, 3), padding="same")(x)
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)     # hidden layer
outputs = Dense(N, activation='softmax')(x) # output layer

model = tf.keras.Model(inputs, outputs)
print(model.summary())
# model = load_model('models/CNN_3l/model.h5')
# print('Model Loaded!!')

opt =  tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=test_generator,verbose=1,
        callbacks=[model_checkpoint_callback])
model.save_weights('first_try.h5')