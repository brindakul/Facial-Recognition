# from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

## Run this if VGG face is not working the currenct computer

# filename = "/usr/local/lib/python3.7/dist-packages/keras_vggface/models.py"
# text = open(filename).read()
# open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))
# from keras_vggface.vggface import VGGFace



from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation,BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


ImgH, ImgW = 224, 224
batch_size = 16

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

print(gpu_devices)
if len(gpu_devices)>0:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.2
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/CNN_VGG/model.h5',
    save_freq='epoch')


train_datagen = ImageDataGenerator(
		rotation_range=60,
    brightness_range=(0.75,1.25),
		rescale=1./255,
		horizontal_flip=True,
		fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255,fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
	directory=r"./dataset/face_images/train/",
	target_size=(ImgH, ImgW),
	color_mode="rgb",
	batch_size=batch_size,
	class_mode="categorical"
)


test_generator = test_datagen.flow_from_directory(
	directory=r"./dataset/face_images/test/",
	target_size=(ImgH, ImgW),
	color_mode="rgb",
	batch_size=batch_size,
	class_mode="categorical"
)

classes_list = train_generator.class_indices
print(classes_list,len(classes_list))
num_classes = len(classes_list)

f = open('models/CNN_VGG/LABELS.txt','w+')
lst = list(classes_list.keys())
f.write('\n'.join(lst))
f.close()


vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(ImgW, ImgH, 3))
# for layer in vgg_model.layers:
# 	layer.trainable = False
x = vgg_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(num_classes,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model =tf.keras. Model(inputs=vgg_model.input, outputs=predictions)
# model = Model(vgg_model.input, outputs)
print(model.summary())
# model = load_model('models/CNN_VGG/model.h5')
# print('Model Loaded!!')

# opt =  tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=test_generator,verbose=1,
        callbacks=[model_checkpoint_callback])
model.save_weights('first_try.h5')


