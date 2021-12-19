# from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import cv2
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
ImgH, ImgW = 224, 224


val = int(sys.argv[-1])
kernel = sys.argv[-2]

vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(ImgW, ImgH, 3))

def features(img_pth):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = cv2.resize(img, (ImgW,ImgH))
  img = np.array(img)/255
  im = np.expand_dims(img, axis=0)
  # img = img.astype(np.float32)
  # img /= 255.
  vgg_feature = vgg_model.predict(im)
  # log({"msg:":"FEATURES EXTRACTED !!!"})
  return vgg_feature


data_path = 'dataset/face_images/test'
class_list = os.listdir(data_path)
print("Classes : ", class_list)
num_of_classes = len(class_list)
print("number of classes : ", num_of_classes)

# kernel = 'poly'
# degree = 100
# gamma = 50


model_path = 'models/SVM'
# if kernel is 'poly':
fname = f'svm_model_{kernel}_{val}.pickle'


# elif kernel is 'rbf':
#     fname = f'svm_model_{kernel}_{gamma}.pickle'


modelfile = os.path.join(model_path,fname)

with open(modelfile, 'rb') as f:
    svm_model = pickle.load(f)

ytest = []
ypred = []
for face_name in tqdm(class_list):

  # if face_name in knownNames:
  #   print('Already in Database!!')
  #   continue
    
  # num_of_train_images += len(os.listdir(os.path.join(data_path,face_name)))

  for img_p in os.listdir(os.path.join(data_path,face_name)):

    img_path = os.path.join(data_path,face_name,img_p)
    face_embb = features(img_path)[0].flatten()
    ypred.append(svm_model.predict([face_embb]))
    ytest.append(face_name)


ytest = np.array(ytest)
ypred = np.array(ypred)

print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))




# for x_batch, y_batch in train_generator:
#   print(x_batch[0].shape,y_batch[0].shape,np.argmax(y_batch[0]))


# classes_list = train_generator.class_indices
# print(classes_list,len(classes_list))
# num_classes = len(classes_list)

# f = open('models/CNN_VGG/LABELS.txt','w+')
# lst = list(classes_list.keys())
# f.write('\n'.join(lst))
# f.close()


# vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(ImgW, ImgH, 3))
# # for layer in vgg_model.layers:
# #   layer.trainable = False
# x = vgg_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024,activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(512,activation='relu')(x)
# x = Dropout(0.5)(x)

# predictions = Dense(num_classes,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

# model =tf.keras. Model(inputs=vgg_model.input, outputs=predictions)
# # model = Model(vgg_model.input, outputs)
# print(model.summary())
# # model = load_model('models/CNN_VGG/model.h5')
# # print('Model Loaded!!')

# # opt =  tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit_generator(
#         train_generator,
#         epochs=50,
#         validation_data=test_generator,verbose=1,
#         callbacks=[model_checkpoint_callback])
# model.save_weights('first_try.h5')


