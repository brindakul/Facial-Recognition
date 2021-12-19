import cv2
import os
import sys
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np
dim = (224,224)

model_name = sys.argv[-1]
print(model_name)
model = load_model(f'models/{model_name}/model.h5')
LABELS = open(f"models/{model_name}/LABELS.txt").read().split('\n')
detector = MTCNN()
print(model.summary())



path  = 'test_video'
def predict(img):
	# img = cv2.imread(os.path.join(path,gt,imgf))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	img = cv2.resize(img, dim)
	img = np.array(img)/255
	im = np.expand_dims(img, axis=0)
	res = model.predict(im)
	pred = LABELS[np.argmax(res[0])]
	return pred,res[0][np.argmax(res[0])]

for fname in os.listdir(path):
  cnt = 0
  gt = fname.split('.')[0]
  print(gt)

  rotate= False
  stream = cv2.VideoCapture(f"{path}/{fname}")
  _, frame = stream.read()
  width,height  = frame.shape[:2]   # float `width`
  # if height>width:
  #   rotate = True
  #   print("rotaing")
  #   frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
  #   height,width  = frame.shape[:2]
  # print(rotate)
  height = stream.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)  # float `height`
  out = cv2.VideoWriter(f'output_vid/{gt}.avi',
                        cv2.VideoWriter_fourcc('M','J','P','G'), 
                        20, (width,height))
  cnt = 0
  while True:
    # grab the frame from the threaded video file stream
    grabbed, frame = stream.read()
    

    if not grabbed:
      break

    
    # if rotate:
    #   frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    #   width,height  = frame.shape[:2]
    img = frame.copy()
    faces = detector.detect_faces(img)
    for result in faces:
      x, y, w, h = result['box']
      # print(x,y,w,h)
      
      face = img[y:y + h, x:x + w]
      print(face.shape)
      pred,conf = predict(face)
      text = "{}:{:.2f}".format(pred,conf*100)
      print(text)
      yhat = y - 10
      cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 2)
      cv2.putText(frame, text, (x, yhat),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
      cv2.imwrite(f'img_{cnt}.jpg',face)
      cnt+=1

    cv2.putText(frame,'Testing', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
    # show the frame and update the FPS counter
    # cv2.imshow("Frame", frame)
    out.write(frame)
    

    # key = cv2.waitKey(1) & 0xFF
    # # if the `q` key was pressed, break from the loop
    # if key == ord("q"):
    #   break
  out.release()


