import cv2 as cv
from keras.models import load_model
import numpy as np
import time

model = load_model('model_mnist.h5')

print(f'This Script Optimized By GPU: {cv.useOptimized()}')
print('Wait Camera will open...')
cap = cv.VideoCapture(0)
print('Camera Opened, done.')

ptime = 0

def predictNumberInImg(img_):

  img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY, -1)
  img_ = cv.resize(img_, (28,28))
  img_ = img_.reshape(1, 28, 28, 1)
  img_ = img_.astype('float32')
  img_ /= 255.0

  res = model.predict_on_batch(img_)
  i = np.argmax(res)
  return i, res[0, i]

while 1:
    suc, img = cap.read(-1)
    resVal = predictNumberInImg(img)

    img = cv.resize(img, (1000, 700), -1)
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime

    cv.putText(img, f'fps:{fps}', (10, 50), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
    cv.putText(img, f'predicated digit:{resVal[0]}', (10, 100), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    cv.putText(img, f'with confidence: {round(resVal[1]*100, 2)}', (10, 150), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv.imshow('result', img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
