import cv2 as cv
from keras.models import load_model
import numpy as np

print(f'cv2 optimized: {cv.useOptimized()}')

model = load_model('model_mnist_hrd 1.h5')
cap = cv.VideoCapture(0)

def predict_number(img_):
    if not (img_ is None or  img_.size == 0):
        img_ = cv.resize(img_, (28, 28), -1)
        img_ = img_.reshape(1, 28, 28, 1)
        img_ = img_.astype('float32')
        img_ /= 255.0

        predicted_number = model.predict_on_batch(img_)
        num = np.argmax(predicted_number)
        return num, predicted_number[0, num]


while 1:
    suc, img = cap.read()
    img_g = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(img_g, 127, 255, cv.THRESH_BINARY_INV)
    contours, hi = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        p = 0.0

        d = []
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)

            if not (500 < area < 15000) or cv.isContourConvex(cnt):
                continue

            x, y, w, h = cv.boundingRect(cnt)
            p = w // 8
            temp_img = thresh[y - p:y + h + p, x - p:x + w + p]
            res = predict_number(temp_img)

            if res is not None:

                cv.rectangle(img, (x - p, y - p),      (x + w + p, y + h + p),  (255, 0, 0),  2)
                cv.rectangle(img, (x - p, y - p - 15), (x + w + p, y - p),      (255, 0, 0), -1)

                cv.putText(img,
                           f'{res[0]}, {res[1] * 100:.2f}%',
                           (x  - p + 2, y - p - 5),
                           cv.FONT_HERSHEY_DUPLEX, .4,
                           (255, 255, 255))


    cv.imshow('result img', img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
