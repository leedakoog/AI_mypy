from cProfile import label
from tokenize import PlainToken
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(0.1, 5, NUM_SAMPLES)
np.random.shuffle(xs)

ys = 1.0/xs
ys += 0.1*np.random.randn(NUM_SAMPLES) #8~16 인공 신경망 학습에 사용할 데이터 생성

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT]) #18~21 데이터를 훈련 데이터와 실험 데이터로 나눔

model_f = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (1,)),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(1)
]) # 23~28 인공 신경망 구성에 필요한 입력 층, 은닉 층, 출력 층을 구성함

model_f.compile(optimizer = 'sgd', loss = 'mean_squared_error') #인공신경망 내부 망을 구성하고, 학습에 필요한 오차함수, 최적화함수를 설정

model_f.fit(x_train, y_train, epochs = 100) #인공신경망 학습

p_test = model_f.predict(x_test) # 학습시킨 인공 신경망을 이용하여 새로 들어온 데이터에 대한 예측을 수행함

plt.plot(x_test, y_test, 'b. ', label = 'actual')
plt.plot(x_test, p_test, 'r. ', label = 'predicted')
plt.legend()
plt.show()  #나머지는 출력