from cProfile import label
from tokenize import PlainToken
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000

np.random.seed(int(time.time()))

xs = np.random.uniform(-2, 2, NUM_SAMPLES)
np.random.shuffle(xs)
print(xs[:5])

ys = (xs + 1.7)*(xs + 0.7)*(xs - 0.3)*(xs - 1.3)*(xs - 1.9) + 0.2
print(ys[:5])

#plt.plot(xs, ys, 'b. ')
#plt.show()

ys += 0.1*np.random.randn(NUM_SAMPLES)

#plt.plot(xs, ys, 'g. ')
#plt.show()

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT]) #앞의 것에 800개
y_train, y_test = np.split(ys, [NUM_SPLIT])

#plt.plot(x_train, y_train, 'b. ', label = 'train')
#plt.plot(x_test, y_test, 'r. ', label = 'test')
#plt.legend()
#plt.show()
model_f = tf.keras.Sequential([  #keras 라이브러리에서 제공하는 tensor를 생성하고, 입력 노드의 개수를 정해줌.
    tf.keras.layers.InputLayer(input_shape = (1,)),
    tf.keras.layers.Dense(16, activation = 'relu'), #16개의 노드생성
    tf.keras.layers.Dense(16, activation = 'relu'), #Dense는 내부적으로 y = activation(x*w + b)식을 생성
    tf.keras.layers.Dense(1) #출력 신경망 생성
])

model_f.compile(optimizer = 'rmsprop', loss = 'mse')

p_test = model_f.predict(x_test)

# plt.plot(x_test, y_test, 'b. ', label = 'actual')
# plt.plot(x_test, p_test, 'r. ', label = 'predicted')
# plt.legend()
# plt.show()

model_f.fit(x_train, y_train, epochs = 200) #x_train, y_train 데이터에 맞도록 학습함을 의미, epochs는 학습 횟수.

p_test = model_f.predict(x_test)

plt.plot(x_test, y_test, 'b. ', label = 'actual')
plt.plot(x_test, p_test, 'r. ', label = 'predicted')
plt.legend()
plt.show()