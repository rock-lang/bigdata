#   01
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
#加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集划分
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#进行数据处理
train_images = train_images/255.0
test_images = test_images/255.0

#搭建神经网络
def create_model():
    #创建神经网络模型
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10)
    ])
    #编译模型
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )
    #返回模型
    return model
#调用模型
new_model=create_model()
#训练模型
new_model.fit(train_images, train_labels, epochs=20)
#保存模型
new_model.save('model/my_model.h5')

#    02
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集划分
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

#数据处理
train_images = train_images/255.0
test_images = test_images/255.0
#搭建神经网络
def create_model():
    #创建神经网络模型
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(10)

    ])
    #编译模型
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )
    #返回模型
    return model
# 调用模型
new_model = create_model()
# 训练模型
new_model.fit(train_images,train_labels,epochs=30)
 # 保存
new_model.save('model/my_model.h5')

#   03
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集划分
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
#数据处理
train_images = train_images/255.0
test_images = test_images/255.0

#搭建神经网络
def create_model():
    #创建神经网络模型
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    #编译模型
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )
    #返回模型
    return model
#调用模型
new_model = create_model()
#训练模型
new_model.fit(train_images,train_labels,epochs=20)
#保存
new_model.save('model/my_model.h5')


#   04
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集划分
(train_images,train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#数据处理
train_images = train_images/255.0
test_images = test_images/255.0

#搭建神经网络
def create_model():
    #创建神经网络模型
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    #编译模型
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )
    return model

#调用模型
new_model = create_model()
#训练模型
new_model.fit(train_images, train_labels, epochs=30)

new_model.save('model/my_model.h5')

#   05
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集划分
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
#数据处理
train_images = train_images/255.0
test_images = test_images/255.0

#搭建神经网络
def create_model():
    #创建神经网络模型
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    # 编译模型
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )
    return model
new_model = create_model()
new_model.fit(train_images,train_labels,epochs=20)
new_model.save("model/my_model.h5")







