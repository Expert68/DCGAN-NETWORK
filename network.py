"""
DCGAN 深度卷积生成对抗网络
"""
import tensorflow as tf


# Hyperparameters 超参数
EPOCHS = 100  # 总共训练的轮数
BATCH_SIZE = 32  # 一次训练图片的数量  #BATCH_SIZE过大会出现tensorflow.python.framework.errors_impl.ResourceExhaustedError,导致内存溢出
LEARNING_RATE = 0.0002
BETA_1 = 0.5


# 定义判别器模型

def discriminator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64  # 64个过滤器
                                     , (5, 5)  # 过滤器在二维的大小是5*5
                                     , padding='same',
                                     input_shape=(64, 64, 3)  # 64像素RGB
                                     ))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))  # 1024个神经元的全连接层
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(1))  # 一个神经元的全连接层
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model


# 定义生成器模型
# 从随机数来生成图片

def generator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))  # 全连接层 输入的维度为100,1*100 1024个神经元
    model.add(tf.keras.layers.Dense(128 * 8 * 8))  # 8192个神经元
    model.add(tf.keras.layers.BatchNormalization())  # 批标准化
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128, 8, 8)))  # 8*8像素
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 16*16像素
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 32*32像素
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 64*64像素
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding='same'))  # 获得的图片为3层 RGB
    model.add(tf.keras.layers.Activation('tanh'))
    return model


# 构造一个Sequential对象，包含一个生成器和一个判别器
# 输入 ->生成器 -> 判别器 ->输出



def generator_containing_discriminator(generator,discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时 判别器不可被训练
    model.add(discriminator)
    return model
