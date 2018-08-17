"""
训练DCGAN
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" #不指定GPU,强制使用CPU进行计算
from network import *
import glob
import numpy as np
from scipy import misc

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto(device_count={'gpu':0})  #强制使用CPU计算，否则导致显卡占用过高，让显卡温度过高，自动熄屏
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))




def train():
    # 获取训练数据
    if not os.path.exists('images'):
        raise Exception('不存在images文件夹，请添加')
    data = []
    for image in glob.glob('images/*'):
        image_data = misc.imread(image)  # imread利用PIL读取图片数据
        data.append(image_data)

    input_data = np.array(data)

    # 将数据标准化成-1到1之间的取值，这也是tanh激活函数的输出范围
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # 单个像素点的值为0-255

    # 构造生成器和判别器
    g = generator_model()
    d = discriminator_model()

    # 构建生成器和判别器组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # 优化器用Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # 配置生成器和判别器
    g.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    d_on_g.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    d.trainable = True  # 优化生成器的时候讲判别器固定住 同理 优化判别器的时候讲生成器固定住
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)



    # 开始训练
    for epoch in range(EPOCHS):
        if os.path.exists('generator_weight'):
            print(os.path.exists('generator_weight'))
            g.load_weights('generator_weight')
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[(index * BATCH_SIZE):(index + 1)*BATCH_SIZE]



        # 连续型均匀分布的随机数据(噪声)
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            #生成器的生成的图片数据
            generated_images = g.predict(random_data,verbose=0)

            input_batch = np.concatenate((input_batch,generated_images))
            output_batch = [1] * BATCH_SIZE + [0]*BATCH_SIZE

            #训练判别器，让它具有识别不合格图片的能力
            d_loss = d.train_on_batch(input_batch,output_batch)
            #当训练生成器时，让判别器不可被训练
            d.trainable = False
            #训练生成器，并通过不可被训练的判别器去判别
            g_loss = d_on_g.train_on_batch(random_data,[1]*BATCH_SIZE)
            #恢复判别器可被训练
            d.trainable=True
            #打印损失
            print('step %d generator loss: %f discriminator loss: %f' %(index,g_loss,d_loss))

        #每个EPOCH保存生成的参数
        if epoch % 10 == 9:
            g.save_weights('generator_weight',True)
            d.save_weights('discriminator_weight',True)



if __name__ == '__main__':
    train()
