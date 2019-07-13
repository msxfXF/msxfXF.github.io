---
layout: post
title: 来玩玩GAN(1)-使用CycleGAN生成firewatch风格图片
category: DeepLearning
tags: [DeepLearning]
excerpt_separator: <!-- more -->
excerpt: 实战将自然风景图片转换为firewatch风格的图片
typora-copy-images-to: ..\assets\images
---

来玩玩GAN(1)-使用CycleGAN生成firewatch风格图片
# 使用CycleGAN生成firewatch风格图片

***

## 介绍

实战将自然风景图片转换为firewatch风格的图片

<!-- more -->

***

***

## CycleGAN网络原理简介

与GAN相同，CycleGAN也有生成网络和判别网络，损失函数也大同小异，只不过有两组G和D。

下面为CycleGAN网络的结构：

![img](/assets/images/5806754-c32814397100c895.webp)

CycleGAN本质上是两个对称的GAN，构成了一个环形网络。
两个GAN共享两个生成器，并各自带一个判别器，即共有两个判别器和两个生成器。

![1563028981129](/assets/images/1563028981129.png)

![1563029012340](/assets/images/1563029012340.png)

## CycleGAN实战

直接使用GitHub上的代码

- CycleGAN-Keras-master：https://github.com/simontomaskarlsson/CycleGAN-Keras

- CycleGAN-tensorflow：https://github.com/xhujoy/CycleGAN-tensorflow

核心模型代码：

``` python
        def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x
    
    
    
    
    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)
        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)
```





## 训练遇到问题

1. keras-contrib安装问题

   ```
   pip install git+https://www.github.com/keras-team/keras-contrib.git
   ```

2. G不收敛，或者D比G收敛快很多，导致图片无法正常生成

   把 self.generator_iterations 设置的大一点，比如5，意思是fit 5次G，fit 1次D

3. 训练太慢！！！

   推荐大家使用极客云，这是我的邀请链接：http://www.jikecloud.net/register.html?iid=JBwQ-u2Rv-YxzcOzlU6hyA==

   使用邀请链接，各的十元哦~

## 训练结果

暴风哭泣！使用垃圾1050Ti训练了大概65个epoch，（200个epoch，1050Ti要2.5天，2080要12个小时，关键没钱！！），训练结果如下，左边为原图，右边为生成的图片：

![1563029939721](/assets/images/1563029939721.png)

![1563029952735](/assets/images/1563029952735.png)

![1563029962026](/assets/images/1563029974606.png)

![1563030016993](/assets/images/1563030016993.png)

## 参考资料和相关网址

CycleGAN的原理与实验详解：

https://yq.aliyun.com/articles/229300

https://www.jianshu.com/p/64bf39804c80