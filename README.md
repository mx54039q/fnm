#MODEL-实验日志:
default: 
1. 超参:    L1:fea:GAN=1:0.1:0.01, L2规则化0.0001, BatchSize=5, Adam(momenta=0.5), kernel=5, strides=2, L1损失输入的图范围[0,1],
           权重初始化stddev=0.02
2. 生成器G: encoder用vgg, decoder输入relu7层, 全连接为7x7x256(不加BN和relu), 6层deconv(k=5,s=2)加一层conv(k=1,s=1), 激活relu
           输出接tanh归一化, 再转换到[0,255] 
3. 判别器D: 直接输入G的结果, 并减均值归一化(/127.5-1), 5层conv(k=5,s=2)加1层fc(除了第一层和fc层其余都用BN), 都用lrelu(k=0.2), 
           判别器的真标签为输入侧脸+正脸gt/假标签为输入侧脸+合成正脸
4. 其它:    mirror, 先训练一个epoch的texture loss再加入GAN loss, 1次D2次G, BN分测试和训练, 初始lr=e-4(不变), 
           L1损失先将图归一化到[0,1], 判别器损失除以2, 特征余弦损失的输入得先归一化, 

#FBI WARNING:
1. Net2.py: 不再通过全连接层将vgg特征转化到(14,14,256), 而是先reshape(4,4,256)再通过deconv操作到(14,14,256);参数量从(4096*14*14*256)减小到
4. 之前的效果不好来自于对fc1层加了BN

# TO-DO
1. 看看GAN是怎么训练的, 加入GAN. done
2. 人脸加MASK
3. VGG特征用的是fc7, 而不是relu7. done
4. 只转正, 不加vgg loss. done
5. 试一试MSE. done
6. 将relu改成lrelu. done
7. L1损失是否要先归一化(255-1). done
8. 优化生成器, 学pix2pix用桥接结构
9. 试一下Wgan的GAN损失函数
10. dropout

# Setting1
1. 根据MPIE的setting1协议构建训练集(100*7*20)和测试集(149*6*19)

# Session01
1. session01中的前200个人(120k张)作为训练集, 后49个人(49*2*15*20)作为测试集

# 实验日志:
11.28
将fc7特征改为用relu7的特征来重建人脸

11.29: 
1. 只用VGG feature进行转正, 学习率设为0.001
2. 忘记把L2规则化加入到损失中了, 已加入
3. 忘记给fc1层加BN, 已加入
4. 重新用setting1的数据进行训练, loss比10:1, 测试还是用20光照+15姿态+2表情, lr=0.001 - 不对
5. 用setting1训练,特征用relu7, loss比1:1, fc1不加BN, LR=0.001, L2范数0.001, bs16, cpu
6. 用setting1训练,特征用relu7, loss比10:1, LR=0.001, L2范数0.00001, bs16, gpu - 不对
7. 用setting1训练,特征用relu7, loss比10:1, fc1不加BN, LR=0.001, L2范数0.001, bs16, gpu

11.30
1. 用session01训练,特征relu7,loss比10:1,fc1不加BN,lr=0.001,l2范数0.0001,bs16,gpu
2. 用session01训练,特征relu7,loss比10:1,fc1加BN,lr=0.001,l2范数0.0001,bs16,gpu
