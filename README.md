#MODEL-实验日志:
default: 超参12(100, 1), 直接输入特征(relu7层), 第一层不用BN, BN分测试和训练, 初始lr=e-4(指数下降), L2规则化0.0001, 
         先训练一个epoch的texture loss再加入GAN loss, 1次D2次G, BatchSize=32, Adam(momenta=0.5), lrelu, 
         mirror, 判别器D从vgg-pool5开始构建, L1损失输入的图范围[0,1]
         
1. session01_1: 

1. setting1_1: 直接从pool5重建人脸, 
2. setting1_2: 从relu7重建人脸(不用全连接),
3. setting1_3: 从relu7重建人脸, 初始fc层7*7*256, batchSize=10,
3. setting1_4: 从relu7重建人脸, 初始fc层7*7*256, batchSize=10, D直接从原图接入, D的输入将侧脸和正脸concat(参照pix2pix)

#FBI WARNING:
1. Net2.py: 不再通过全连接层将vgg特征转化到(14,14,256), 而是先reshape(4,4,256)再通过deconv操作到(14,14,256);参数量从(4096*14*14*256)减小到
3. relu7, 余弦距离fc7, L2范数0.0001, 损失10:1, lr=0.001, setting1训练, fc1层不加BN
4. 之前的效果不好来自于对fc1层加了BN
5. 特征文件末尾是fc7的表示fc7层, 否则是reul7

# TO-DO
1. 看看GAN是怎么训练的, 加入GAN
2. 人脸加MASK
3. VGG特征用的是fc7, 而不是relu7. done
4. 只转正, 不加vgg loss. done
5. 试一试MSE
6. 将relu改成lrelu
7. L1损失是否要先归一化(255-1)

# Setting1
1. 根据MPIE的setting1协议构建训练集(100*7*20)和测试集(149*6*19)
2. 

# Session01
1. session01中的前200个人(120k张)作为训练集, 后49个人(49*2*15*20)作为测试集
2. 正面化损失和VGG特征损失比1:1
3. 初始learning rate:0.001

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
