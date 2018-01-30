#MODEL:
default: 
1. 超参:    l1:fea:GAN:sym=0:500:1:0, L2规则化0.0001, BatchSize=5, Adam(0.5,0.9), kernel=4, strides=2, 
           左右对称损失输入的图范围[0,1], 权重初始化stddev=0.02, 初始lr=2*e-4(不变),
2. 生成器G: (G的enc一般用lrelu,dec用RELU)encoder用vgg2, decoder为全卷积网络, pixel_norm+relu, 第一层和最后一层不加PN
           三个大小的生成器协同训练, 原始特征join到每个生成器的开始
3. 判别器D: patchGAN, 直接输入G的结果, 并减均值归一化(/127.5-1), 激活函数用lrelu(k=0.2), 三个尺度的D对应三个G, 第一层和最后一层不加ins norm
4. 其它:    mirror, 随机crop成224, critic=5, 特征余弦损失的输入得先归一化, 训练图片多线程读取, ins norm代替BN, 生成器输入样本和正脸样本不匹配,
           

# 实验日志:
1. setting1_8: WGAN, critic=5, 
2. setting1_1: WGAN, critic=5, 加入正脸GT绝对值损失, 
3. (work)setting1_2: WGAN-GP(w_gp=10), critic=1, Layer-Norm, RMS 
4. (work)setting1_3: LSGAN, critic=1, 加入正脸GT绝对值损失, beat1=0.5, 
5. (work)setting1_4: WGAN-GP2(w_gp=10), critic=1, adam(0,.9), 人脸五块区域各设一个判别器(全部/眼/鼻/嘴/脸), 判别器LayerNorm
6. setting1_5: WGAN-GP2(w_gp=10), critic=1, adam(0,.9), 人脸五块区域各设一个判别器(全部/眼/鼻/嘴/脸), 判别器LayerNorm, 侧脸使用casia

#FBI WARNING:
1. Net2.py: 不再通过全连接层将vgg特征转化到(14,14,256), 而是先reshape(4,4,256)再通过deconv操作到(14,14,256);参数量从(4096*14*14*256)减小到
2. 之前的效果不好来自于对fc1层加了BN

# TO-DO
1. 看看GAN是怎么训练的, 加入GAN. done
2. 人脸加MASK, 统计正脸图片的热度图, 方差大的地方是背景,小的地方是人脸特征
3. VGG特征用的是relu7, 而不是fc7. done
4. 只转正, 不加vgg loss. done
5. 试一试MSE. done
6. 将relu改成lrelu. done
7. L1损失是否要先归一化(255-1). done
8. 优化生成器, 学pix2pix用桥接结构. doing
9. 试一下Wgan的GAN损失函数
10. dropout.(应该不用dropout比较好, 这会引入随机性) doing. 
11. 正脸监督图片用20个光照的平均脸/或者用当前光照下的正脸图片
12. 预训练之后减小L1损失的比重
13. 将输入侧脸mirror前后的特征相加作为G_enc的输出(这样的话要将预处理的mirror取消)
14. 第一次G/D一起训练
15. 调整结构, 前端卷积网络下采样/res模块/后端卷积网络上采样
16. 加入noize向量

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




