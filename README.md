#FBI WARNING:
1. gpu版本需要修改

# TO-DO
1. 看看GAN是怎么训练的
2. 人脸加MASK
3. VGG特征用的是fc7, 而不是relu7

# Setting1
1. 根据MPIE的setting1协议构建训练集(100*7*20)和测试集(149*6*19)
2. 

# Session01
1. session01中的前200个人(120k张)作为训练集, 后49个人(49*2*15*20)作为测试集
2. 正面化损失和VGG特征损失比1:1
3. 初始learning rate:0.01
