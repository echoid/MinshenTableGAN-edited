启动虚拟环境 conda activate TableGAN
cd Table-GAN Minshen/tableGAN


1. 操作步骤
    在 Researh/tableGAN 目录下运行 pip3 install -r requirements.txt。
    在 Researh/tableGAN 目录下运行 python3 main.py --train --dataset=Adult --epoch=5000 --test_id=OI_11_00。 
    
    1. 开始训练 Rule Model
        每训练 50 轮 会在 Research/tableGAN/samples/Adult/Adult_rm_pred.csv 路径下输出 Rule Model 的输出。
        训练结束后，模型会被保存在 Table-GAN/tableGAN/checkpoint/Adult/OI_11_00/tableGan 下。
        将模型移动到 Table-GAN/tableGAN/checkpoint/Adult/ruleModel 后，即可在训练 tableGAN 时使用了。
    2. 开始训练 tableGAN
        注释 main.py 159 行的 return。
        在 Researh/tableGAN 目录下运行 python3 main.py --train --dataset=Adult --epoch=5000 --test_id=OI_11_00。
        训练完成后，模型会被保存在 able-GAN/tableGAN/checkpoint/Adult/OI_11_00/tableGan 下。
    3. sample
        在 Researh/tableGAN 目录下运行 python3 main.py --dataset=Adult --epoch=5000 --test_id=OI_11_00。
        sample 的结果会被保存在 Table-GAN/tableGAN/samples/Adult 下。


2. 关键代码内容

   1. 将数据转化为 CNN 的输入

        922 load_dataset()

   2. Table GAN

        Discriminator: 684 Generator: 804 loss func: 311

   3. Rule Model

        model: 1272 loss func: 1163