"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
"""

from __future__ import division

import random
import time

from tensorflow.python.ops.math_ops import reduce_sum
from ops import *
from utils import *
import re

from LSTM_G import *
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class RuleModel(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=500, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default', sample_dir=None,
                 checkpoint_dir=None, alpha=1.0, beta=1.0, delta_mean=0.0, delta_var=0.0
                 , label_col=-1, attrib_num=0
                 , is_shadow_gan=False
                 , test_id=''
                 ):

        self.test_id = test_id

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.feature_size = 0
        self.attrib_num = 1

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.r_bn1 = batch_norm(name='r_bn1')
        self.r_bn2 = batch_norm(name='r_bn2')
        self.r_bn3 = batch_norm(name='r_bn3')

        self.label_col = label_col
        self.attrib_num = attrib_num

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        # mm if self.dataset_name in ["LACity", "Health", "Adult", "Ticket"]:
        """
        data_X: 填充到 2000 * 49 (7*7) 维的 data
        data_y: one-hot 后的 Labels. 2000 * 2
        data_y_normal: 原始的 labels. 2000 * 1
        """
        print("【 Load_dataset...】")
        self.data_X, self.data_y, self.data_y_normal = self.load_dataset(is_shadow_gan)
        self.r_dim = 1

        self.grayscale = (self.r_dim == 1)
        print("r_dim 1= " + str(self.r_dim))

        # 得到 train data 的元信息 以及 scale
        self.min_max_scaler, self.origin_data = get_min_max_scalar(model=self)

        # todo self.test
        self.test = False

        print("【 Building GAN model...】")
        self.build_model()

    def build_model(self):
        # one hot
        self.y = tf.compat.v1.placeholder(
            tf.float32, [self.batch_size, self.y_dim], name='y')

        # normal 
        self.y_normal = tf.compat.v1.placeholder(
            tf.int16, [self.batch_size, 1], name='y_normal')

        # self.input_height, self.input_width: 卷积核心的长宽，self.r_dim: 输出通道
        data_dims = [self.input_height, self.input_width, self.r_dim]

        # 500 * 7 * 7 * 1
        self.inputs = tf.compat.v1.placeholder(
            tf.float32, [self.batch_size] + data_dims, name='inputs')


        # 使用 rule model
        self.R, self.R_logits, self.R_features = self.rule(self.inputs)


        self.r_sum = histogram_summary("c", self.R)

        # ------------------- 算 loss 👇 -------------------------
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        # 原始的 labels 2000 * 1
        # rule 和 原始 y 的 cross entropy
        y_normal = tf.to_float(self.y_normal)
        self.r_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.R_logits, y_normal))

        t_vars = tf.compat.v1.trainable_variables()
        self.r_loss_sum = scalar_summary("rule | Sum Loss", self.r_loss)
        self.r_vars = [var for var in t_vars if 'r_' in var.name]

        self.saver = tf.compat.v1.train.Saver()



    def train(self, config, experiment=None):
        print("Start Training...\n")

        r_optim = tf.compat.v1.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.r_loss, var_list=self.r_vars)

        try:
            tf.compat.v1.global_variables_initializer().run()
        except:
            tf.compat.v1.global_variables_initializer().run()
        # 训练时的各种信息

        # Classifier
        if self.y_dim:
            self.r_sum = merge_summary([self.r_sum, self.r_loss_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        # 加载最新的 checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 开始训练
        for epoch in xrange(config.epoch):

            # 打乱数据
            batch_idxs = min(len(self.data_X),
                             config.train_size) // config.batch_size  # train_size= np.inf

            seed = np.random.randint(100000000)
            np.random.seed(seed)
            np.random.shuffle(self.data_X)

            np.random.seed(seed)
            np.random.shuffle(self.data_y)

            np.random.seed(seed)
            np.random.shuffle(self.data_y_normal)

            for idx in xrange(0, batch_idxs):

                # 从 data_X 中取出一个 batch 的数据
                batch = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = self.data_y[
                               idx * config.batch_size: (idx + 1) * config.batch_size]
                batch_labels_normal = self.data_y_normal[
                                      idx * config.batch_size: (idx + 1) * config.batch_size]

                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str, r_output = self.sess.run([r_optim, self.r_sum, self.R],
                                                         feed_dict={
                                                             self.inputs: batch_images,
                                                             self.y: batch_labels,
                                                             self.y_normal: batch_labels_normal
                                                         })
                self.writer.add_summary(summary_str, counter)

                # todo: 测试 rule Model 准确率
                if self.test:
                    get_rule_model_score(r_output)
                    return

                # 计算 loss
                errR = self.r_loss.eval({
                    self.inputs: batch_images,
                    self.y: batch_labels,
                    self.y_normal: batch_labels_normal
                })

                counter += 1
                # experiment.log_metric("d_loss", errD_fake + errD_real, step=idx)
                # experiment.log_metric("g_loss", errG, step=idx
                if self.y_dim:
                    # experiment.log_metric("r_loss", errC, step=idx)
                    print(
                        "Dataset: [%s] -> [%s] -> Epoch: [%2d] [%4d/%4d] time: %4.4f, r_loss: %8.6f" % (
                            config.dataset, config.test_id, epoch, idx + 1, batch_idxs,
                            time.time() - start_time, errR))
                    # if time.time() - start_time > 3.5 * 60 * 60:
                    #     self.save(checkpoint_dir=self.checkpoint_dir, step=2)
                    #     print()
                    #     return
                if np.mod(counter, 50) == 2:
                    self.save(config.checkpoint_dir, counter)
                    get_rule_model_score(r_output, batch_labels_normal)

    def rule(self, image, y=None, reuse=False):

        with tf.compat.v1.variable_scope("rule") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='r_h0_conv'))
            h1 = lrelu(self.r_bn1(
                conv2d(h0, self.df_dim * 2, name='r_h1_conv')))
            h2 = lrelu(self.r_bn2(
                conv2d(h1, self.df_dim * 4, name='r_h2_conv')))
            h3 = lrelu(self.r_bn3(
                conv2d(h2, self.df_dim * 8, name='r_h3_conv')))

            h3_f = tf.reshape(h3, [self.batch_size, -1])
            # h4 = linear(tf.reshape(
            #     h3, [self.batch_size, -1]), 1, 'r_h3_lin')

            h4 = linear(h3_f, 1, 'r_h3_lin')

            return tf.nn.sigmoid(h4), h4, h3_f

    def load_dataset(self, load_fake_data=False):

        return self.load_tabular_data(self.dataset_name, self.input_height, self.y_dim, self.test_id, load_fake_data)

    def load_tabular_data(self, dataset_name, dim, classes=2, test_id='', load_fake_data=False):
        # todo: 替换训练集
        # self.train_data_path = f"./data/{dataset_name}/{dataset_name}"
        # self.train_data_path = f'data/{dataset_name}/{dataset_name}'
        # self.train_data_path = "./data/Adult/Adult_rm_0>mean"
        # self.train_label_path = "./data/Adult/Adult_rm_0>mean_labels"
        # self.train_data_path = "./data/Ticket/Ticket_rulemodel"
        # self.train_label_path = "./data/Ticket/Ticket_rulemodel_labels"
        self.train_data_path = "./data/Adult/Adult_rulemodel"
        self.train_label_path = "./data/Adult/Adult_rulemodel_labels"
        if os.path.exists(self.train_data_path + ".csv"):

            X = pd.read_csv(self.train_data_path + ".csv", sep=',')
            print("Loading X CSV input file : %s" % (self.train_data_path + ".csv"))

            # load 一共多少 attribute
            self.attrib_num = X.shape[1]

            # default 2 or none
            if self.y_dim:
                y = np.genfromtxt(open(self.train_label_path + ".csv", 'r'), delimiter=',')

                print("Loading Y CSV input file : %s" % (self.train_label_path + ".csv"))

                self.zero_one_ratio = 1.0 - (np.sum(y) / len(y))

        elif os.path.exists(self.train_data_path + ".pickle"):
            with open(self.train_data_path + '.pickle', 'r', encoding='UTF-8') as handle:
                X = pickle.load(handle)

            with open(self.train_label_path + '.pickle', 'r', encoding='UTF-8') as handle:
                y = pickle.load(handle)

            print("Loading pickle file ....")
        else:
            print("2. Error Loading Dataset: can't find {}".format(self.train_data_path))
            exit(1)
        

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        # Normalizing Initial Data
        X = pd.DataFrame(min_max_scaler.fit_transform(X))
        # X is [rows * config.attrib_num] 2000 * 14

        ''' 
        1.  X 的 shape 为 20000 * 14; 14 个 feature
            [-0.39726027  0.71428571 -0.88517043 -1.          0.6        -0.33333333,  0.28571429  0.2        -1.          1.         -0.95651957 -1., -0.20408163 -0.94736842]
            [-0.09589041 -0.42857143 -0.87373955 -1.          0.6        -1., -0.28571429 -0.2        -1.          1.         -1.         -1., -0.75510204 -0.94736842]
            ...
        2.  使用的卷积核为 7*7, 所以将数据从 14 纬填充到 49 维
            1.  在每一列的最后填充 49-14 = 35 个 0
                [-0.39726027  0.71428571 -0.88517043 -1.          0.6        -0.33333333,  0.28571429  0.2        -1.          1.         -0.95651957 -1., -0.20408163 -0.94736842  0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.        ]
                [-0.09589041 -0.42857143 -0.87373955 -1.          0.6        -1., -0.28571429 -0.2        -1.          1.         -1.         -1., -0.75510204 -0.94736842  0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.          0.          0.          0.          0.          0.,  0.        ]
                ...
            2.  用原先 14 维的数据替代 0 -> padded_ar
                [-0.39726027  0.71428571 -0.88517043 -1.          0.6        -0.33333333,  0.28571429  0.2        -1.          1.         -0.95651957 -1., -0.20408163 -0.94736842 -0.39726027  0.71428571 -0.88517043 -1.,  0.6        -0.33333333  0.28571429  0.2        -1.          1., -0.95651957 -1.         -0.20408163 -0.94736842 -0.39726027  0.71428571, -0.88517043 -1.          0.6        -0.33333333  0.28571429  0.2, -1.          1.         -0.95651957 -1.         -0.20408163 -0.94736842,  0.          0.          0.          0.          0.          0.,  0.        ]
                [-0.09589041 -0.42857143 -0.87373955 -1.          0.6        -1., -0.28571429 -0.2        -1.          1.         -1.         -1., -0.75510204 -0.94736842 -0.09589041 -0.42857143 -0.87373955 -1.,  0.6        -1.         -0.28571429 -0.2        -1.          1., -1.         -1.         -0.75510204 -0.94736842 -0.09589041 -0.42857143, -0.87373955 -1.          0.6        -1.         -0.28571429 -0.2, -1.          1.         -1.         -1.         -0.75510204 -0.94736842,  0.          0.          0.          0.          0.          0.,  0.        ]
                ...
        '''
        # from utils
        padded_ar = padding_duplicating(X, dim * dim)

        X = reshape(padded_ar, dim)

        print("Final Real Data shape = " + str(X.shape))  # 2000 * 7 * 7
        print("Total attribute: ",self.attrib_num)

        # 如果没有 y 就 只output 处理过的x，y为None
        if self.y_dim:
            y = y.reshape(y.shape[0], -1).astype(np.int16)
            y_onehot = np.zeros((len(y), classes), dtype=np.float)
            for i, lbl in enumerate(y):
                y_onehot[i, y[i]] = 1.0
            return X, y_onehot, y

        return X, None, None

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "rule_model"
        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                if f.startswith(f'{self.test_id}'):
                    file_idx = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_idx)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {f} is not an integer. Skipping')
            checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num + 1}'
            print(checkpoint_dir)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print(" [Saving checkpoints in " + checkpoint_dir + " ...")
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints from " + checkpoint_dir + " ...")

        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                print(f)
                if f.startswith(f'{self.model_dir}') and f.replace(self.model_dir, '') != '':
                    print(f)
                    file_name = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_name)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {file_name} is not an integer. Skipping')
            if highest_num == 0:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}'
            else:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num}'
        print(f'checkpoint dir: {checkpoint_dir}')
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("ckpt:", ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))

            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(os.path.join(
                checkpoint_dir, ckpt_name)))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
