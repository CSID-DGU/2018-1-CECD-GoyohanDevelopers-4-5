import tensorflow as tf
import math
import numpy as np
import vectorize

# error 무시 코드 : Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input data
window_size = 5
epoch = 1

class cnnlstm_model:  # cnn-lstm
    def __init__(self):
        self.inputdata = tf.placeholder(tf.float32, [None, 32, 32, 1], name="inputdata")
        self.tagvector = tf.placeholder(tf.float32, [None, 45], name="tagvector")
        self.pe = tf.placeholder(tf.float32, [None, 1], name="position_embedding")
        self.bf = tf.placeholder(tf.float32, [None, 1], name="binaryfeature")

        # weight
        w1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
        w2 = tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=0.01))
        w3 = tf.Variable(tf.random_normal([7, 7, 1, 64], stddev=0.01))
        w4 = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=0.01))

        # weight for Convolutional Neural Network output
        w_o = tf.Variable(tf.random_normal([64, 32], stddev=0.01))

        # character composition model - convolution layer (단어구성요소(자소)의 특징벡터 추출)
        l1a = tf.nn.relu(tf.nn.conv2d(self.inputdata, w1, strides=[1, 1, 1, 1], padding='SAME'))
        l2a = tf.nn.relu(tf.nn.conv2d(self.inputdata, w2, strides=[1, 1, 1, 1], padding='SAME'))
        l3a = tf.nn.relu(tf.nn.conv2d(self.inputdata, w3, strides=[1, 1, 1, 1], padding='SAME'))
        l4a = tf.nn.relu(tf.nn.conv2d(self.inputdata, w4, strides=[1, 1, 1, 1], padding='SAME'))

        # max_pooling for composition model
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l1 = tf.nn.dropout(l1, 0.8)
        l2 = tf.nn.dropout(l2, 0.8)
        l3 = tf.nn.dropout(l3, 0.8)
        l4 = tf.nn.dropout(l4, 0.8)

        # 2nd convolution layer : [?, 16, 16, 64] -> [?, 8, 8, 16]
        w11 = tf.Variable(tf.random_normal([3, 3, 64, 16], stddev=0.01))
        w12 = tf.Variable(tf.random_normal([5, 5, 64, 16], stddev=0.01))
        w13 = tf.Variable(tf.random_normal([7, 7, 64, 16], stddev=0.01))
        w14 = tf.Variable(tf.random_normal([9, 9, 64, 16], stddev=0.01))

        l1b = tf.nn.relu(tf.nn.conv2d(l1, w11, strides=[1, 1, 1, 1], padding='SAME'))
        l2b = tf.nn.relu(tf.nn.conv2d(l2, w12, strides=[1, 1, 1, 1], padding='SAME'))
        l3b = tf.nn.relu(tf.nn.conv2d(l3, w13, strides=[1, 1, 1, 1], padding='SAME'))
        l4b = tf.nn.relu(tf.nn.conv2d(l4, w14, strides=[1, 1, 1, 1], padding='SAME'))

        l1 = tf.nn.max_pool(l1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.nn.max_pool(l3b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l4 = tf.nn.max_pool(l4b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l1 = tf.nn.dropout(l1, 0.8)
        l2 = tf.nn.dropout(l2, 0.8)
        l3 = tf.nn.dropout(l3, 0.8)
        l4 = tf.nn.dropout(l4, 0.8)

        # 3nd convolution layer & reshaping :  [?, 8, 8, 16] -> [?, 25]
        w21 = tf.Variable(tf.random_normal([3, 3, 16, 4], stddev=0.01))
        w22 = tf.Variable(tf.random_normal([5, 5, 16, 4], stddev=0.01))
        w23 = tf.Variable(tf.random_normal([7, 7, 16, 4], stddev=0.01))
        w24 = tf.Variable(tf.random_normal([9, 9, 16, 4], stddev=0.01))

        l1c = tf.nn.relu(tf.nn.conv2d(l1, w21, strides=[1, 1, 1, 1], padding='SAME'))
        l2c = tf.nn.relu(tf.nn.conv2d(l2, w22, strides=[1, 1, 1, 1], padding='SAME'))
        l3c = tf.nn.relu(tf.nn.conv2d(l3, w23, strides=[1, 1, 1, 1], padding='SAME'))
        l4c = tf.nn.relu(tf.nn.conv2d(l4, w24, strides=[1, 1, 1, 1], padding='SAME'))

        l1 = tf.nn.max_pool(l1c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.max_pool(l2c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.nn.max_pool(l3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l4 = tf.nn.max_pool(l4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        l1 = tf.nn.dropout(l1, 0.8)
        l2 = tf.nn.dropout(l2, 0.8)
        l3 = tf.nn.dropout(l3, 0.8)
        l4 = tf.nn.dropout(l4, 0.8)

        l1 = tf.reshape(l1, [-1, w_o.get_shape().as_list()[0]])
        l2 = tf.reshape(l2, [-1, w_o.get_shape().as_list()[0]])
        l3 = tf.reshape(l3, [-1, w_o.get_shape().as_list()[0]])
        l4 = tf.reshape(l4, [-1, w_o.get_shape().as_list()[0]])

        # matmul - each vector dimensions : [?, -1 , 64] -> [?, 25]
        pyx1 = tf.matmul(l1, w_o)  # 25 dimensions.
        pyx2 = tf.matmul(l2, w_o)  # 25 dimensions.
        pyx3 = tf.matmul(l3, w_o)  # 25 dimensions.
        pyx4 = tf.matmul(l4, w_o)  # 25 dimensions.

        # concatenate word embedding data. : [5, 179]
        m = []
        m.append(pyx1)
        m.append(pyx2)
        m.append(pyx3)
        m.append(pyx4)
        m.append(self.pe)
        m.append(self.bf)

        # concatenate with CCR, word-embedding
        c = tf.concat(m, 1, name="CharCompositionResult")
        c = tf.expand_dims(c, axis=2)


        # simple lstm -> After completion, Change Multi_cell-lstm.

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=45, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, c, dtype=tf.float32)
        self.model = tf.contrib.layers.fully_connected(outputs[:, -1], 45, activation_fn=None)
        # self.model = tf.layers.dense(outputs, units=45, activation=None)
        print("output : ", end="")
        print(self.model)
        # loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
        # cost = tf.reduce_sum(loss) / batch_size
        # train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
        # self.loss = tf.reduce_sum(tf.square(self.model - self.tagvector))
        # NLL_loss func.
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.tagvector))
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.tagvector, 1))
        self.acuu = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.train = tf.train.RMSPropOptimizer(0.00002, 0.0001).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def getanswer(self, word, pe, bf):
        pass


    # Tag_List = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV',
    #             'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG',
    #             'MAJ', 'IC', 'JKS', 'JKC', 'JKG', 'JKO',
    #             'JKB', 'JKV', 'JKQ', 'JC', 'JX', 'EP',
    #             'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN',
    #             'XSV', 'XSA', 'XR', 'SF', 'SE', 'SS',
    #             'SP', 'SO', 'SW', 'SH', 'SL', 'SN',
    #             'NF', 'NV', 'NA']

    def saveModel(self, sess):
        self.saver.save(sess,'./train_model.ckpt')
        print('Trained Model Saved.')

    def loadModel(self, sess):
        try:
            self.saver.restore(sess, './train_model.ckpt')
            print('Model load complete.')
        except Exception:
            print('Model load fail.')


    def train(self):
        pass


if __name__ == "__main__":
    a, b, c = vectorize.main("tag_20.txt")  # a : 문장 단위의 단어 벡터, b : 문장 단위의 형태소 벡터

    import word2vec
    import TagClassifier

    start, end = word2vec.getStartSymbolVector(), word2vec.getEndSymbolVector()

    nullTagvector = TagClassifier.getNullTagVector()
    nullVocavector = word2vec.getNullVector()
    nullVocabvector = word2vec.getNullWordEmbedding()

    model = cnnlstm_model()
    sess = tf.Session()
    sess.run(model.init)
    model.loadModel(sess)

    count = 0
    for cnt in range(epoch):
        for i in range(a.__len__()):
            # setData
            voca_vector = list()
            tag_vector = list()

            voca_vector.append(start)
            tag_vector.append(nullTagvector.tolist())

            for j in range(a[i].__len__()):
                voca_vector.append(a[i][j])
                tag_vector.append(b[i][j])

            voca_vector.append(end)
            tag_vector.append(nullTagvector.tolist())

            # train model
            # select training data
            for j in range(voca_vector.__len__()):
                vocaVec = list()
                tagVec = list()

                if j + window_size <= voca_vector.__len__():
                    for k in range(j, j + window_size, 1):
                        vocaVec.append(voca_vector[k])
                        tagVec.append(tag_vector[k])

                else:
                    for k in range(j, voca_vector.__len__(), 1):
                        vocaVec.append(voca_vector[k])
                        tagVec.append(tag_vector[k])

                for x in range(window_size - voca_vector.__len__() + j):
                    vocaVec.append(nullVocavector.tolist())
                    tagVec.append(nullTagvector.tolist())


                for w in range(window_size):
                    Pe = np.zeros([window_size, 1], dtype=float)
                    Bf = np.zeros([window_size, 1], dtype=float)
                    for v in range(window_size):
                        if j + v >= voca_vector.__len__():
                            Pe[v] = 0.0
                        else:
                            Pe[v] = j + v

                    for k in range(window_size):
                        if w == k:
                            Bf[k] = 1.0
                        else:
                            Bf[k] = 0.0
                    # inputdata 완료
                    _, cost, acc = sess.run([model.train, model.loss, model.acuu],
                                            feed_dict={model.inputdata: np.array(vocaVec),
                                                       model.tagvector: np.array(tagVec),
                                                       model.pe: np.array([Pe]).reshape(-1, 1),
                                                       model.bf: np.array([Bf]).reshape(-1, 1)})
                    count = count + 1
                    if count % 100 == 0:
                        print("[step: {}] loss: {}".format(count, cost))
                        print("accu : {}".format(acc))
                    elif count % 100000 == 0:
                        model.saveModel(sess)
        print("[count: {}] loss: {}".format(cnt + 1, cost))
    model.saveModel(sess)

    # Testset Validation
    # arr = sess.run(model.model, {model.inputdata: word, model.pe: np.array([pe]).reshape(-1, 1),
    #                              model.bf: np.array([bf]).reshape(-1, 1)})
    # print('Prediction :', TagClassifier.getTag(arr[0].tolist()))
    # print(arr[0])