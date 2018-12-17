import numpy as np
import tensorflow as tf
import train
import vectorize
import word2vec
import TagClassifier
import time

epoch = 1
window_size = 5


def main(x_data):
    Tag = list()
    print(x_data)
    a = vectorize.xdata_vector(x_data)  # 각 단어 벡터
    start, end = word2vec.getStartSymbolVector(), word2vec.getEndSymbolVector()

    nullVocavector = word2vec.getNullVector()

    model = train.cnnlstm_model()
    sess = tf.Session()
    sess.run(model.init)
    model.loadModel(sess)

    for cnt in range(epoch):
        for i in range(a.__len__()):
            # setData
            voca_vector = list()


            voca_vector.append(start)


            for j in range(a[i].__len__()):
                voca_vector.append(a[i][j])


            voca_vector.append(end)


            # train model
            # select training data
            for j in range(voca_vector.__len__()):
                vocaVec = list()

                if j + window_size <= voca_vector.__len__():
                    for k in range(j, j + window_size, 1):
                        vocaVec.append(voca_vector[k])
                else:
                    for k in range(j, voca_vector.__len__(), 1):
                        vocaVec.append(voca_vector[k])

                for x in range(window_size - voca_vector.__len__() + j):
                    vocaVec.append(nullVocavector.tolist())

                for w in range(window_size):
                    Pe = np.zeros([5, 1], dtype=float)
                    Bf = np.zeros([5, 1], dtype=float)
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
                    data = sess.run([model.model],
                                    feed_dict={model.inputdata: np.array(vocaVec),
                                               model.pe: np.array([Pe]).reshape(-1, 1),
                                               model.bf: np.array([Bf]).reshape(-1, 1)})


                    # print(data[0][2], end= " ")
                    predict_tag = TagClassifier.getTag(data[0][np.unravel_index(np.argmax(Bf, axis=None), Bf.shape)[0]].tolist())
                    Tag.append(predict_tag)

    return Tag

if __name__ == "__main__":
    data = list()
    data = input("문장의 형태소들을 입력해주세요.\n").split()
    tag = main(x_data=data)
    for i in range(data.__len__()):
         print(data[i], end="/")
         print(tag[(i+1) * window_size], end=" ")