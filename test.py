import numpy as np
import tensorflow as tf
import train
import vectorize
import word2vec
import TagClassifier

epoch = 1
window_size = 5

if __name__ == "__main__":
    a, b, _ = vectorize.main("tag_20.txt")  # a : 문장 단위의 단어 벡터, b : 문장 단위의 형태소 벡터


    start, end = word2vec.getStartSymbolVector(), word2vec.getEndSymbolVector()

    nullTagvector = TagClassifier.getNullTagVector()
    nullVocavector = word2vec.getNullVector()

    model = train.cnnlstm_model()
    sess = tf.Session()
    sess.run(model.init)
    model.loadModel(sess)

    count = 0.0
    total = 0.0
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

                    import TagClassifier
                    # print(data[0][2], end= " ")
                    predict_tag = TagClassifier.getTag(data[0][np.unravel_index(np.argmax(Bf, axis=None), Bf.shape)[0]].tolist())
                    correct_tag = TagClassifier.getTag(tagVec[np.unravel_index(np.argmax(Bf, axis=None), Bf.shape)[0]])

                    if predict_tag == correct_tag:
                        count = count + 1.0
                    total = total + 1.0

    print(count, end=" ")
    print(total)
    print("Predict Proposition : %f" % (count / total))
    model.saveModel(sess)