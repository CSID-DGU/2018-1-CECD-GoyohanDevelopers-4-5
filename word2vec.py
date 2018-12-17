from gensim.models.word2vec import Word2Vec
import koreanHandler
import numpy as np


class CustomWord2Vec:
    def __init__(self):
        try:
            self.model = Word2Vec.load("./output/model")
            print("model load complete")
            print("model length : %d" % len(self.model.wv.vocab))
        except:
            self.model = Word2Vec(list("".join(koreanHandler.main("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"))), min_count=1, size=32)
            print("model load error")
        pass

    def train(self, voca):
        new_sentences = list("".join(koreanHandler.main(voca)))
        self.model.build_vocab(new_sentences, update=True)
        self.model.train(new_sentences, total_examples=1, epochs=1)
        model.save()

    def getvector(self, voca):
        devris_voca = list("".join(koreanHandler.main(voca)))
        # temp = np.zeros(shape=32, dtype=float)
        try:
            temp = np.array([np.array(self.model.wv.get_vector(devris_voca[0])).reshape(32, 1)])
        except IndexError:
            self.train(devris_voca[0])
            # temp = np.zeros(shape=32, dtype=float)
            temp = np.array([np.array(self.model.wv.get_vector(devris_voca[0])).reshape(32, 1)])

        for i in range(1, 32):  # fixed size
            try:
                # temp = np.concatenate((temp, self.model.wv.get_vector(devris_voca[i])), axis=0)
                temp = np.append(temp, np.array([np.array(self.model.wv.get_vector(devris_voca[i])).reshape(32, 1)]), axis=0)
            except:
                # temp = np.concatenate((temp, np.zeros(shape=32, dtype=float)), axis=0)
                temp = np.append(temp, np.array([np.array(np.zeros(shape=32, dtype=float)).reshape(32, 1)]), axis=0)

        return temp

    def getWordEmbedding(self, voca):
        devris_voca = list("".join(koreanHandler.main(voca)))
        temp = np.zeros(shape=32, dtype=float)
        for i in range(devris_voca.__len__()):
            temp = temp + self.model.wv.get_vector(devris_voca[i])
        temp = temp / devris_voca.__len__()
        return temp.reshape(32, 1)

    def save(self):
        self.model.save("./output/model")

    def fileTrain(self, filename):
        try:
            with open('./trainingData/origin/' + filename, 'r', encoding='utf-8') as infile:
                print("File Found")
                for line in infile:
                    self.train(line.replace("\n", ""))

            print("End Train File %s" % filename)
        except FileNotFoundError:
            print("File Not Found")


def getStartSymbolVector():
    temp = np.array([np.array([0.1] * 32).reshape(32, 1)])
    for i in range(1, 32):  # fixed size
        temp = np.append(temp, np.array([np.array([0.1] * 32).reshape(32, 1)]), axis=0)
    return temp


def getEndSymbolVector():
    temp = np.array([np.array([0.2] * 32).reshape(32, 1)])

    for i in range(1, 32):  # fixed size
        temp = np.append(temp, np.array([np.array([0.2] * 32).reshape(32, 1)]), axis=0)
    return temp


def getNullVector():
    temp = np.array([np.zeros(shape=32, dtype=float).reshape(32, 1)])
    for i in range(1, 32):  # fixed size
        temp = np.append(temp, np.array([np.array(np.zeros(shape=32, dtype=float)).reshape(32, 1)]), axis=0)
    return temp


def getNullWordEmbedding():
    temp = np.zeros(shape=32, dtype=float)
    return temp.reshape(32, 1)


if __name__ == "__main__":
    model = CustomWord2Vec()
    data = model.getWordEmbedding("Hello, world")
    print(data)