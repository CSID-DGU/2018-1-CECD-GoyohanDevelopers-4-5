# training set을 위한 vectorize
import numpy as np
import tensorflow as tf
import word2vec
import TagClassifier

# 2018-10-17 15:17 최종수정

def departdata(filename):

    x_data = list()
    y_data = list()

    with open('./trainingData/tag/' + filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line == "\n":
                x_data.append("\n")
                y_data.append("\n")
                pass
            else:
                data = "".join(line.replace("\n","").split("\t")[-1:]).split(" + ")
                for text in data:
                    a = text.split("/")
                    if a.__len__() > 2:  # //TAG
                        x = "/"
                        y = a[2]
                        x_data.append(x)
                        y_data.append(y)
                    elif a.__len__() == 2:  # VOCA/TAG
                        x = a[0]
                        y = a[1]
                        x_data.append(x)
                        y_data.append(y)
                    else:
                        pass

        # end point
        x_data.append("\n")
        y_data.append("\n")
    return x_data, y_data


def vectorize(x_data, y_data):
    customword2vec = word2vec.CustomWord2Vec()
    x_vectorset = list()
    y_vectorset = list()
    z_vectorset = list()

    x_vector = list()
    y_vector = list()
    z_vector = list()

    for x in x_data:
        if x_vector.__len__() > 0 and x == "\n":
            x_vectorset.append(x_vector)
            x_vector = list()
        elif x == "\n":
            pass
        else:
            x_vector.append(customword2vec.getvector(x).tolist())
            # x_vector = np.append(x_vector, np.array([customword2vec.getvector(x)]), axis=0)

    for y in y_data:
        if y_vector.__len__() > 0 and y == "\n":
            y_vectorset.append(y_vector)
            y_vector = list()
        elif y == "\n":
            pass
        else:
            y_vector.append(TagClassifier.getVector(y))
            # y_vector = np.append(y_vector, np.array([TagClassifier.getVector(y)]), axis=0)

    for x in x_data :
        if z_vector.__len__() > 0 and x == "\n":
            z_vectorset.append(z_vector)
            z_vector = list()
        elif x == "\n":
            pass
        else:
            z_vector.append(customword2vec.getWordEmbedding(x).tolist())

    return x_vectorset, y_vectorset, z_vectorset


def main(filename):
    x, y = departdata(filename)
    x_data, y_data, z_data = vectorize(x, y)
    return x_data, y_data, z_data


# testProgram용 : List형태
def xdata_vector(x_data):
    customword2vec = word2vec.CustomWord2Vec()
    x_vectorset = list()
    x_vector = list()

    for x in x_data:

        if x_vector.__len__() > 0 and x == "\n":
            x_vectorset.append(x_vector)
            x_vector = list()
        elif x == "\n":
            pass
        else:
            x_vector.append(customword2vec.getvector(x).tolist())
            # x_vector = np.append(x_vector, np.array([customword2vec.getvector(x)]), axis=0)
    if x_vector.__len__() > 0:
        x_vectorset.append(x_vector)
        x_vector = list()
    return x_vectorset

if __name__ == "__main__":
    main("tag_20.txt")

