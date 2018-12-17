# Date : 2018-09-18
# FileName : TagClassifier.py
# made by Lee YongHo
#  input : Tag or Vector
# output : Vector or Tag
# method : getTag(vector), getVector(Tag)

import numpy as np

Tag_List = ['NNG', 'NNP', 'NNB', 'NR', 'NP',
            'VV', 'VA', 'VX', 'VCP', 'VCN',
            'MM', 'MAG', 'MAJ', 'IC',
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JC', 'JX',
            'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR',
            'SF', 'SE', 'SS', 'SP', 'SO', 'SW', 'SH', 'SL', 'SN', 'NF', 'NV', 'NA']


def getNullTagVector():
    return np.zeros(shape=Tag_List.__len__(), dtype=float)


def getTag(vector):
    index = vector.index(max(vector))
    return Tag_List[index]

def getVector(Tag):     # One-hot encoding 방식
    temp = np.zeros(shape=Tag_List.__len__(), dtype=float)
    for index, each in enumerate(Tag_List):
        if each == Tag:
            temp[index] = 1.0
    return temp.tolist()


if __name__ == "__main__":
    print("NNP", end=" ")
    print(getVector("NNP"))