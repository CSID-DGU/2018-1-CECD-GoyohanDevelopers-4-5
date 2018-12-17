# input data filiter
# 2018-09-27 make by Lee Yong Ho


def main(filename):
    vocadata = list()
    tagdata_list = list()
    count = 0

    # OutPut
    originalData = list()
    tagDataList = list()

    with open('./input/' + filename, 'r',encoding='utf-8') as infile:
        for line in infile:
            if line == "\n":
                pass
            elif line[0].isdigit():  # 원문 영역
                if tagdata_list.__len__() == 0:  # data가 존재하지 않을때
                    pass
                elif tagdata_list.__len__() == vocadata.__len__():  # 여기서 저장
                    for i in range(tagdata_list.__len__()):
                        tagDataList.append(vocadata[i] + "\t" + tagdata_list[i])
                    tagDataList.append("")  # 문장구분

                else:  # 다른 경우
                    if originalData.__len__() != 0:
                        originalData.pop()
                
                originalData.append(line[line.find(";") + 2:])

                # initialize for sentence.
                tagdata_list = list()
                originData = line[line.find(";") + 2:].split(" ")
                if '\n' in originData:
                    originData.remove('\n')
                vocadata = originData

            elif "</body" in line or "</text" in line or "</tei.2" in line:  # html 태그 제거
                pass

            else:  # Tagging 영역
                tmpvoca = line.replace("\n", "")
                vocas = tmpvoca.split(" ")

                tmp = list()
                for voca in vocas:
                    if '/' in voca:
                        if voca == vocas[vocas.__len__() - 1]:  # 마지막 부분에 대해서 괄호 처리
                            tmp.append(voca[:voca.find(')')])
                        else:
                            tmp.append(voca)
                    elif '+' in voca:
                        tmp.append(" " + voca + " ")
                    else:
                        pass  # Tagger 용으로 사용하는것이기 때문에, Parser의 경우, 코드 추가

                # Parser의 경우 이 밑에부터 모두 수정해야 한다.
                tagdata = "".join(tmp)
                if not tagdata == "":
                    tagdata_list.append(tagdata)
                    pass
                else:
                    pass

    with open('./trainingData/origin/' + "origin_"+ filename, 'w', encoding='utf-8') as infile:
        for i in range(originalData.__len__()):
            infile.write(originalData[i])

    with open('./trainingData/tag/' + "tag_"+ filename, 'w', encoding='utf-8') as infile:
        for i in range(tagDataList.__len__()):
            infile.write(tagDataList[i] + "\n")


if __name__ == "__main__":
    main("11.txt")