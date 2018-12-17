import os
# 날짜 : 2018-08-07
# 작성자 : 이용호
# 내용 : input 폴더에 존재하는 dataset 파일 이름
# 입력 : x
# 출력 : input에 존재하는 파일이름 list return


def getinput():
    path_dir = "./input"
    file_list = os.listdir(path_dir)
    file_list.sort()
    print("input file %d exist." % file_list.__len__())
    return file_list


def getTestset():
    path_dir = "./testset"
    file_list = os.listdir(path_dir)
    file_list.sort()
    print("test file %d exist." % file_list.__len__())
    return file_list


if __name__ == "__main__":
    getinput()