from PIL import Image
import numpy as np
from feature import *
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
import ensemble

neg_num = 0
pos_num = 0

#这个函数用来将图片转换成特征,结果生成一个.npy文件,数据格式为(n_sample,n_feature).
def transeToNPD(pos_dir, neg_dir):
    global neg_num, pos_num #正例与反例图片的个数
    NPD_feature = np.array((1,165600)) #用来保存图片特征的矩阵

    flag = False   #判断是否是处理的第一个图片
    for root, dirs, files in os.walk(pos_dir, topdown=False):  #遍历目录
        for file in files:  #遍历文件
            im = Image.open(pos_dir + file).convert('L')  #转为灰度图
            im = im.resize((24, 24)) #转为大小为24 * 24 的大小
            im_array = np.array(im)  #转为数组
            img_feature = NPDFeature(im_array).extract()   #提取特征
            if not flag:
                NPD_feature = img_feature.reshape(1,165600) #赋值给特征矩阵
                flag = True
            else:
                #添加特征给特征矩阵
                NPD_feature = np.row_stack((NPD_feature, img_feature.reshape(1,165600)))




    for root, dirs, files in os.walk(neg_dir, topdown=False): #遍历目录
        for file in files:  #转为灰度图
            im = Image.open(neg_dir + file).convert('L')
            im = im.resize((24, 24))  #转为大小为24 * 24 的大小
            im_array = np.array(im) #转为数组
            img_feature = NPDFeature(im_array).extract()  #提取特征
            if not flag:
                NPD_feature = np.array(img_feature)  #赋值给特征矩阵
                flag = True
            else:
                # 添加特征给特征矩阵
                NPD_feature = np.row_stack((NPD_feature, img_feature))

    np.save('a.npy', NPD_feature) #特征矩阵保存在 a.npy 文件中
    return

def loadDataSet():
    X = np.load('a.npy')  #从特征矩阵文件中导入特征矩阵
    y = [1 for t in range(pos_num)]  #生成特征矩阵对应的label
    y.extend([-1 for t in range(neg_num)]) #生成特征矩阵对应的label
    y = np.array(y).reshape(len(y),1) #将label转成(n_sample,1)形式
    return X, y





if __name__ == "__main__":
    # write your code here

    neg_num = 500
    pos_num = 500
    pwd =  os.getcwd() #获取当前工作目录
    pos_dir =pwd + 'datasets/original/face/' #正例目录
    neg_dir =pwd + 'datasets/original/nonface/' #反例目录

    #从文件导入特征矩阵,如果文件还没生成(特征未提取),则提取特征并保存在文件中
    if not os.path.exists('a.npy'):
        transeToNPD(pos_dir, neg_dir)
    else:
        X, y = loadDataSet()

    #将特征集和标记集切分,33.3%划分为测试集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

    #声明决策树
    DTC = tree.DecisionTreeClassifier(max_depth=6)

    #声明Adaboost类,base_learner的最大个数为10
    myAdaBoost = ensemble.AdaBoostClassifier(DTC, 10)

    #训练AdaBoost模型
    myAdaBoost.fit(X_train, y_train)

    #保存AdaBoost模型
    myAdaBoost.save(myAdaBoost, 'train_model')


    #预测测试集,产生的标记集保存在AdaBoost类中.
    myAdaBoost.predict(X_val)

    validation = myAdaBoost.is_good_enough(y_val)
    print(validation)
    fh = open('report.txt', 'w')
    fh.write(validation)
    fh.close()



    pass

