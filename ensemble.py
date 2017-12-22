import pickle
import numpy as np
import math
from sklearn.metrics import classification_report

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''

        self.base_classifier = weak_classifier #传进来的base learner
        self.n_limit = n_weakers_limit  #base learner的最大个数
        self.learner = []  #各个base learner模型
        self.w_learner = np.ones(self.n_limit) #各个base learner的权重
        self.y_result = [] #最近产生的预测结果


    #返回预测结果
    def is_good_enough(self,y_val):
        #Optional

        target_names = ['face', 'nonface']
        labels = [1, -1]
        return classification_report(y_val, self.y_result, labels=labels, target_names=target_names)



    #训练AdaBoost
    def fit(self,X_train,y_train):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        n_sample = X_train.shape[0] #训练集数目
        w = np.ones(n_sample)/n_sample #初始化数据集权重

        epsilon = 0.0 #误差率
        for i in  range(self.n_limit):
            model = self.base_classifier.fit(X_train, y_train, sample_weight=w) #训练base learner
            y_predict = model.predict(X_train).reshape(y_train.shape[0],1) #base learner的预测结果
            for j in range(n_sample):#计算误差率
                if not y_predict[j] == y_train[j]:
                    epsilon += w[j]


            if epsilon < 0.5:
                self.learner.append(model)#base learner合格,保存
                self.w_learner[i] = 0.5 * np.log((1-epsilon)/epsilon) #记录学习器权重

                # 更新数据集权重
                w =w * np.exp(-self.w_learner[i] * y_train * y_predict).reshape(n_sample)
                w = w / w.sum()
        return


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''

        scores = np.zeros(X.shape[0])
        for i in range(self.w_learner.shape[0]):
            scores += self.w_learner[i] * self.learner[i].predict(X)
        return scores  #多个base learner共同产生的分数,大于0为正例,小于0为负例



    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        scores = self.predict_scores(X)
        self.y_result = np.sign(scores)
        return self.y_result  #预测结果



    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
