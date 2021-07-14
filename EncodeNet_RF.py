# coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Index import ROC_curve
from Index import c_matrics
import scikitplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import pickle

import xlrd
'''读取数据'''
def read_train_data(train_file_name):
    data = xlrd.open_workbook(train_file_name)
    table = data.sheet_by_index(0)

    all_data = []
    all_data_label = []
    for i in range(1, table.nrows):
        temp_data = []
        for j in range(table.ncols-1):
            temp_data.append(table.cell(i,j).value)
        all_data.append(temp_data)
        all_data_label.append([table.cell(i, table.ncols-1).value])

    all_data = np.array(all_data)
    all_data_label = np.array(all_data_label)

    feature_names = []
    for i in range(table.ncols-1):
        feature_names.append(table.cell(0,i).value)
    return all_data, all_data_label, feature_names

def read_test_data(test_file_name):
    data = xlrd.open_workbook(test_file_name)
    table = data.sheet_by_index(0)
    all_data = []
    for i in range(1, table.nrows):
        temp_data = []
        for j in range(table.ncols):
            temp_data.append(table.cell(i,j).value)
        all_data.append(temp_data)

    all_data = np.array(all_data)
    print("all_example:{}".format(all_data.shape[0]))
    return all_data

'''训练'''
def train_RF(best_parameter_ga):    
    n_estimators = int(best_parameter_ga[0])
    max_features = int(best_parameter_ga[1])
    min_samples_split = int(best_parameter_ga[2])
    min_samples_leaf = int(best_parameter_ga[3])
    RF = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf)
    RF = RandomForestClassifier()
    RF.fit(train_data, train_label) 
    
    #保存模型
    f = open('RF.pickle','wb')
    pickle.dump(RF,f)
    f.close()    

'''训练曲线'''
def train_curve(model):
    scikitplot.estimators.plot_learning_curve(model,train_data, train_label)#训练曲线    

'''测试''' 
def test_RF(test_data, test_label,model):
    
    Y_pred=[]
    Y_pre = model.predict(test_data)
    for i in Y_pre:
       Y_pre=int(i)
       Y_pred.append(Y_pre)
    Y_pred=np.array(Y_pred)

    Y_test=[]
    for i in test_label:
        Y_test.append(int(i[0]))
    Y_test=np.array(Y_test)

    
    mseoos=[]
    mseoos.append(mean_squared_error(Y_test,Y_pred))
    print('MSE：',mseoos[-1])
    
    Accuracy=accuracy_score(Y_test,Y_pred)
    print('准确率：',Accuracy)
    
    kappa = cohen_kappa_score(np.array(Y_test).reshape(-1,1), np.array(Y_pred).reshape(-1,1))
    print('Kappa系数：',kappa)
    
    c_matrics(Y_test,Y_pred)
    
    Y_pred_prob=model.predict_proba(test_data)[:,1]
    ROC_curve(test_label,Y_pred_prob)
    
    result = np.array([Y_test, Y_pred, Y_pred_prob]).T
    np.savetxt(r"结果\RF验证.txt", result, delimiter = "\t")
    

'''预测'''    
def predict_RF(model):
    p_RF=model.predict(predict_data)
    pre_p_RF=model.predict_proba(predict_data)[:,1]
    result = np.array([p_RF, pre_p_RF]).T
    np.savetxt(r"结果\RF预测.txt", result, delimiter = "\t")

def feature_importance(RF,all_data, all_data_label, feature_names):
    if len(all_data_label.shape)==2:
        all_data_label = np.squeeze(all_data_label, axis=1)
    RF.fit(all_data, all_data_label)
    print("Features sorted by their score:")
    sort_feature = sorted(zip(map(lambda x: round(x, 4), RF.feature_importances_), feature_names), reverse=True)
    print(sort_feature)

#主程序
if __name__ == '__main__':
    
    global train_data
    global train_label
    global test_data
    global test_label
    global varify_data
    global varify_label
    global predict_data
    
    train_data, train_label, feature_names = read_train_data(train_file_name = r"结果\train.xlsx") 
    test_data, test_label, feature_names = read_train_data(train_file_name = r"结果\test.xlsx") 
    varify_data, varify_label, feature_names = read_train_data(train_file_name = r"结果\varify.xlsx") 
    predict_data, predict_label, feature_names = read_train_data(train_file_name = r"结果\predict.xlsx") 
    
    
    '''训练,输入参数组合'''
    parameter=[200,2,1,2]

    train_RF(parameter)
    
    #下载模型
    f = open('RF.pickle','rb')
    RF= pickle.load(f)
    f.close()  
    
    print('RF参数：')
    print(RF) #打印RF参数
    print('训练曲线：')
    train_curve(RF)
    
    
    #测试准确率
    print('20%varify准确率：')
    test_RF(varify_data, varify_label,RF)
    
    print('test准确率：')
    test_RF(test_data, test_label,RF)


#    预测
    predict_RF(RF)
    print('预测完毕')
   
    feature_importance(RF,test_data, test_label, feature_names)
    
    
   
    
    
