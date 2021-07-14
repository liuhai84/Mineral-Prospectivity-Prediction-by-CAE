# coding=utf-8
import numpy as np
from Index import ROC_curve
from Index import c_matrics
import scikitplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import pickle
from sklearn.svm import SVC
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
'''
参考网址：
https://blog.csdn.net/c1z2w3456789/article/details/105247565
http://wenda.chinahadoop.cn/question/4787
https://www.cnblogs.com/ceason/articles/12288603.html
'''
def train_SVM():    
    print('----------支持向量机----------')
    svc= SVC(kernel='poly', degree=3,coef0=0,probability=True).fit(train_data, train_label)  # 核函数为多项式函数   

    #保存模型
    f = open('SVM.pickle','wb')
    pickle.dump(svc,f)
    f.close()    

'''训练曲线'''
def train_curve(model):
    scikitplot.estimators.plot_learning_curve(model,train_data, train_label)#训练曲线    

'''测试''' 
def test_SVM(test_data, test_label,model):
    
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
    
    # Kappa系数是基于混淆矩阵的计算得到的模型评价参数(越接近 1 越好)
    kappa = cohen_kappa_score(np.array(Y_test).reshape(-1,1), np.array(Y_pred).reshape(-1,1))
    print('Kappa系数：',kappa)
    
    c_matrics(Y_test,Y_pred)
    
    Y_pred_prob=model.predict_proba(test_data)[:,1]
    ROC_curve(test_label,Y_pred_prob)
    
    result = np.array([Y_test, Y_pred, Y_pred_prob]).T
    np.savetxt(r"结果\SVM验证.txt", result, delimiter = "\t")
    

'''预测'''    
def predict_SVM(model):
    p_svc=model.predict(predict_data)
    pre_p_svc=model.predict_proba(predict_data)[:,1]
    result = np.array([p_svc, pre_p_svc]).T
    np.savetxt(r"结果\SVM预测.txt", result, delimiter = "\t")


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
    

    train_SVM()
    
    #下载模型
    f = open('SVM.pickle','rb')
    SVM= pickle.load(f)
    f.close()  
    
    print('SVM参数：')
    print(SVM) #打印SVM参数
    print('训练曲线：')
    
    
    #测试准确率
    print('20%varify准确率：')
    test_SVM(varify_data, varify_label,SVM)
    
    print('test准确率：')
    test_SVM(test_data, test_label,SVM)


#    预测
    predict_SVM(SVM)
    print('预测完毕')
   
    
    
    
   
    
    
