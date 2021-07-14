# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_excel(r"结果\train.xlsx")
X = data.iloc[:, : -1]
X = np.array(X)
y = data.iloc[:, -1]
y = LabelBinarizer().fit_transform(y)
X_train=X
y_train=y

data = pd.read_excel(r"结果\varify.xlsx")
X = data.iloc[:, : -1]
X = np.array(X)
y = data.iloc[:, -1]
y = LabelBinarizer().fit_transform(y)
X_test=X
y_test=y

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)


def add_layer(input_data,in_size,out_size,evalfunc=None):
    Weights=tf.Variable(0.01 * tf.truncated_normal([in_size,out_size]))
    biases=tf.Variable(0.01 * tf.truncated_normal([1,out_size]))
    u=tf.matmul(input_data,Weights)+biases
    if evalfunc==None:
        y=u
    else:
        y=evalfunc(u)
    cache = (Weights, biases)
    return y, cache

xs=tf.placeholder(tf.float32,[None,11])
ys=tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)

h1, cache1 = add_layer(xs, 11, 100, tf.nn.relu)
h1_drop = tf.nn.dropout(h1,keep_prob)

h2, cache2 = add_layer(h1_drop, 100, 50, tf.nn.relu)
h2_drop = tf.nn.dropout(h2, keep_prob)

h3, cache3 = add_layer(h2_drop, 50, 20, tf.nn.relu)
h3_drop = tf.nn.dropout(h3, keep_prob)

prediction, cache3 = add_layer(h3_drop, 20, 2, tf.nn.softmax)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction))
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_pred=tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
step_num =5000

test_accuracy=[]
train_accuracy=[]
test_loss=[]
train_loss=[]


for step in range(step_num):
    sess.run(train_step, feed_dict = {xs: X_train, ys: y_train, keep_prob: 1})
    if step % 1==0:
        
        acc_test=sess.run(accuracy,feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        cost_test=sess.run(loss,feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        
        acc_train=sess.run(accuracy,feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        cost_train=sess.run(loss,feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        
        test_accuracy.append(acc_test)
        train_accuracy.append(acc_train)
        test_loss.append(cost_test)
        train_loss.append(cost_train)
        print('Step:{},accuracy_train:{},accuracy_test:{},'.format(step,acc_train,acc_test))

result = np.array([test_accuracy,train_accuracy,test_loss,train_loss]).T
np.savetxt(r"结果\accuracy_loss.txt", result, delimiter = "\t")

#验证        
y_predd=sess.run(prediction,feed_dict={xs: X_test, keep_prob: 1})
ypred = np.argmax(y_predd, axis = 1)
ytest = np.argmax(y_test, axis = 1)
print('varify准确率：',metrics.accuracy_score(ypred,ytest))
result_gailv=np.array(y_predd)
np.savetxt(r"结果\varify概率结果.txt", result_gailv, delimiter = "\t")
result= np.array([ypred,ytest]).T
np.savetxt(r"结果\varify结果.txt", result, delimiter = "\t")

#测试
data = pd.read_excel(r"结果\test.xlsx")
X_varify = data.iloc[:, : -1]
X_varify = np.array(X_varify)
y_test1 = data.iloc[:, -1]
y_test1 = LabelBinarizer().fit_transform(y_test1)
y_test1  = to_categorical(y_test1 , num_classes=None)

y_pred11=sess.run(prediction,feed_dict={xs: X_varify, keep_prob: 1})
ypred1 = np.argmax(y_pred11, axis = 1)
ytest1 = np.argmax(y_test1, axis = 1)
print('test准确率：',metrics.accuracy_score(ypred1,ytest1))
result1= np.array([ypred1,ytest1]).T
np.savetxt(r"结果\test结果.txt", result1, delimiter = "\t")
result_gailv=np.array(y_pred11)
np.savetxt(r"结果\test概率结果.txt", result_gailv, delimiter = "\t")

#预测
data = pd.read_excel(r"结果\predict.xlsx")
X_predict = data.iloc[:, : -1]
X_predict = np.array(X_predict)
y_pred2=sess.run(prediction,feed_dict={xs: X_predict, keep_prob: 1})
result2= np.array(y_pred2)
np.savetxt(r"结果\BP预测.txt", result2, delimiter = "\t")
sess.close()








