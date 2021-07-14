# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
import argparse
from data_loader_channels import MyDataset
from EncodeNet import EncodeNet  #结构为EncodeNet1
import pandas as pd
import cv2

parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=1000, type=int, help='number of epochs tp train for')
args = parser.parse_args()
 
class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        self.train_loader = MyDataset(datatxt='Data1/train/data_list.txt', transform=transforms.ToTensor())
        self.varify_loader= MyDataset(datatxt='Data1/verify/data_list.txt', transform=transforms.ToTensor())
        self.test_loader = MyDataset(datatxt='Data1/test/data_list.txt', transform=transforms.ToTensor())
        self.predict_loader = MyDataset(datatxt='Data1/predict/data_list.txt', transform=transforms.ToTensor())
    
    def load_model(self):
        self.device = torch.device('cpu')
        self.model = EncodeNet(judge=True) 
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)  #Adam优化器
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)#default=0.001
        self.criterion = nn.MSELoss().to(self.device) #criterion定义为交叉熵损失

    def train(self):
        print("train:", end='')
        self.model.train()  
        train_loss = 0
        total = 0
        output_N_sum=0
        output_P_sum=0
        output_N_average=0
        output_P_average=0
        N=0
        P=0

        for batch_num, (data, target) in enumerate(self.train_loader):  #加载训练样本，数据是data，标签是target
            data= data.view(1,11,80,80)
            target=torch.tensor([target])
            target1=target.cpu().numpy()
            self.optimizer.zero_grad() #梯度初始化为零
            # forward
            output = self.model(data,True)  

            if(target1==0):
                output_N=output.cpu().detach().numpy()
                output_N_sum+=output_N
                N=N+1
                output_N_average=output_N_sum/N    
            if(target1==1):
                output_P=output.cpu().detach().numpy()
                output_P_sum+=output_P
                P=P+1
                output_P_average=output_P_sum/P      
            loss = self.criterion(output, data) #计算output和target的损失值

            if (batch_num==715):  #train的总数量
                nn=output_N_average
                pp=output_P_average                
                d=abs(np.sqrt(np.sum(np.square(nn-pp))))  #欧式距离                
                loss=loss+1/d
            
            # backward
            loss.backward() #误差反向传播
            self.optimizer.step() #优化器优化参数

            train_loss += loss.item()  #loss的叠加
            total += target.size(0) #total是当前样本总数量
        Loss = train_loss / len(self.train_loader)
        print('Loss: %.4f ' % Loss)
        return Loss

    def test(self):
        print("test:", end='')
        self.model.eval()
        test_loss = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.varify_loader): #载入测试数据
                data= data.view(1,11,80,80) 
                target=torch.tensor([target])
                output = self.model(data,True) #测试数据进入模型
                loss = self.criterion(output, data) #计算损失
                test_loss += loss.item() 
                total += target.size(0) 
            Loss=  test_loss/ len(self.varify_loader)     
            print('Loss: %.4f ' % Loss)
        return Loss
    
    def run(self):
        self.load_data() #加载训练数据和验证数据
        self.load_model() #加载模型

        loss_list_train = []
        loss_list_test = []
        min_test_loss=10000

        for epoch in range(1, self.epochs + 1):  #epochs=50
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))

            train_result = self.train()  
            loss_list_train.append(train_result)

            test_result = self.test()  
            loss_list_test.append(test_result)

            
#每100次保存一次模型
            if (epoch%100==0):
                 a=str(epoch)    
                 model_out_path = "model_" +a+ ".pth"
                 torch.save(self.model, model_out_path)
                 print("Checkpoint saved to {}".format(model_out_path))

            if(test_result<=min_test_loss):
                epoch_best=epoch
                min_test_loss=test_result
                model_out_path = "model_best.pth"
                torch.save(self.model, model_out_path)

            np.savetxt(r"结果\losstrain.txt", loss_list_train, delimiter = "\t")
            np.savetxt(r"结果\losstest.txt",  loss_list_test, delimiter = "\t")
        
        print('迭代完成')
        print('最优epoch:',epoch_best)
        print('最优loss:',min_test_loss)
    
    
    def Get_middle_feature(self, model_name):  
        print("train_get_middle_feature:")
        model = torch.load(model_name) 
        model.eval() 
        result=[]
        label=[]
        if self.test_loader is None:
            self.load_data()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.train_loader):   
                print(batch_num)
                data= data.view(1,11,80,80) 
                output = model(data,False)#输出
                output = output.view(output.size(0), -1)
                if (batch_num==0): 
                    output = output.numpy()
                    m=output
                if (batch_num>=1): 
                    output = output.numpy()
                    output1=np.append(m,output, axis=0)
                    m=output1
                label.append(target)               
            
            result = np.array(label).T
            
            test=pd.DataFrame(data=result)
            test.to_csv(r"结果\train_label.csv") 
            test=pd.DataFrame(data=output1)
            test.to_csv(r"结果\train.csv")
            
            for batch_num, (data, target) in enumerate(self.varify_loader):    
                print(batch_num)
                data= data.view(1,11,80,80) 
                output = model(data,False)#输出
                output = output.view(output.size(0), -1)
                if (batch_num==0): 
                    output = output.numpy()
                    m=output
                if (batch_num>=1): 
                    output = output.numpy()
                    output1=np.append(m,output, axis=0)
                    m=output1
                label.append(target)               
            
            result = np.array(label).T
            
            test=pd.DataFrame(data=result)
            test.to_csv(r"结果\varify_label.csv") 
            
            test=pd.DataFrame(data=output1)
            test.to_csv(r"结果\varify.csv") 
            
            

            for batch_num, (data, target) in enumerate(self.test_loader):   
                print(batch_num)
                data= data.view(1,11,80,80) 
                output = model(data,False)#输出
                output = output.view(output.size(0), -1)
                if (batch_num==0): 
                    output = output.numpy()
                    m=output
                if (batch_num>=1): 
                    output = output.numpy()
                    output1=np.append(m,output, axis=0)
                    m=output1
                label.append(target)               
            
            result = np.array(label).T
            
            test=pd.DataFrame(data=result)
            test.to_csv(r"结果\test_label.csv") 
            
            test=pd.DataFrame(data=output1)
            test.to_csv(r"结果\test.csv")
##
            for batch_num, (data, target) in enumerate(self.predict_loader):    
                print(batch_num)
                data= data.view(1,11,80,80) 
                output = model(data,False)#输出
                output = output.view(output.size(0), -1)
                if (batch_num==0): 
                    output = output.numpy()
                    m=output
                if (batch_num>=1): 
                    output = output.numpy()
                    output1=np.append(m,output, axis=0)
                    m=output1
                label.append(target)               
            
            result = np.array(label).T
            
            test=pd.DataFrame(data=result)
            test.to_csv(r"结果\predict_label.csv") 
            
            test=pd.DataFrame(data=output1)
            test.to_csv(r"结果\predict.csv")
        
if __name__ == '__main__':
    solver = Solver(args)
    solver.run() #第一步：训练得到自编码模型
    solver.Get_middle_feature("model_1000.pth")