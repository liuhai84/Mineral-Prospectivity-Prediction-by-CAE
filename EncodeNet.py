import torch.nn as nn
class EncodeNet(nn.Module):
    def __init__(self,judge=True):
        super(EncodeNet, self).__init__()
        self.encoder = nn.Sequential(
           
            nn.Conv2d(1,10, kernel_size=3, stride=2,padding=1), #卷积层1 40*40*10
            nn.ReLU(inplace=True),   #激活函数
            nn.MaxPool2d(kernel_size=2),    #20*20*10

            nn.Conv2d(10,20, kernel_size=3, stride=1,padding=1),  #卷积层2  #20*20*20
            nn.ReLU(inplace=True),  #激活函数
            nn.MaxPool2d(kernel_size=2),  #10*10*20
            
            nn.Conv2d(20, 40, kernel_size=3,stride=1,padding=1), #卷积层3  #10*10*40
            nn.ReLU(inplace=True), 
            nn.Conv2d(40, 40, kernel_size=3,stride=1,padding=1), #卷积层3  #10*10*40
            nn.ReLU(inplace=True), 
            nn.Conv2d(40, 20, kernel_size=3, stride=1,padding=1),  #10*10*20
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2), #5*5*20
            
            nn.Conv2d(20, 1, kernel_size=5, stride=1,padding=0),  #10*10*20
            )
        
        #output=(input-1)*stride+outputpadding-2padding+kernelsize
        #40=(20-1)*2+1-2+3
        #参考网址：https://zhuanlan.zhihu.com/p/39240159
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(1,20,kernel_size=5,stride=1,padding=0,output_padding=0),  #1*1*1-->5*5*20
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(20,40,kernel_size=3,stride=2,padding=1,output_padding=1),  #5*5*20-->10*10*40
            nn.ReLU(inplace=True),
            
            nn.Conv2d(40, 40, kernel_size=3,stride=1,padding=1), #10*10*40-->10*10*40
            nn.ReLU(inplace=True),
        
            nn.ConvTranspose2d(40,20,kernel_size=3,stride=2,padding=1,output_padding=1), #10*10*40-->20*20*20
            nn.ReLU(inplace=True),
        
            nn.ConvTranspose2d(20,10,kernel_size=3,stride=2,padding=1,output_padding=1), #20*20*20-->40*40*10
            nn.ReLU(inplace=True),
        
            nn.ConvTranspose2d(10,1,kernel_size=3,stride=2,padding=1,output_padding=1), #40*40*10-->80*80*1
            nn.ReLU(inplace=True))

    def forward(self, x, judge):
        EnOutputs=self.encoder(x)
        outputs=self.decoder(EnOutputs)
        if judge:
            return outputs
        else:
            return EnOutputs
