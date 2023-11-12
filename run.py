#!/usr/bin/env python
# coding: utf-8

# In[54]:


from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

# 股票代码集合
symbols =  ["sh000001", "sh601398", "sh601857", "sh601288", "sh601988", "sh600036", "sh601328", "sh601166", "sh601668", "sh600104", "sh601318", "sh600519", "sh600028", "sh601169", "sh601601", "sh600000", "sh601988", "sh600016", "sh600887", "sh601628", "sh601688", "sh601818", "sh601088", "sh601988", "sh601668", "sh600104", "sh601318", "sh600519", "sh600028", "sh601169", "sh601601", "sh600000", "sh601988", "sh600016", "sh600887", "sh601628", "sh601688", "sh601818", "sh601088", "sh601988", "sh601668", "sh600104", "sh601318", "sh600519", "sh600028", "sh601169", "sh601601", "sh600000", "sh601988", "sh600016", "sh600887"]
print(len(symbols))
for symbol in symbols:
    print(f'Processing {symbol}..')
    # 获取数据
    df = ak.stock_zh_a_daily(symbol=symbol, start_date="20201101", end_date="20231106")
    df



    # In[67]:


    df1= df[['date','open','high','low','close','volume','amount','turnover']]

    scaler1=MinMaxScaler()
    None_data=scaler1.fit_transform(df1[['open','close']])
    scaler=MinMaxScaler()
    df1[['open','high','low','close','volume','amount','turnover']]=scaler.fit_transform(df1[['open','high','low','close','volume','amount','turnover']])





    # In[56]:


    df1["date"] = pd.to_datetime(df1["date"], format="%Y-%m-%d")
    # 判断每个日期属于星期几,其中 Monday 为 0，Tuesday 为 1，以此类推
    df1['week_day'] = df1['date'].dt.dayofweek

    # 判断月初 月中 月末
    yczm = []
    mr = []
    for i in df.index:
        items = int(str(df1.loc[i,'date']).split(' ')[0].split("-")[-1])
        mr.append(items)
        if items < 10:
            #月初用0代表
            yczm.append(0)
        elif items < 20:
            #月中用1代表
            yczm.append(1)
        else:
            #月末用2代表
            yczm.append(2)
    df1['day'] = mr
    df1['month_period'] = yczm
    df1.columns


    # In[57]:


    #Params
    time_window = 28
    pred_window=7

    BATCH_SIZE=128
    nums_layer=1
    lr=5e-4
    test_size=0.2
    N_epoch=200
    device = torch.device('mps')

    feature_count=10
    Price_data=df1[['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover',
           'week_day', 'day', 'month_period']].values


    #print(Price_data.shape)

    data_his=[]
    data_pred=[]
    for i in range(len(Price_data)-time_window-pred_window):
        data_his.append(np.array(Price_data[i:i+time_window,]))
        data_pred.append(np.array(Price_data[i+time_window:i+time_window+pred_window,[0,3]]))

    data_pred=np.array(data_pred)
    data_his=np.array(data_his)

    #print(data_his.shape,data_pred.shape)
    #print(len(data_his),len(data_pred))


    # In[58]:


    data_his=np.array(data_his)
    data_pred=np.array(data_pred)

    #print(data_his.shape,data_pred.shape)

    Train_x,Train_y=data_his[:-int(test_size*len(data_his))],data_pred[:-int(test_size*len(data_his))]
    Test_x,Test_y=data_his[-int(test_size*len(data_his)):],data_pred[-int(test_size*len(data_his)):]
    #print(Train_x.shape,Train_y.shape,Test_x.shape,Test_y.shape)

    class TimeDataset(Dataset):
        def __init__(self,df_x,df_y):
            self.data=df_x
            self.label=df_y
        def __len__(self):
            return len(self.data)
        def __getitem__(self,index):
            data_value=torch.FloatTensor(self.data[index,:])
            label_value=torch.FloatTensor([self.label[index,:]])
            return data_value,label_value.view(pred_window,2)

    train_dataset=TimeDataset(df_x=Train_x,df_y=Train_y)
    valid_dataset=TimeDataset(df_x=Test_x,df_y=Test_y)
    train_iterator=DataLoader(train_dataset,batch_size=BATCH_SIZE)
    valid_iterator=DataLoader(valid_dataset,batch_size=BATCH_SIZE)
    #test
    for (data,label) in train_iterator:
        #print(data.shape)
        #print(label.shape)
        break


    # In[59]:


    class FCN(nn.Module):
        def __init__(self,InputDim=time_window*feature_count,OutputDim=pred_window*2,nums_layer=6):
            super().__init__()
            layers=[]
            for i in range(nums_layer):
                layers+=[nn.Linear(4096,4096),
                         nn.LeakyReLU(),
                         nn.Linear(4096,4096)
                         ]
            self.layer1=nn.Sequential(
                nn.Linear(InputDim, 4096),
                nn.LeakyReLU(),
            )
            self.layer2=nn.Sequential(*layers)
            self.layer3=nn.Sequential(
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                nn.Linear(4096,64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32,out_features=OutputDim),

            )

        def forward(self,data):
            x=self.layer1(data)
            x=self.layer2(x)
            x=self.layer3(x)
            return x

    def train(model, iterator, optimizer, criterion,device='cpu'):
        epoch_loss = 0
        model=model.to(device)
        model.train()
        for batch in iterator:
            batch[0]=batch[0].reshape(batch[0].shape[0],-1).to(device)
            batch[1]=batch[1].reshape(batch[1].shape[0],-1).to(device)
            criterion=criterion.to(device)
            optimizer.zero_grad()
            predictions = model(batch[0])

            ##print(predictions.shape,batch[1].shape)

            loss = criterion(predictions, batch[1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(model, iterator,criterion,device='cpu'):
        epoch_loss = 0
        model=model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in iterator:
                batch[0]=batch[0].reshape(batch[0].shape[0],-1).to(device)
                batch[1]=batch[1].reshape(batch[1].shape[0],-1).to(device)
                criterion=criterion.to(device)
                predictions = model(batch[0])
                loss = criterion(predictions, batch[1])
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    lossfunction=nn.MSELoss()
    fc_model=FCN(nums_layer=nums_layer)
    # fc_model=FullyConnectedModel()
    optimizer=torch.optim.Adam(fc_model.parameters(),lr=lr)


    # In[60]:


    train_loss_list=[]
    valid_loss_list=[]

    par=tqdm(range(N_epoch))

    for i in par:
        train_loss=train(model=fc_model,iterator=train_iterator,criterion=lossfunction,optimizer=optimizer,device=device)
        valid_loss=evaluate(model=fc_model,iterator=valid_iterator,criterion=lossfunction,device=device)
        # #print("Epoch:{}".format(i+1))
        # #print("Training Loss:{:.4f}".format(train_loss),"Valid Loss:{:.4f}".format(valid_loss))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        if i ==0:
            best_valid_loss=valid_loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # #print('Save Model..')
            par.set_description_str("Epoch:{}".format(i+1)+' Save!')
            torch.save(fc_model.state_dict(), './result/fc_model.pt')
        else:
            par.set_description_str("Epoch:{}".format(i+1))

        par.set_postfix_str("Training Loss:{:.4f}".format(train_loss)+" Valid Loss:{:.4f}".format(valid_loss))


    # In[61]:


    fc_model.load_state_dict(torch.load('./result/fc_model.pt',map_location='cpu'))
    fc_model.to('cpu')
    train_iterator=DataLoader(train_dataset,batch_size=1)
    valid_iterator=DataLoader(valid_dataset,batch_size=1)
    list1=[]
    list2=[]
    real_y_1=[]
    real_y_2=[]

    with torch.no_grad():
        for batch in train_iterator:
            data=batch[0].reshape(batch[0].shape[0],-1).to('cpu')
            pre1=fc_model(data)[0][0]
            list1.append(pre1)
            real_y_1.append(batch[1].to('cpu')[0][0][0])

        for batch in valid_iterator:
             data=batch[0].reshape(batch[0].shape[0],-1).to('cpu')
             pre2=fc_model(data)[0][0]
             list2.append(pre2)
             real_y_2.append(batch[1].to('cpu')[0][0][0])

    list2=np.array([i.detach().numpy() for i in list2]).reshape(-1)
    list1=np.array([i.detach().numpy() for i in list1]).reshape(-1)
    real_y_1=np.array([i.detach().numpy() for i in real_y_1]).reshape(-1)
    real_y_2=np.array([i.detach().numpy() for i in real_y_2]).reshape(-1)


    # In[62]:


    #print(len(list1),len(list2),len(real_y_1),len(real_y_2))


    # In[63]:


    plt.rcParams['font.family']='serif'
    plt.figure(figsize=(15,6),dpi=300)
    plt.title('10 Feature Stock Price')
    plt.plot(range(len(real_y_1)),real_y_1,color='blue',label='Real')
    plt.plot(range(len(list1)),list1,label='Train Predicted')
    plt.plot(range(len(real_y_1)+1,len(real_y_1)+len(list2)+1),list2,label='FCN Test Predicted')
    plt.plot(range(len(real_y_1)+1,len(real_y_1)+len(list2)+1),real_y_2,label='FCN Test Real')
    # plt.plot(range(len(Y_train_p)+1,len(Y_train_p)+len(Yvalidation)+1),Yvalidation)
    # plt.plot(validation.index,pd.DataFrame(LSTM_pred),color='red',label='LSTM Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.axvline(len(real_y_1),color='black')
    plt.legend(loc = "upper left")
    plt.savefig(f'./result/ResultImage/{symbol} 4 Feature Prediction.jpg')


    # In[68]:


    #prediction
    pred_data=torch.FloatTensor(Price_data[-time_window:].reshape(1,-1))
    pre1=fc_model(pred_data)[0].reshape(pred_window,2)
    result=scaler1.inverse_transform(pre1.detach().numpy())
    pd.DataFrame(result).to_csv(f'./result/NumberResult/{symbol}.csv')


