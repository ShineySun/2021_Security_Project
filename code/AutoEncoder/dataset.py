import pandas as pd
import numpy as np

# from tensorflow.keras.layers import Embedding

def ip_to_bin(ip):

    ip_ = ip.split('.')

    binary = []
    cls = '0'
    for index,i in enumerate(ip_):
        # print(bin(int(i)))
        if index != 3:
            cls += i
        ip_bin = bin(int(i))[2:]
        ip_bin = '0'*(8-len(ip_bin)) + ip_bin
        # binary.append(np.array(list(ip_bin),dtype=int))
        temp = []
        for i in ip_bin:
            r = [0,0]
            r[int(i)] = 1
            temp.append(np.array(r))
        binary.append(np.array(temp))
        # binary.append(int(i))
    return binary,int(cls)

data = pd.read_csv('./04_hashed.csv')

#print(data)

# print(data['src_ip'])
# emb = Embedding(256,20)
X = []
Y = []
vocab = {}
vocab['<PAD>'] = 0
vindex = 1
ck = {}
index = 0

from collections import defaultdict

map = defaultdict(list)
datatime = []
for date,i,d,a,sp,dp,sc,dc in zip(data['Rdate'],data['src_ip'],data['dst_ip'],data['Action'],data['src_port'],data['dst_port'],data['src_country'],data['dst_country']):
    #binary,cls = ip_to_bin(i)
    #binary2,_ = ip_to_bin(d)
    # print(binary)
    #sp = sp
    #dp = dp
    classip = i.split('.')
    classip = '.'.join(classip[:-1])    
    # print(classip)
    map[str(date)[:12]+classip].append([i,d,a,sp,dp,sc,dc])
    # print(str(date)[:12])
    country = []

    if sc not in ck:
        ck[sc] = index
        index += 1
    if dc not in ck:
        ck[dc] = index
        index += 1

    country = [ck[sc],ck[dc]]
    country = np.array(country)
    country = country/242
    port = []
    binary = []
    datatime.append(str(date))
    for tk in i.split('.'):
        binary.append(int(tk))
    for tk in d.split('.'):
        binary.append(int(tk))
    
    
    ip = np.array(binary)
    ip = ip/255
    port = [sp,dp]
    # port.append(dp)
    port = np.array(port)
    port = port/65535

    feature = np.concatenate((ip,port,country))
    # feature = np.reshape(feature,(12,1))
    # feature = np.concatenate((ip,port))
    X.append(feature)
    Y.append(np.array([int(a)]))
    # break

print(max(port))
print(max(ck.values()))
X = np.array(X)
print(X.shape)

deny = []
denytime = []
for x,y,dt in zip(X,Y,datatime):
    if y[0] == 2:
        deny.append(x)
        denytime.append(dt)

allow = []
# datatime =[]
allowtime = []
for x,y,dt in zip(X,Y,datatime):
    if y[0] != 2:
        allow.append(x)
        allowtime.append(dt)
deny = np.array(deny)
allow = np.array(allow)
denytime = np.array(denytime)
allowtime = np.array(allowtime)

X_train = X[1164062:]
X_test = X[:1164062]

np.save('allow',allow)
np.save('deny',deny)
np.save('train',X_train)
np.save('test',X_test)
np.save('denytime',denytime)
np.save('allowtime',allowtime)

Y = np.array(Y)

Y_train = Y[1164062:]
Y_test = Y[:1164062]

np.save('y_train',Y_train)
np.save('y_test',Y_test)

print(Y)

import pickle
with open('vocab.pkl','wb') as f:
    pickle.dump(vocab,f)
with open('vocab.pkl','wb') as f:
    vocabi2w = { v:k for k,v in vocab.items()}
    pickle.dump(vocabi2w,f)
with open('country.pkl','wb') as f:
    pickle.dump(ck,f)
with open('datatimedist.pkl','wb') as f:
    pickle.dump(map,f)
    # 4