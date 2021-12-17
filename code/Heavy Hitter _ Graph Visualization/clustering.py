import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import ipaddress
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline


for dirname, _, filenames in os.walk('/home/sun/Desktop/보안프로젝트'):
  for filename in filenames:
      if filename.endswith('.csv'):
          data_file = filename

          print("Find Data File : {}".format(data_file))

# f = open(data_file, 'r', encoding='utf-8')
#
# rdr = csv.reader(f)

df = pd.read_csv(data_file)

df['dst_ip'] = df['dst_ip'].apply(ipaddress.IPv4Address).astype(int)
df['src_ip'] = df['src_ip'].apply(ipaddress.IPv4Address).astype(int)

datas = df.to_numpy()

train_data = list()
test_data  = list()

for data in datas:
    src_ip = float(data[1])
    dst_ip = float(data[2])
    proto = float(data[3])
    src_port = float(data[4])
    dst_port = float(data[5])
    action_feature = np.zeros(3)
    action = int(data[6])

    action_feature[action] = 1

    feature = np.array([
            src_ip,
            dst_ip,
            proto,
            src_port,
            dst_port
    ])

    feature = np.concatenate((feature, action_feature), axis=None)

    if action != 0:
        train_data.append(feature)
    else:
        test_data.append(feature)

np.save('train_data', train_data)
np.save('test_data', test_data)


# feature = df[['src_ip', 'dst_ip', 'src_port']]
# feature = df[['src_ip', 'src_port']]
#
# print(feature)
#
# scaler = StandardScaler()
# model = KMeans(n_clusters=2, algorithm='auto')
#
# pipeline = make_pipeline(scaler, model)
#
# pipeline.fit(feature)
#
# predict = pd.DataFrame(pipeline.predict(feature))
# predict.columns = ['predict']
#
# r = pd.concat([feature, predict], axis=1)
#
# # plt.scatter(r['src_ip'],r['dst_ip'],c=r['predict'],alpha=0.5)
# plt.scatter(r['src_ip'],r['src_port'],c=r['predict'],alpha=0.5)
#
# centers = pd.DataFrame(model.cluster_centers_,columns=['src_ip','src_port'])
#
# center_x = centers['src_ip']
#
# center_y = centers['src_port']
#
# plt.scatter(center_x,center_y,s=50,marker='D',c='r')
#
# plt.show()


# for idx, line in enumerate(rdr):
#
#     if idx == 0: continue
#
#     src_ip = line[1].apply(ipaddress.IPv4Address).astype(int)
#     dst_ip = line[2].apply(ipaddress.IPv4Address).astype(int)
#
#     print(src_ip)
#     print(dst_ip)
