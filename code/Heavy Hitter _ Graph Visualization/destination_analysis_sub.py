import os
import csv
import tqdm
import collections

class streamTrend:
    def __init__(self, noOfTopicsToTrack = 100):
        # number of items to track for heavy hitting
        self.noOfTopics = noOfTopicsToTrack
        self.topicsList = collections.Counter()

    def updateItemHit(self, data):
        if data in self.topicsList:
            self.topicsList[data] += 1
        elif len(self.topicsList) < self.noOfTopics:
            self.topicsList[data] = 1
        else:
            for topic in self.topicsList:
                self.topicsList[topic] -= 1
            # remove 0 or -ve counts
            self.topicsList += collections.Counter()

    def getTrends(self):
        return self.topicsList.most_common()


data_file = ''

for dirname, _, filenames in os.walk('/home/sun/Desktop/보안프로젝트'):
  for filename in filenames:
      if filename.endswith('.csv'):
          data_file = filename

          print("Find Data File : {}".format(data_file))


most_src_ip = [
    '154.58.159',
    '253.211.157',
    '72.192.214',
    '221.20.249',
    '12.150.252',
    '197.16.226',
    '207.161.122',
    '45.111.16',
    '121.7.158',
    '74.120.129',
]


most_dst_ip = [
    '154.78.95',
    '140.65.156',
    '173.224.148',
    '72.192.214',
    '45.111.16',
    '52.77.83'
]


f = open(data_file, 'r', encoding='utf-8')

rdr = csv.reader(f)

src_ip_hitter = streamTrend(50)
not_src_ip_hitter = streamTrend(50)

action_count = [0,0,0]
non_action_count = [0,0,0]

for idx, line in enumerate(rdr):

    if idx == 0: continue


    src_ip = line[1].split('.')
    dst_ip = line[2].split('.')

    src_ip = src_ip[0] + '.' + src_ip[1] + '.' + src_ip[2]
    dst_ip = dst_ip[0] + '.' + dst_ip[1] + '.' + dst_ip[2]


    if dst_ip in most_dst_ip:

        action = int(line[6])

        action_count[action] += 1

        src_ip_hitter.updateItemHit(src_ip)
    else:
        action = int(line[6])

        non_action_count[action] += 1

        not_src_ip_hitter.updateItemHit(src_ip)


    # src_ip_hitter.updateItemHit(src_ip)
    # dst_ip_hitter.updateItemHit(dst_ip)

most_src_ip = src_ip_hitter.getTrends()
most_non_src_ip = not_src_ip_hitter.getTrends()

print("most_src_ip : {}".format(most_src_ip))
print("most_non_src_ip : {}".format(most_non_src_ip))
print("action_count : {}".format(action_count))
print("non_action_ip : {}".format(non_action_count))
