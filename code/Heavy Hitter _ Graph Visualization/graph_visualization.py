import networkx as nx
import matplotlib.pyplot as plt
import os
import csv

G = nx.Graph()
connections = set()
nodes = set()

data_file = ''

for dirname, _, filenames in os.walk('/home/sun/Desktop/보안프로젝트'):
  for filename in filenames:
      if filename.endswith('.csv'):
          data_file = filename

          print("Find Data File : {}".format(data_file))

f = open(data_file, 'r', encoding='utf-8')

rdr = csv.reader(f)

for idx, line in enumerate(rdr):

    if idx == 0: continue


    src_ip = line[1].split('.')
    dst_ip = line[2].split('.')

    src_ip = src_ip[0] + '.' + src_ip[1] + '.' + src_ip[2]
    dst_ip = dst_ip[0] + '.' + dst_ip[1] + '.' + dst_ip[2]

    src_node = (src_ip, int(line[6]))
    dst_node = (dst_ip, int(line[6]))

    nodes.add(src_node)
    nodes.add(dst_node)
    connections.add((src_node, dst_node))

    if idx == 3000: break

G.add_nodes_from(nodes)
G.add_edges_from(connections)

plt.rcParams['figure.figsize'] = 300, 300

pos = nx.spring_layout(G, scale=1.0, iterations=100)
# nx.draw(G, pos, node_color='c',edge_color='k', with_labels=True)
nx.draw(G, pos)

plt.show()
