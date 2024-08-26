
from common import *
import time

import re
import numpy as np
from collections import deque
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.cluster.hierarchy import dendrogram



#function for cleaning the data
def import_wiki_vote_data(path: str):
    wiki = open(path,"r")
    edges = set()
    #checking for duplicates
    for line in wiki:
        if re.search("^#",line):
            continue
        u,v = map(int,line.strip().split())
        
        edges.add(tuple(sorted([u,v])))

    edges = np.array(list(edges))
    
    return edges

def import_lastfm_asia_data(path):
    df = pd.read_csv(path)
    edges = set()
    df = df.to_numpy(dtype=int)
    for row in df:
        edges.add(tuple(sorted([row[0],row[1]])))
    
    edges = np.array(list(edges))
    return edges


"""
#creating dummy graph with 30 vertices, with two cliques of size 6 and connected to each other sparsely
validating on dummy data

nodes_connectivity_list = np.array((generate_test_graph(30)))
vertices = set()
for u,v in nodes_connectivity_list:
    vertices.add(u)
    vertices.add(v)
vertices = list(vertices)
vertices.sort()
encoding_vertices_wiki = {vertices[i]:i for i in range(len(vertices))}
decoding_vertices_wiki = {i:vertices[i] for i in range(len(vertices))}
for row in nodes_connectivity_list:
    row[0],row[1] = encoding_vertices_wiki[row[0]],encoding_vertices_wiki[row[1]]

st = time.time()
community_mat_last,list_output, modularity_perlevel,starting_edges= Girvan_Newman(nodes_connectivity_list)
et = time.time()
print(f"Time take in girvan is {et-st} seconds")


y = [(x[1]+1)/2 for x in list_output[1:]]
x = [i for i in range(len(modularity_perlevel))]
plt.title("Graph Centrality Values vs Number of Components Formed")
plt.xlabel("Centrality Value Threshold")
plt.ylabel("Number of Components")
plt.plot(x,y)
plt.show()
plt.close()

partition_for_linkage = np.unique(community_mat_last,axis=1)
l = []
for i in range(len(list_output)):
    if not l:
        l.append(list_output[i])
    else:
        if len(l[-1][-1])==len(list_output[i][-1]):
            continue
        else:
            l.append(list_output[i])

def create_dendogram(partition_for_linkage,l):
    cluster = {i:(i,1) for i in range(len(partition_for_linkage))}
    linkage = []
    n = len(partition_for_linkage)
    c = 0
    for i in range(len(l)-1):
        u,v = l[-1-i][0]
        
        id1,id2 = partition_for_linkage[u][-i-1],partition_for_linkage[v][-i-1]
        edge_betweennes = l[-1-i][1]
        c1,l1 = cluster[id1]
        c2,l2 = cluster[id2]
        #print(u,v,id1,id2,c1,c2)
        linkage.append([c1,c2,edge_betweennes,l1+l2])
        cluster[min(id1,id2)] = (n+c,l1+l2)
        c+=1
    
    dn = dendrogram(linkage)
node_labels = {i:f"{i}" for i in range(community_mat_last.shape[0])}
draw_partitioned_graph(nodes_connectivity_list, node_labels, community_mat_last[:,0])
create_dendogram(partition_for_linkage,l)
"""

######### Answers for Dataset 1
nodes_connectivity_list_wiki = import_wiki_vote_data("../data/wiki-Vote.txt")


#Encoding vertices to get them in range 0 to n-1
vertices = set()
for u,v in nodes_connectivity_list_wiki:
    vertices.add(u)
    vertices.add(v)
vertices = list(vertices)
vertices.sort()
encoding_vertices_wiki = {vertices[i]:i for i in range(len(vertices))}
decoding_vertices_wiki = {i:vertices[i] for i in range(len(vertices))}

for row in nodes_connectivity_list_wiki:
    row[0],row[1] = encoding_vertices_wiki[row[0]],encoding_vertices_wiki[row[1]]

# Returning community matrix after m levels, and list of edges,betweenness and components for linkage matrix

community_mat_wiki, list_output, modularity_perlevel, starting_edges = Girvan_Newman(nodes_connectivity_list_wiki)
graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki)
print(len(set(graph_partition_louvain_wiki)))


######### Answers for Dataset 2
nodes_connectivity_list_lastfm = import_lastfm_asia_data("../data/lastfm_asia_edges.csv")

vertices = set()
for u,v in nodes_connectivity_list_lastfm:
    vertices.add(u)
    vertices.add(v)
vertices = list(vertices)
vertices.sort()
encoding_vertices_lastfm = {vertices[i]:i for i in range(len(vertices))}
decoding_vertices_lastfm = {i:vertices[i] for i in range(len(vertices))}

for row in nodes_connectivity_list_lastfm:
    row[0],row[1] = encoding_vertices_lastfm[row[0]],encoding_vertices_lastfm[row[1]]

community_mat_lastfm, list_output, modularity_perlevel, starting_edges = Girvan_Newman(nodes_connectivity_list_lastfm)
graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm)
print(len(set(graph_partition_louvain_lastfm)))