

import numpy as np
from collections import deque
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_partitioned_graph(edges_array, node_labels, partition_list):
    G = nx.Graph()
    G.add_nodes_from(node_labels.keys())
    G.add_edges_from(edges_array)
    unique_partitions = list(set(partition_list))
    random_colors = ['#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(1030)]

    
    tableau_colors = list(mcolors.TABLEAU_COLORS.values())

    color_map = tableau_colors + random_colors
    
    if len(unique_partitions) > len(color_map):
        raise ValueError("Not enough colors in the default color map for the number of partitions.")
    partition_colors = {partition: color_map[i] for i, partition in enumerate(unique_partitions)}
    node_colors = [partition_colors[partition_list[node]] for node in G.nodes()]
    pos = nx.spring_layout(G)
    filtered_labels = {n: node_labels[n] for n in G.nodes() if n in node_labels}
    nx.draw(G, pos, labels=filtered_labels, node_color=node_colors, with_labels=True, 
            node_size=50, font_size=5, edge_color='gray')
    plt.show()



def generate_test_graph(num_nodes):
    if num_nodes < 12:
        raise ValueError("Number of nodes should be at least 12 to form two cliques of 6 nodes each.")

    G = nx.Graph()
    clique1_nodes = list(range(6))
    G.add_edges_from([(u, v) for u in clique1_nodes for v in clique1_nodes if u < v])
    clique2_nodes = list(range(6, 12))
    G.add_edges_from([(u, v) for u in clique2_nodes for v in clique2_nodes if u < v])
    inter_clique_edges = [(np.random.choice(clique1_nodes), np.random.choice(clique2_nodes)) for _ in range(5)]
    G.add_edges_from(inter_clique_edges)
    remaining_nodes = list(range(12, num_nodes))
    for node in remaining_nodes:
        if np.random.rand() > 0.5:
            G.add_edge(node, np.random.choice(clique1_nodes))
        else:
            G.add_edge(node, np.random.choice(clique2_nodes))
    return G.edges()

class Graph:
    def __init__(self,edges,n) -> None:
        self.vertices = n
        self.graph = [set() for _ in range(n)]
        self.edges = set()
        
        for row in edges:
            u,v = row[0],row[1]
            e = [u,v]
            self.edges.add(tuple(sorted(e)))
            
            #print(u,v)
            
            self.graph[u].add(v)
            self.graph[v].add(u)

    def remove_edge(self,e):
        u,v = e
        self.edges.remove(e)
        self.graph[u].remove(v)
        self.graph[v].remove(u)



def edge_betweennes(g):
    bw = dict.fromkeys(g.edges,0.0)
    for v in tqdm(range(g.vertices)):
        bw = cal_edgebetweenness(g,v,bw)
    
    e = max(bw, key=bw.get)
    return e,bw[e]/2

def cal_edgebetweenness(g,source,bw):
    bfstraversal = []
    
    parents = [[] for i in range(g.vertices)]
    distance = [-1 for i in range(g.vertices)]
    weight = [0 for i in range(g.vertices)]
    queue = deque([source])
    visited = set([source])
    
    
    distance[source]=0
    weight[source]=1
    while queue:
        current = queue.popleft()
        bfstraversal.append(current)
        
        for neighbour in g.graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                distance[neighbour]=distance[current]+1
                weight[neighbour]=weight[current]
                
                parents[neighbour].append(current)
                queue.append(neighbour)
                
            else:
                if distance[neighbour] == distance[current]+1:
                    weight[neighbour]+=weight[current]
                    #children[current].append(neighbour)
                    parents[neighbour].append(current)
    #print(bfstraversal,weight)
    bw = find_betweenness(bw,bfstraversal,parents,weight)
    return bw


def find_betweenness(bw,bfstraversal,parents,weight):
    # stores sum of adj edges
    adj_edge = dict.fromkeys(bfstraversal,0.0)

    # Now moving backwards and updating edge betweenness
    while bfstraversal:
        t = bfstraversal.pop()
        temp = (1+adj_edge[t])/weight[t]
        for j in parents[t]:
            c = weight[j]*temp
            if (t,j) in bw:
                bw[(t,j)]+=c
            else:
                bw[(j,t)]+=c
            adj_edge[j]+=c
    #print(set(bw.values()))
    return bw

def bfs(g, start, visited):
    component = []
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            component.append(node)
            queue.extend(g.graph[node])
    
    return component

def find_connected_components(g):
    visited = set()
    components = []
    
    for node in range(g.vertices):
        if node not in visited:
            component = bfs(g, node, visited)
            component.sort()
            components.append(component)
    
    return components
                
def Girvan_Newman_one_level(g):
    e = None
    betweenness = None
    components = find_connected_components(g)
    num_components = len(components)
    new_components = components
    new_num_components = num_components
    while num_components>=new_num_components:
        e,betweenness = edge_betweennes(g)
        g.remove_edge(e)
        new_components = find_connected_components(g)
        new_num_components = len(new_components)
        print(betweenness/g.vertices)
    graph_partition = np.zeros((1,g.vertices), dtype=int)
    for comp in new_components:
        id = comp[0]
        for i in comp:
            graph_partition[0,i] = min(id,i)
    x = graph_partition.T.copy()
    return e,betweenness, new_components,x

def Girvan_Newman(nodes_connectivity_list):

    vertices = set()
    for u,v in nodes_connectivity_list:
        vertices.add(u)
        vertices.add(v)
    vertices = list(vertices)
    g = Graph(nodes_connectivity_list,len(vertices))
    adj_mat = array_to_adjmat(nodes_connectivity_list,g.vertices)
    
    graph_partition = np.zeros((1,g.vertices), dtype=int)
    new_components = find_connected_components(g)
    for comp in new_components:
        id = comp[0]
        for i in comp:
            graph_partition[0,i] = min(id,i)
    community_mat_wiki = np.zeros((g.vertices,1), dtype=int)
    community_mat_wiki= graph_partition.T.copy()
    
    i=1
    list_output = []

    list_output.append([None,None,new_components])
    modularity_perlevel = []
    starting_edges = list(g.edges)

    betwenness=float('inf')
    while betwenness/g.vertices>1 and i<3:
    #while len(g.edges)>=1:
        
        print(f"Level {i}: \n")

        e,betwenness, new_components,graph_partition = Girvan_Newman_one_level(g)
        community_mat_wiki = np.concatenate((community_mat_wiki,graph_partition.copy()), axis=1)
        
        #removing edge from adj matrix
        adj_mat[e[0],e[1]],adj_mat[e[1],e[0]]=0,0
        m = modularity(adj_mat,graph_partition,g)
        modularity_perlevel.append(m)
        print(f"Modularity after Level {i} is {m}")

        list_output.append([e,betwenness,new_components])
        i+=1
    print(modularity_perlevel,starting_edges)
    return community_mat_wiki,list_output,modularity_perlevel,starting_edges

def array_to_adjmat(nodes_connectivity_list,n):
    m = np.zeros((n,n),dtype=int)
    for u,v in nodes_connectivity_list:
        m[u][v],m[v][u]=1,1
    return m

def modularity(adj_mat,graph_partition,g):
    temp = np.diag(np.diag(adj_mat))
    adj_mat+=temp
    sum_edge= adj_mat.sum()
    modularity=0
    k = np.zeros((g.vertices,))
    for i in range(len(adj_mat)):
        k[i] = adj_mat[i].sum()
    for u in range(g.vertices):
        for v in range(g.vertices):
            if graph_partition[u]==graph_partition[v]:
                modularity += adj_mat[u][v]-((k[u]*k[v])/(sum_edge))

    modularity/=sum_edge
    return modularity

def louvain_one_iter(nodes_connectivity_list):
    
    vertices = set()
    for u,v in nodes_connectivity_list:
        vertices.add(u)
        vertices.add(v)
    vertices = list(vertices)
    g = Graph(nodes_connectivity_list,len(vertices))
    adj_mat = array_to_adjmat(nodes_connectivity_list,g.vertices)
    
    graph_partition = np.array([i for i in range(g.vertices)])

    com = {i:set() for i in set(graph_partition)}
    for i in range(g.vertices):
        com[graph_partition[i]].add(i)
    #modularity = None
    temp = np.diag(np.diag(adj_mat))
    adj_mat+=temp
    sum_edge= adj_mat.sum() #2m
    #new_modularity=-1
    n=0
    modgain = -1
    while True:
        old_modularity = modularity(adj_mat,graph_partition,g)
        old_mod_gain = 0
        
        for i in tqdm(range(g.vertices)):
            community=graph_partition[i]
            for v in g.graph[i]:
                if graph_partition[i]==graph_partition[v]:
                    continue
                c = graph_partition[v]

                sumin=sum_in(i,c,com,adj_mat)
                ki = adj_mat[i].sum()
                sumtot = adj_mat[list(com[c]),].sum()
                ki_in=0
                for j in com[c]:
                    ki_in+=adj_mat[i][j]
                modgain = (sumin+ki_in)/sum_edge-((sumtot+ki)/sum_edge)**2-sumin/sum_edge-(sumtot/sum_edge)**2-(ki/sum_edge)**2
                if modgain>0:
                    old_mod_gain=max(modgain,old_mod_gain)
                    community = c

            if old_mod_gain >0 and graph_partition[i]!=community:
                n+=1 #number of moves
                #updating comunity graph
                old_c = graph_partition[i]
                graph_partition[i]=min(old_c,community)
                
                com[min(community,old_c)].add(i)
                
                if old_c<community:
                    graph_partition[list(com[community])]=old_c
                    com[old_c].update(com[community])
                    
                    del com[community]
                else:
                    graph_partition[list(com[old_c])]=community
                    com[community].update(com[old_c])
                    del com[old_c]
        
        
        new_modularity=modularity(adj_mat,graph_partition,g)
        print(new_modularity,old_modularity)
        print(graph_partition)
        if new_modularity-old_modularity<=0.01:
            break
    return graph_partition
def sum_in(i,c,com,adj_mat):

    mask = list(com[c])
    s = 0

    for i in com[c]:
        s+=sum(adj_mat[i,mask])
    return s
