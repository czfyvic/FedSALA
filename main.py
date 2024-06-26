import time
from copy import deepcopy
import pyro

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer, BlogCatalog, Flickr, Facebook, Gowalla
from dhg.models import GAT, GCNS, GCNT0, GCNT1, GIN, HyperGCN, GraphSAGE
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np
from scipy import sparse
import random
import pandas as pd
import collections
import networkx as nx
from scipy.sparse import csr_matrix
import pickle
import math
import cmath
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import warnings
import matplotlib.pyplot as plt
import louvain.community as community_louvain
from sklearn import preprocessing, model_selection
from stellargraph.core.graph import StellarGraph
import stellargraph as sg
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore', category=FutureWarning)


class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        # 节点编号
        self._vid = vid
        # 社区编号
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in  # 结点内部的边的权重


class Louvain:
    def __init__(self, G):
        self._G = G
        self._m = 0  # 边数量 图会凝聚动态变化
        self._cid_vertices = {}  # 需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}  # 需维护的关于结点的信息(结点编号，相应的Vertex实例)
        for vid in self._G.keys():
            # 刚开始每个点作为一个社区
            self._cid_vertices[vid] = {vid}
            # 刚开始社区编号就是节点编号
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            # 计算边数  每两个点维护一条边
            self._m += sum([1 for neighbor in self._G[vid].keys()
                           if neighbor > vid])

    # 模块度优化阶段
    def first_stage(self):
        mod_inc = False  # 用于判断算法是否可终止
        visit_sequence = self._G.keys()
        # 随机访问
        random.shuffle(list(visit_sequence))
        while True:
            can_stop = True  # 第一阶段是否可终止
            # 遍历所有节点
            for v_vid in visit_sequence:
                # 获得节点的社区编号
                v_cid = self._vid_vertex[v_vid]._cid
                # k_v节点的权重(度数)  内部与外部边权重之和
                k_v = sum(self._G[v_vid].values()) + \
                    self._vid_vertex[v_vid]._kin
                # 存储模块度增益大于0的社区编号
                cid_Q = {}
                # 遍历节点的邻居
                for w_vid in self._G[v_vid].keys():
                    # 获得该邻居的社区编号
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        # tot是关联到社区C中的节点的链路上的权重的总和
                        tot = sum(
                            [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        # k_v_in是从节点i连接到C中的节点的链路的总和
                        k_v_in = sum(
                            [v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        # 由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        delta_Q = k_v_in - k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q

                # 取得最大增益的编号
                cid, max_delta_Q = sorted(
                    cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                if max_delta_Q > 0.0 and cid != v_cid:
                    # 让该节点的社区编号变为取得最大增益邻居节点的编号
                    self._vid_vertex[v_vid]._cid = cid
                    # 在该社区编号下添加该节点
                    self._cid_vertices[cid].add(v_vid)
                    # 以前的社区中去除该节点
                    self._cid_vertices[v_cid].remove(v_vid)
                    # 模块度还能增加 继续迭代
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
        return mod_inc

    # 网络凝聚阶段
    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        # 遍历社区和社区内的节点
        for cid, vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            # 将该社区内的所有点看做一个点
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                # k,v为邻居和它们之间边的权重 计算kin社区内部总权重 这里遍历vid的每一个在社区内的邻居   因为边被两点共享后面还会计算  所以权重/2
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            # 新的社区与节点编号
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex

        G = collections.defaultdict(dict)
        # 遍历现在不为空的社区编号 求社区之间边的权重
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                # 找到cid后另一个不为空的社区
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                # 遍历 cid1社区中的点
                for vid in vertices1:
                    # 遍历该点在社区2的邻居已经之间边的权重(即两个社区之间边的总权重  将多条边看做一条边)
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        # 更新社区和点 每个社区看做一个点
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def execute(self):
        iter_time = 1
        while True:
            iter_time += 1
            # 反复迭代，直到网络中任何节点的移动都不能再改善总的 modularity 值为止
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()


# 可视化划分结果
def showCommunity(G, partition, pos):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号

    # 可视化节点
    colors = ['r', 'g', 'b', 'y', 'm']
    shapes = ['v', 'D', 'o', '^', '<']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape=shapes[index],
                               node_size=350,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)

    for index, edgelist in enumerate(edges.values()):
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        else:
            # cluster间
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index])

    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.axis('off')
    plt.show()


def cal_Q(partition, G):  # 计算Q
    # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    m = len(G.edges(None, False))
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            # G.neighbors(node)找node节点的邻接节点
            t += len([x for x in G.neighbors(node)])
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

class GraphXX:
    graph = nx.DiGraph()

    def __init__(self):
        self.graph = nx.DiGraph()

    def createGraph(self, edges):
        for edge in edges:
            self.graph.add_edge(*(edge[0], edge[1]))

        return self.graph

    # def createGraph(self, filename):
    #     file = open(filename, 'r')
    #
    #     for line in file.readlines():
    #         nodes = line.split()
    #         edge = (int(nodes[0]), int(nodes[1]))
    #         self.graph.add_edge(*edge)
    #
    #     return self.graph


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs, outlist = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
        return res
    else:
        res = evaluator.test(lbls, outs)
        predict = np.argmax(outs, axis=-1)
        recall0 = recall_score(lbls, predict, average='macro')
        precision0 = precision_score(lbls, predict, average='macro')
        f0 = f1_score(lbls, predict, average='macro')

        print("accuracy:", res["accuracy"], "f1:", f0, "recall:", recall0, "precision", precision0)

        return res, f0, recall0, precision0

        # recall1 = recall_score(lbls, predict, average='micro')
        # precision1 = precision_score(lbls, predict, average='micro')
        # f1 = f1_score(lbls, predict, average='micro')


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def collectingData(data):
    feature = data["features"].numpy()
    labels = data["labels"].numpy()
    edges = data["edge_list"]

    np.save("BlogCatalog/feature.npy", feature)
    np.save("BlogCatalog/edges.npy", edges)
    np.save("BlogCatalog/labels.npy", labels)

    return None

def getSamplingMatrix(samplingGlobalAdjWithReduceNode, sampling_idx_range):
    adjLen = len(samplingGlobalAdjWithReduceNode)
    samplingMatrix = np.zeros((adjLen, adjLen))

    for idx in sampling_idx_range:
        currentList = samplingGlobalAdjWithReduceNode[idx]
        for listIdx in currentList:
            samplingMatrix[idx, listIdx] = 1

    return samplingMatrix

def getSamplingAdj1(adjList, sampling_idx_range): #using for global adj and take the node out of sampling nodes
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(listIndex)
    return newNodeAdjList

def getSamplingAdj(adjList, sampling_idx_range):   #the index of sampling_idx_range is the current node index
    newNodeAdjList = []
    for listIndex in adjList:
        withInFlag = listIndex in sampling_idx_range
        if withInFlag:
            newNodeAdjList.append(sampling_idx_range.index(listIndex))
    return newNodeAdjList

def getSamplingGlobalAdj(graph, sampling_idx_range):  #pos the sampling nodes in global adj
    adjLen = len(graph)
    samplingGlobalAdj = collections.defaultdict(list)
    for idx in range(adjLen):
        withInFlag = idx in sampling_idx_range
        if withInFlag:
            currentList = graph[idx]
            newCurrentList = getSamplingAdj1(currentList, sampling_idx_range)
            samplingGlobalAdj[idx] = newCurrentList
        else:
            samplingGlobalAdj[idx] = []
    samplingMatrix = getSamplingMatrix(samplingGlobalAdj, sampling_idx_range)
    return samplingMatrix

def loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum, saveName):
    saveName = saveName + '-TrainLabelIndex'
    trainsl = samplingTrainsetLabel.tolist()
    label = trainsl

    samplingTrainIndex111 = []
    # -----getting the train index from the file---------#
    # nd = np.genfromtxt(saveName, delimiter=',', skip_header=True)
    # samplingIndex = np.array(nd).astype(int)
    # for i in range(classNum):
    #     currentSamplingIndex = samplingIndex[:, i]
    #     samplingTrainIndex111 += currentSamplingIndex.tolist()

    # --------getting train index and save train index--------#
    samplingClassList = []
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(label) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass[k])
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingTrainIndex111 += samplingFixedIndex

    fileCol = {}
    for i in range(classNum):
        colName = 'TrainLabelIndex' + str(i)
        fileCol[colName] = samplingClassList[i]

    # dataframe = pd.DataFrame(fileCol)  # save the samplingIndex of every client
    # dataframe.to_csv(saveName, index=False, sep=',')  #

    return samplingTrainIndex111
#

#----------------------for federated average-----------------------#
def getSamplingIndex(nodesNum, samplingRate, testNodesNum, valNodesNum, trainLabel, labels):
    totalSampleNum = nodesNum - testNodesNum - valNodesNum  # get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  # get
    testAndValIndex = [i for i in range(totalSampleNum, nodesNum)]

    #-----analysis the label distributation result of test and val------#
    classNum = 6
    classDict = {}
    classDictIndex = {}
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        labelCount = len(labelIndex)
        classDict[k] = labelCount
        classDictIndex[k] = labelIndex

    lastSamplingNum = int(samplingNum / (classNum + 5))
    beforeSamplingNum = int((samplingNum - lastSamplingNum) / 5)
    finalSamplingNum = samplingNum - beforeSamplingNum * 5

    samplingNumList = [beforeSamplingNum] * 6
    samplingNumList[1] = finalSamplingNum

    samplingIndex = []
    for k in range(classNum):
        currentSamplingNum = samplingNumList[k]
        currentSamplingIdex = random.sample(classDictIndex[k], currentSamplingNum)
        samplingIndex += currentSamplingIdex

    return samplingIndex

def dataOverlapSplitting(samplingIndex, nodesNum, testNodesNum, valNodesNum, graph,
                  trainLabel, labels, sampleNumEachClass, classNum, saveName):
    train_sampling_idx_range = np.sort(samplingIndex)
    totalSampleNum = nodesNum - testNodesNum - valNodesNum
    sampling_idx_range = train_sampling_idx_range.tolist() + [i for i in range(totalSampleNum, nodesNum)]
    samplingLabels = labels[sampling_idx_range]
    trainLabel = trainLabel.numpy()
    samplingTrainsetLabel = trainLabel[train_sampling_idx_range]
    samplingTrainFixedIndex = loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum,
                                                 saveName)  # getting the sampling train data

    samplingMatrix = getSamplingGlobalAdj(graph, sampling_idx_range)  # get the global adj matrix with reduced node

    count = 0
    samplingAdj = collections.defaultdict(
        list)  # getting current sampling adj which is used for training is this client
    for index in sampling_idx_range:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, sampling_idx_range)
        samplingAdj[count] = newCurrentList
        count += 1

    return samplingAdj, samplingTrainFixedIndex, sampling_idx_range, samplingLabels, samplingMatrix


def dataSplitting(nodesNum, testNodesNum, valNodesNum, graph,
                  trainLabel, labels, samplingRate,
                  sampleNumEachClass, classNum, saveName):    #the val and test data keep and random sampling train data
    totalSampleNum = nodesNum - testNodesNum - valNodesNum  # get the train num according to the fixed testset and valset
    samplingNum = int(samplingRate * totalSampleNum)  # get
    labels = labels.numpy()
    trainLabel = trainLabel.numpy()

    #----getting sampling index from file----------------#
    # nd = np.genfromtxt(saveName + '-TrainIndex', delimiter=',', skip_header=True)
    # samplingIndex = np.array(nd).astype(int).tolist()

    #-----saving sampling index---------------------------#
    samplingIndex = random.sample(range(0, totalSampleNum), samplingNum)  # get random samplingIndex
    print('samplingIndex:', samplingIndex)
    # samplingIndex = getSamplingIndex(nodesNum, samplingRate, testNodesNum, valNodesNum, trainLabel, labels)
    # dataframe = pd.DataFrame({'samplingIndex': samplingIndex})  # save the train samplingIndex of every client
    # dataframe.to_csv(saveName + '-TrainIndex', index=False, sep=',')

    train_sampling_idx_range = np.sort(samplingIndex)  # sort the sampling index
    sampling_idx_range = train_sampling_idx_range.tolist() + [i for i in range(totalSampleNum, nodesNum)] #getting the whole graph node index
    samplingLabels = labels[sampling_idx_range]
    samplingTrainsetLabel = trainLabel[train_sampling_idx_range]  # the new graph train set label????

    samplingTrainFixedIndex = loadFixedTraindata(samplingTrainsetLabel, sampleNumEachClass, classNum, saveName)   #getting the sampling train data

    samplingMatrix = getSamplingGlobalAdj(graph, sampling_idx_range)  # get the global adj matrix with reduced node

    count = 0
    samplingAdj = collections.defaultdict(list)    #getting current sampling adj which is used for training is this client
    for index in sampling_idx_range:
        currentList = graph[index]
        newCurrentList = getSamplingAdj(currentList, sampling_idx_range)
        samplingAdj[count] = newCurrentList
        count += 1

    return samplingAdj, samplingNum, samplingTrainFixedIndex, sampling_idx_range, samplingLabels, samplingMatrix

def getTheCutSampling(samplingAdj, saveName):
    saveName = saveName + '-cut'

    #------get cut index from file----------------------#
    # nd = np.genfromtxt(saveName, delimiter=',', skip_header=True)
    # samplingCutIndexs = np.array(nd).astype(int).tolist()

    #-----------save cut index --------------------------#
    adjLen = len(samplingAdj)
    samplingNum = int(adjLen * 0.3)  # cut 30% nodes from train set  #BlogCatalog 0.5
    samplingCutIndexs = random.sample(range(0, adjLen), samplingNum)
    dataframe = pd.DataFrame({'samplingCutIndex': samplingCutIndexs})  # save the samplingIndex of every client
    dataframe.to_csv(saveName, index=False, sep=',')  #

    for cutIdx0 in samplingCutIndexs:
        cutRow0 = samplingAdj[cutIdx0]
        if (cutRow0 != []):
            cutIdx1 = cutRow0[len(cutRow0) - 1]
            cutPos0 = len(cutRow0) - 1
            cutRow0.pop(cutPos0)  # remove the last element

            cutRow1 = samplingAdj[cutIdx1]
            if cutRow1 != []:
                if cutIdx0 in cutRow1:
                    cutPos1 = cutRow1.index(cutIdx0)
                    cutRow1.pop(cutPos1)

            samplingAdj[cutIdx0] = cutRow0
            samplingAdj[cutIdx1] = cutRow1

    return samplingAdj

def get_graph1(data):
    currentGraph = {}
    nodesNum = data["num_vertices"]
    for i in range(nodesNum):
        currentGraph[i] = []

    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        currentGraph[currentNIdx].append(edge[1])

    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = currentGraph[i]
    return graph

def get_graph(data):
    nodesNum = data["num_vertices"]
    graphList = []
    nodeIdx = 0
    row = []
    for edge in data["edge_list"]:
        currentNIdx = edge[0]
        if currentNIdx != nodeIdx:
            nodeIdx = currentNIdx
            graphList.append(row)
            row = []
        row.append(edge[1])
    graphList.append(row)
    graph = collections.defaultdict(list)
    for i in range(nodesNum):
        graph[i] = graphList[i]
    return graph

def get_edge_list(samplingAdj):
    nodesNum = len(samplingAdj)
    edge_list = []
    for i in range(nodesNum):
        rowNodes = samplingAdj[i]
        for j in rowNodes:
            node = tuple([i, j])
            edge_list.append(node)
    return edge_list

def getOverlapClientData(data, samplingIndex, sampleNumEachClass, classNum, saveName):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    idx_train = range(0, trainNum)
    trainLabel = lbl[idx_train]

    samplingAdj, samplingTrainFixedIndex, sampling_idx_range, \
    samplingLabels, samplingMatrix = dataOverlapSplitting(samplingIndex, nodesNums, testNodesNum, valNodesNum, graph,
                  trainLabel, lbl, sampleNumEachClass, classNum, saveName)

    samplingAdj = getTheCutSampling(samplingAdj, saveName)

    X = X[sampling_idx_range, :]
    edge_list = get_edge_list(samplingAdj)

    samplingAdjNum = len(samplingAdj)
    idx_test = range(samplingAdjNum - testNodesNum, samplingAdjNum)  # get the last 2800 indexes as test set
    idx_val = range(samplingAdjNum - testNodesNum - valNodesNum,
                    samplingAdjNum - testNodesNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_train = samplingTrainFixedIndex  # samplingTrainFixedIndex  # get 1400 indexes as val set
    # range(0, samplingAdjNum - testNodesNum - valNodesNum)

    train_mask = sample_mask(idx_train, samplingAdjNum)
    val_mask = sample_mask(idx_val, samplingAdjNum)
    test_mask = sample_mask(idx_test, samplingAdjNum)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": samplingAdjNum,
                "num_edges": data["num_edges"],
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": torch.tensor(samplingLabels),
                "sampling_idx_range": sampling_idx_range}

    return dataDict, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum

def getCommunitData(data, communities, node_subjects, clientNum):
    graph = get_graph1(data)

    dataList = []
    train_mask_list = []
    val_mask_list = []
    test_mask_list = []
    test_idx_list = []
    val_idx_list = []
    train_list = []
    for i in range(clientNum):
        communitie = list(communities[i])
        communitieLen = len(communitie)
        count = 0
        samplingAdj = collections.defaultdict(
            list)  # getting current sampling adj which is used for training is this client
        for index in communitie:
            currentList = graph[index]
            newCurrentList = getSamplingAdj(currentList, communitie)
            samplingAdj[count] = newCurrentList
            count += 1

        print('clientId:', {i})
        print(len(communitie))
        print(communitie)

        X = data["features"][communitie, :]
        edge_list = get_edge_list(samplingAdj)
        lbls = data["labels"][communitie]

        samplingAdjNum = len(samplingAdj)
        sub_node_subjects = node_subjects[communitie]

        trainIdx, testIdx = model_selection.train_test_split(
            sub_node_subjects, train_size=0.5, test_size=0.2, stratify=sub_node_subjects
        )
        trainIdx = list(trainIdx.index.values)
        testIdx = list(testIdx.index.values)
        valIdx = [idx for idx in communitie if idx not in trainIdx and idx not in testIdx]

        idx_train = []
        idx_test = []
        idx_val = []
        for idx, val in enumerate(communitie):
            if val in trainIdx:
                idx_train.append(idx)
            elif val in testIdx:
                idx_test.append(idx)
            else:
                idx_val.append(idx)

        train_mask = sample_mask(idx_train, samplingAdjNum)
        val_mask = sample_mask(idx_val, samplingAdjNum)
        test_mask = sample_mask(idx_test, samplingAdjNum)

        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)

        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask)
        test_mask_list.append(test_mask)

        dataDict = {"num_classes": data["num_classes"],
                    "num_vertices": communitieLen,
                    "num_edges": len(edge_list),
                    "dim_features": X.shape[1],
                    "features": X,
                    "edge_list": edge_list,
                    "labels": lbls,
                    "sampling_idx_range": communitie}

        dataList.append(dataDict)
        train_list.append(trainIdx)
        test_idx_list.append(testIdx)
        val_idx_list.append(valIdx)

    return dataList, train_mask_list, val_mask_list, test_mask_list, test_idx_list, train_list, val_idx_list




def getClientData(data, samplingRate, sampleNumEachClass, classNum, saveName):
    X, lbl = data["features"], data["labels"]

    graph = get_graph1(data)

    nodesNums = data["num_vertices"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    idx_train = range(0, trainNum)
    trainLabel = lbl[idx_train]

    samplingAdj, samplingNum, samplingTrainFixedIndex,\
    sampling_idx_range, samplingLabels, samplingMatrix = dataSplitting(nodesNums, testNodesNum, valNodesNum, graph,
                  trainLabel, lbl, samplingRate,
                  sampleNumEachClass, classNum, saveName)

    # samplingAdj = getTheCutSampling(samplingAdj, saveName)

    X = X[sampling_idx_range, :]
    edge_list = get_edge_list(samplingAdj)

    samplingAdjNum = len(samplingAdj)
    idx_test = range(samplingAdjNum - testNodesNum, samplingAdjNum)  # get the last 2800 indexes as test set
    idx_val = range(samplingAdjNum - testNodesNum - valNodesNum,
                    samplingAdjNum - testNodesNum)  # sampling 420 indexes as train set, each class has 60 labels
    idx_train = samplingTrainFixedIndex #samplingTrainFixedIndex  # get 1400 indexes as val set
    #range(0, samplingAdjNum - testNodesNum - valNodesNum)

    train_mask = sample_mask(idx_train, samplingAdjNum)
    val_mask = sample_mask(idx_val, samplingAdjNum)
    test_mask = sample_mask(idx_test, samplingAdjNum)

    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": samplingAdjNum,
                "num_edges": data["num_edges"],
                "dim_features": X.shape[1],
                "features": X,
                "edge_list": edge_list,
                "labels": torch.tensor(samplingLabels),
                "sampling_idx_range": sampling_idx_range}

    return dataDict, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum

def loadFoursquare(data_str):
    edgesfilePath = data_str + '/following.npy'
    userlabelPath = data_str + '/multilabel2id.pkl'
    userFeaturesPath = data_str + '/userattr.npy'
    user = data_str + '/user'

    features = np.load(userFeaturesPath)
    features = np.float32(features)

    edges = np.load(edgesfilePath)

    f = open(userlabelPath, 'rb')
    labels = pickle.load(f)

    users = []
    for userNode in range(len(features)):
        if userNode in labels.keys():
            users.append(userNode)

    currentEdges = []
    for edge in edges:
        startNode = edge[0]
        endNode = edge[1]
        if startNode in labels.keys() and endNode in labels.keys():
            currentEdges.append(tuple([users.index(startNode), users.index(endNode)]))

    userlabels = []
    for userid in range(len(users)):
        userlabels.append(sum(labels[users[userid]])-1)

    currentFeatures = features[users]
    currentNodeNums = len(currentFeatures)
    current_num_edges = len(currentEdges)

    dataDict = {"num_classes": 9,
                "num_vertices": currentNodeNums,
                "num_edges": current_num_edges,
                "dim_features": features.shape[1],
                "features": torch.tensor(currentFeatures),
                "edge_list": currentEdges,
                "labels": torch.tensor(userlabels)}

    return dataDict

def getSamplingData(data):
    globalLabelStatistic = {}
    globalLabelIndex = {}
    classNum = 6  # flickr_9 facebook_4 BlogCatalog_6
    labels = data["labels"].numpy().tolist()
    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(labels) if x == k]
        labelCount = len(labelIndex)
        globalLabelStatistic[k] = labelCount
        globalLabelIndex[k] = labelIndex

    #sampling 300 from class 1 and construct new edges__Blogcatalog
    samplingNum = 300
    samplingClassIndex = random.sample(globalLabelIndex[1], samplingNum)
    newSamplingNode = []
    newSamplingLabels = []
    newSamplingFeatures = []
    labels = data["labels"].numpy()
    for k in range(classNum):
        if k == 1:
            newSamplingNode += samplingClassIndex
            newSamplingLabels += list(labels[samplingClassIndex])
            newSamplingFeatures += data["features"][samplingClassIndex]
        else:
            newSamplingNode += globalLabelIndex[k]
            newSamplingLabels += list(labels[globalLabelIndex[k]])
            newSamplingFeatures += data["features"][globalLabelIndex[k]]

    currentGlobalNode = []
    for i in range(len(labels)):
        if i in newSamplingNode:
            currentGlobalNode.append(i)
    currentGlobalLabels = labels[currentGlobalNode]

    newSamplingEdges = []
    edges = data["edge_list"]
    for edge in edges:
        start = edge[0]
        end = edge[1]
        if start in currentGlobalNode and end in currentGlobalNode:
            newStartIndex = currentGlobalNode.index(start)
            newEndIndex = currentGlobalNode.index(end)
            newSamplingEdges.append(tuple([newStartIndex, newEndIndex]))
    np.save('BlogCatalog/newSampling_edge_list.npy', newSamplingEdges)
    np.save('BlogCatalog/newSamplingNode.npy', currentGlobalNode)
    np.save('BlogCatalog/newSamplingLabels.npy', currentGlobalLabels)
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    dataDict = {"num_classes": classNum,
                "num_vertices": len(newSamplingNode),
                "num_edges": len(newSamplingEdges),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": newSamplingEdges,
                "labels": newSamplingLabels}

    return dataDict

def loadSamplingData(data):
    classNum = 6
    edges = np.load('BlogCatalog/newSampling_edge_list.npy')
    samplingNode = np.load('BlogCatalog/newSamplingNode.npy')
    samplingLables = np.load('BlogCatalog/newSamplingLabels.npy')
    # np.save('BlogCatalog/newSamplingFeatures.npy', newSamplingFeatures)

    newSamplingFeatures = data["features"][samplingNode]

    # edges_list = [tuple([edge[0], edge[1]]) for edge in edges]

    dataDict = {"num_classes": classNum,
                "num_vertices": len(samplingNode),
                "num_edges": len(samplingLables),
                "dim_features": data["dim_features"],  # data["dim_features"], #facebook_feature_dim,
                "features": newSamplingFeatures,
                "edge_list": edges,
                "labels": torch.tensor(samplingLables)}

    return dataDict

def getCutedGradient(grad, clip):
    gradShape = np.array(grad).shape
    norm2 = np.linalg.norm(grad, ord=2, axis=1, keepdims=True)
    norm2 = norm2 / clip
    cutedGrad = []
    for i in range(gradShape[0]):
        currentNorm = norm2[i]

        if currentNorm > 1:
            currentGrad = grad[i] / norm2[i]
        else:
            currentGrad = grad[i]

        cutedGrad.append(np.array(currentGrad).tolist())

    return cutedGrad

def getNoise(sigma, sensitivity, shape):  # what the mean of batchsize?

    noise = torch.normal(0, sigma * sensitivity, size=shape)#

    return noise

def pca_project(param, sigma, sensitivity, clip):
    param = param.detach().numpy()
    mean = np.average(param, axis=0)
    pca = decomposition.PCA(n_components=3)   #n_components=0.99
    pca.fit(param)   #training
    X = pca.transform(param)   #return the result of dimensionality reduction

    shape = np.array(X).shape
    noise = torch.normal(0, sigma * sensitivity, size=shape)
    XN = X + np.array(noise)                               #perturb the result

    a0 = np.matrix(X)
    b0 = np.matrix(pca.components_)                     #inverse matrix
    inverseMatrix0 = a0 * b0 + mean                     #get the inverse transformation
    redisual0 = param - inverseMatrix0

    a1 = np.matrix(XN)
    b1 = np.matrix(pca.components_)                 # inverse matrix
    inverseMatrix1 = a1 * b1 + mean                 # get the inverse transformation
    theFinalNM = inverseMatrix1 + redisual0
    theFinalNM = np.array(theFinalNM)
    return theFinalNM

def getSigma(eps, delta, sensitivity):  #Gaussian mechanism

    sigma = math.sqrt((2*(sensitivity**2)*math.log(1.25/delta))/(eps**2))

    return sigma

def getProjectedNoiseGrad(grad, clip, eps, delta, sensitivity):
    cutedGrad = getCutedGradient(grad, clip)
    sigma = getSigma(eps, delta, sensitivity)
    noisedGrad = pca_project(cutedGrad, sigma, clip)
    return noisedGrad

def getGaussianSigma(eps, delta):
    sigma = math.sqrt((2 * math.log(1.25 / delta)) / (eps ** 2))

    return sigma

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size

def scale(input, clipVal):
    norms = torch.norm(input, dim=1)

    scale = torch.clamp(clipVal / norms, max=1.0)

    pp = input * scale.view(-1, 1)

    return pp


# def getHistogramSplit(param, quentile): ###getting the index of different region
#     size = param.shape
#     param = param.detach().numpy()
#
#     max = np.amax(param)  # get the max val of each column
#     min = np.amin(param)
#     interval = (max - min) / quentile
#     his = []
#     hisC = []
#     for k in range(quentile):
#         start = min + interval * k
#         end = min + interval * (k + 1)
#         curHis = []
#         for i in range(size[0]):
#             for j in range(size[1]):
#                 aa = param[i][j]
#                 if aa >= start and aa < end:
#                     curHis.append([i, j])
#         his.append(curHis)
#         hisC.append(len(curHis))
#
#     return his, hisC

def getHistogramSplit(grad):
    grad = np.array(grad)
    size = grad.shape

    max = np.amax(grad, axis=0)  # get the max val of each column
    min = np.amin(grad, axis=0)
    splitNum = 5
    interval = (max - min) / splitNum
    splitCount = []

    for i in range(size[1]):
        eachColIntervalCount = []
        for j in range(splitNum):
            curCount = 0
            for k in range(size[0]):
                start = min[i] + interval[i] * j
                end = min[i] + interval[i] * (j + 1)
                if grad[k][i] >= start and grad[k][i] < end:
                    curCount += 1
            eachColIntervalCount.append(curCount)
        splitCount.append(eachColIntervalCount)

    splitCount = np.array(splitCount)
    maxSplit = np.amax(splitCount, axis=1)
    index = []
    for i in range(len(maxSplit)):
        currentIndex = splitCount[i].tolist().index(maxSplit[i])
        index.append(currentIndex)

    return max, min, interval, index, splitCount

def getHistogramNoiseGradient(sigma, grad, min, interval, index, splitCount, sensitivity):
    grad = np.array(grad)
    size = grad.shape


    for i in range(size[1]):
        start = min[i] + interval[i] * index[i]
        end = min[i] + interval[i] * (index[i] + 1)
        # sensitivity = end - start
        maxCount = splitCount[i][index[i]]
        noise = torch.normal(0, sigma * sensitivity/2708, size=[maxCount, 1])#
        count = 0
        for j in range(size[0]):
            if grad[j][i] >= start and grad[j][i] < end:
                grad[j][i] += noise[count]
                count += 1

    return grad

def partial_noise(grad, importPos, eps, delta, sensitivity):
    grad = np.array(grad)
    size = grad.shape
    sigma = getGaussianSigma(eps / 200, delta)

    maxCount = np.sum(importPos)

    noise = torch.normal(0, sigma * sensitivity / 2708, size=[maxCount, 1])  #
    count = 0

    for i in range(size[0]):
        for j in range(size[1]):
            if importPos[i][j] == 1:
                grad[i][j] += noise[count]
                count += 1

    return grad

def all_noise(grad, eps, delta, sensitivity):
    sigma = getGaussianSigma(eps / 200, delta)
    size = grad.shape

    for i in range(size[1]):

        noise = torch.normal(0, sigma * sensitivity / 2708, size=[size[0], 1])  #

        for j in range(size[0]):
            grad[j][i] += noise[j]

    return grad

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break
    return cur_sigma, previous_eps

def get_sigma_(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma

    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps

def getSigmaAndEps(batchsize, n_training, n_epoch, eps, delta, rgp):
    sampling_prob = batchsize / n_training
    steps = int(n_epoch / sampling_prob)
    sigma, eps = get_sigma_(sampling_prob, steps, eps, delta, rgp=rgp)
    noise_multiplier0 = noise_multiplier1 = sigma
    print('noise scale for gradient embedding: ', noise_multiplier0, 'noise scale for residual gradient: ',
          noise_multiplier1, '\n rgp enabled: ', rgp, 'privacy guarantee: ', eps)
    return noise_multiplier0, eps

def get_noise():
    # steps = args.n_epoch
    # sampling_prob = batchsize / n_training
    # steps = int(n_epoch / sampling_prob)
    sampling_prob = 1
    steps = 200
    eps = 8
    delta = 1e-5
    sigma, eps = get_sigma_(sampling_prob, steps, eps, delta, rgp=True)
    clip0 = 1
    batchsize = 50
    noise_multiplier0 = noise_multiplier1 = sigma
    theta_noise = torch.normal(0, noise_multiplier0 * clip0 / batchsize, size=[100, 100],
                               device="cpu")

def getGlobalAdjMatrix(clientList):
    globalMatrix = 0
    for client in clientList:
        globalMatrix += client.samplingMatrix
    return globalMatrix

def getOverlapNodes1(clientList, clientNum):
    overlapNodes = []
    for i in range(clientNum):
        for j in range(i + 1, clientNum):
            samplingIndex_i = clientList[i]
            samplingIndex_j = clientList[j]
            for ind in samplingIndex_i:
                if ind in samplingIndex_j and ind not in overlapNodes:
                    overlapNodes.append(ind)
    return overlapNodes


def getOverlapNodes(globalMatrix):
    overlapNodes = []
    globalMatrixShape = globalMatrix.shape
    for idx in range(globalMatrixShape[0]):
        eachRow = globalMatrix[idx]
        rowIndexs = [i for i, x in enumerate(eachRow) if x != 1]

        for rowIdx in rowIndexs:
            if rowIdx not in overlapNodes:
                overlapNodes.append(rowIdx)
    return overlapNodes

def getGlobalOverlapNodesEmd(clientList, overlapNodes, clientOuts):
    globalNodeEmbeddings = []
    for idx in overlapNodes:
        mean = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmb = clientOut[nodeIndex]
                # mean += currentNodeEmb
                mean += currentNodeEmb.detach().numpy()
                count += 1
        if count == 0:count = 1
        mean = mean / count
        # mean = mean.detach().numpy()
        expDis = []
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                currentNodeEmd = currentNodeEmd.detach().numpy()
                dist = np.linalg.norm(currentNodeEmd - mean)
                # dist = torch.norm(currentNodeEmd - mean)
                try:
                    expDis.append(math.exp(dist))
                except OverflowError:
                    expDis.append(math.exp(700))
                # print('dist:', dist)
                # print('math.exp(dist):', cmath.exp(dist))
                # expDis.append(math.exp(dist))

        finalNodeEmb = 0
        count = 0
        for i in range(clientNum):
            clientOut = clientOuts[i]
            net = clientList[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                currentNodeEmd = clientOut[nodeIndex]
                currentNodeEmd = currentNodeEmd.detach().numpy()
                finalNodeEmb += (expDis[count] / sum(expDis)) * currentNodeEmd
        globalNodeEmbeddings.append(finalNodeEmb)
    return globalNodeEmbeddings

def setGlobalNodeEmdForLocalNodes(clientList, clientOuts, overlapNodes, globalNodeEmbeddings):      ##设置不同客户端的重叠节点的增量节点
    for idx in overlapNodes:
        for i in range(clientNum):
            net = clientList[i]
            # clientOut = clientOuts[i]
            sampling_idx_range = net.data['sampling_idx_range']
            if idx in sampling_idx_range:
                nodeIndex = sampling_idx_range.index(idx)
                # nodeIndexs[clientNum].append(nodeIndex)
                clientOuts[i][nodeIndex].data = torch.Tensor(globalNodeEmbeddings[idx])
    return clientOuts

def getLeftSamplingNodes(totalSampleNum, samplingOverlappedNodesIndex):
    totalIndex = list(range(0, totalSampleNum))
    totalCopyIndex = deepcopy(totalIndex)

    for currentIndex in samplingOverlappedNodesIndex:
        totalIndex.remove(currentIndex)

    return totalIndex, totalCopyIndex

def random_dataset_overlap(data, clientNum, overlapRate):   #
    #5%10%15%
    nodesNums = data["num_vertices"]
    classNum = data["num_classes"]
    testNodesNum = int(nodesNums * 0.4)
    valNodesNum = int(nodesNums * 0.2)
    trainNum = nodesNums - testNodesNum - valNodesNum
    eachSamplingNum = int(trainNum / clientNum)
    overlapNodes = int(eachSamplingNum * overlapRate)
    everySplitLen = eachSamplingNum - overlapNodes

    labels = data['labels']
    idx_train = range(0, trainNum)
    trainLabel = labels[idx_train]
    samplingClassList = []
    samplingOverlappedNodesIndex = []
    # sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 2, 2]  #flickr_5%
    # sampleNumEachClass = [3, 3, 3, 3, 3, 2]  #Blogcatalog_5%
    # sampleNumEachClass = [6, 6, 6, 6, 6, 4]  # Blogcatalog_10%
    sampleNumEachClass = [9, 9, 9, 9, 9, 6]  # Blogcatalog_15%

    # sampleNumEachClass = [3, 3, 3, 3, 3, 3, 3, 3, 1]  #Flickr_5%
    # sampleNumEachClass = [6, 6, 6, 6, 6, 6, 6, 6, 2]  #Flickr_10%
    # sampleNumEachClass = [9, 9, 9, 9, 9, 9, 9, 9, 3]  #Flickr_15%

    # sampleNumEachClass = [19, 19, 19, 17]  #Facebook_5%
    # sampleNumEachClass = [38, 38, 38, 35]  # Facebook_10%
    # sampleNumEachClass = [56, 56, 56, 56]  #Facebook_15%

    for k in range(classNum):
        labelIndex = [i for i, x in enumerate(trainLabel) if x == k]
        samplingIndex = random.sample(range(0, len(labelIndex)), sampleNumEachClass[k])
        samplingFixedIndex = np.array(labelIndex)[samplingIndex].tolist()
        samplingClassList.append(samplingFixedIndex)
        samplingOverlappedNodesIndex += samplingFixedIndex

    currentTotalIndex, orginalTotalIndex = getLeftSamplingNodes(trainNum, samplingOverlappedNodesIndex)

    splitNodes = []
    for i in range(clientNum):
        splitIndex = random.sample(currentTotalIndex, everySplitLen)
        splitNodes.append(splitIndex + samplingOverlappedNodesIndex)
        for currentIndex in splitIndex:
            currentTotalIndex.remove(currentIndex)

    return splitNodes

def getClusterAdj(localOuts, clientList, n_clusters):
    centers = localOuts[0][0][np.random.choice(len(localOuts[0][0]), 10, replace=False)]
    for i in range(1, clientNum):
        curCenter = localOuts[i][0][np.random.choice(len(localOuts[i][0]), 10, replace=False)]
        centers = torch.cat([centers, curCenter], dim=0)

    for i in range(clientNum):
        clientList[i].kmeans.centers = centers.detach().numpy()

    for i in range(5):
        finalCenters = 0
        labels = []
        for j in range(clientNum):
            curLabels, curCenters = clientList[j].getIterKmeansCenterAndLabels(
                localOuts[j][0].detach().numpy())
            finalCenters += curCenters
            labels.append(curLabels)
        finalCenters = finalCenters / clientNum

    for j in range(clientNum):
        clientList[j].kmeans.centers = finalCenters

        # ----get global adj---------#
        clientIndex = []
        for i in range(clientNum):
            index = []
            row = []
            col = []
            for j in range(n_clusters):
                clusterIndex = [idx for idx, lbl in enumerate(labels[i]) if lbl == j]
                for idx0 in clusterIndex:
                    ind = [[idx0, idx1] for idx1 in clusterIndex]
                    index += ind
            index = np.array(index)
            row = index[:, 0]
            col = index[:, 1]
            val = [1] * len(index)
            adj_csr = csr_matrix((val, (row, col)), shape=(len(labels[i]), len(labels[i]))).toarray()
            clientList[i].globalAdj_clu = torch.Tensor(adj_csr)

class KMeans:
    def __init__(self, n_clusters, centers=[], max_iter=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = centers

    def fit(self, X):
        # 随机初始化簇中心
        for i in range(self.max_iter):
            # 计算每个样本到簇中心的距离
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2)

            # 分配样本到最近的簇
            labels = np.argmin(distances, axis=1)

            # 更新簇中心
            for j in range(self.n_clusters):
                if sum(labels == j) != 0:
                   self.centers[j] = np.mean(X[labels == j], axis=0)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

klayer = 3
class GlobalGCN(nn.Module):
    def __init__(self, data: dict,
                 test_idx: list):
        super().__init__()
        self.data = data
        self.test_idx = test_idx
        hid_channels = 16

        self.globalModel = GCNS(data["dim_features"], hid_channels, data["num_classes"], 0)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.testNodesNum = int(self.data["num_vertices"] * 0.4)
        self.test_mask = self.getTestMask()
        self.acc = []
        self.f1 = []
        self.pre = []
        self.recall = []

    def getTestMask(self):
        idx_test = []
        for i in range(len(self.test_idx)):
            idx_test += self.test_idx[i]
        # idx_test = range(self.data["num_vertices"] - self.testNodesNum, self.data["num_vertices"])
        test_mask = sample_mask(idx_test, self.data["num_vertices"])
        test_mask = torch.tensor(test_mask)
        return test_mask

    def test(self, globalParam, epoch, iteration):
        # test
        print(f"global--test...")
        for pid, param in enumerate(list(self.globalModel.parameters())):
            param.data = torch.tensor(globalParam[pid+3], dtype=torch.float32)

        res, f0, recall0, precision0 = infer(self.globalModel, self.X, self.A.L_GCN.to_dense(), self.lbls, self.test_mask, test=True)
        self.acc.append(res['accuracy'])
        self.f1.append(f0)
        self.recall.append(recall0)
        self.pre.append(precision0)

        if epoch == iteration - 1:
            acc_avg = sum(self.acc[iteration - 1 - 10:iteration - 1])
            f1_avg = sum(self.f1[iteration - 1 - 10:iteration - 1])
            recall_avg = sum(self.recall[iteration - 1 - 10:iteration - 1])
            pre_avg = sum(self.pre[iteration - 1 - 10:iteration - 1])
            print('avg acc:', acc_avg / 10, 'avg f2:', f1_avg / 10, 'avg recall:', recall_avg / 10, 'avg pre:',
                  pre_avg / 10)

            idx_test = []
            for i in range(len(self.test_idx)):
                idx_test += self.test_idx[i]

            self.globalModel.eval()
            outs, outlist = self.globalModel(self.X, self.A.L_GCN.to_dense())
            tsne = TSNE(n_components=2)
            x_tsne = tsne.fit_transform(outlist[klayer-1].detach().numpy())
            fig = plt.figure()
            preLabel = np.argmax(outs.detach().numpy(), axis=1)

            dataframe = pd.DataFrame(
                {'x0': x_tsne[:, 0], 'x1': x_tsne[:, 1],
                 'c': preLabel})  # save data
            dataframe.to_csv('cora/Community3/cora_fedla_intra_0.5_inter_0.1_scatter.csv', index=False, sep=',')

            plt.scatter(x_tsne[:, 0][idx_test], x_tsne[:, 1][idx_test], c=preLabel[idx_test], label="t-SNE")
            fig.savefig('cora/Community3/cora_fedla_intra_0.5_inter_0.1_scatter.png')
            plt.show()

        print(res)

class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) #参数化
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        return attention, e


    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class ClientGCN(nn.Module): #
    def __init__(self, data: dict,
                       train_idx: torch.Tensor,
                       test_mask: torch.Tensor,
                       val_mask: torch.Tensor,
                       trainNum: int
                       ):
        super().__init__()
        self.data = data
        hid_channels = 16
        project_channels = 300

        self.localSModel = GCNS(data["dim_features"], hid_channels, data["num_classes"])
        self.attentions = GraphAttention(data["dim_features"], hid_channels, alpha=0.2)

        self.layer0Centers = torch.nn.Parameter(torch.randn(data["num_classes"], hid_channels))
        self.layer1Centers = torch.nn.Parameter(torch.randn(data["num_classes"], hid_channels))
        self.layer2Centers = torch.nn.Parameter(torch.randn(data["num_classes"], hid_channels))

        self.act = nn.Sigmoid()
        self.tc = 1

        self.alpha = 0.5 #0.05
        self.temp = 0.5

        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train_idx = train_idx
        self.X, self.lbls = self.data["features"], self.data["labels"]
        self.A = Graph(self.data["num_vertices"], self.data["edge_list"])
        self.augAdj = self.A.L_GCN.to_dense()   #get the original adjacency
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.best_state = None
        self.best_epoch = 0
        self.best_val = 0
        self.trainNum = trainNum
        self.valAccuracy = []


    def getLocalTrainOut(self):
        self.train()
        self.st = time.time()
        self.optimizer.zero_grad()  # 清空过往的梯度

        alignedX = self.X

        self.augAdj, self.adj_logits = self.getAttAdj()
        out, outslist = self.localSModel(alignedX, self.augAdj)#self.A.L_GCN.to_dense()
        return out, outslist

    def getAttAdj(self):
        attAdj, adj_logits = self.attentions(self.data["features"], self.A.L_GCN.to_dense())
        adj_orig = self.A.L_GCN.to_dense()
        adj_orig[adj_orig <= 0] = 0
        adj_orig[adj_orig > 0] = 1
        edge_probs = self.alpha * attAdj + (1 - self.alpha) * adj_orig

        # Gumbel-Softmax Sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp,
                                                                         probs=edge_probs).rsample()
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T

        adj_sampled.fill_diagonal_(1)
        D_norm = torch.diag(torch.pow(adj_sampled.sum(1), -0.5))
        adj_sampled = D_norm @ adj_sampled @ D_norm

        return adj_sampled, adj_logits

    def trainNet(self, localOut, localOutlist, clientId, epoch):
        outl, lbls = localOut[self.train_idx], self.lbls[self.train_idx]

        layer0CenterLoss = self.getLayerCenterLoss(self.layer0Centers, localOutlist[0][self.train_idx], lbls, 0.5, 0.1)#0.05,0.1

        layer1CenterLoss = self.getLayerCenterLoss(self.layer1Centers, localOutlist[1][self.train_idx], lbls, 0.5, 0.1)#0.05,0.1

        layer2CenterLoss = self.getLayerCenterLoss(self.layer2Centers, localOutlist[2][self.train_idx], lbls, 0.5, 0.1)#0.5,0.5

        a = 0.0001
        b = 0.0001
        c = 0.01

        adj_orig = self.A.L_GCN.to_dense()
        adj_orig[adj_orig <= 0] = 0
        adj_orig[adj_orig > 0] = 1
        norm_w = self.data["num_vertices"] ** 2 / float((self.data["num_vertices"] ** 2 - adj_orig.sum()) * 2)
        pos_weight = torch.FloatTensor([float(self.data["num_vertices"] ** 2 - adj_orig.sum()) / adj_orig.sum()])
        genAdj_loss = norm_w * F.binary_cross_entropy_with_logits(self.adj_logits, adj_orig, pos_weight=pos_weight)

        class_loss0 = F.cross_entropy(outl, lbls)

        self.loss = class_loss0 + layer0CenterLoss + layer1CenterLoss + \
                    layer2CenterLoss + c * genAdj_loss

        self.loss.backward()  # 计算梯度
        self.optimizer.step()  # 更新梯度
        print(f"clientId:{clientId}, Epoch: {epoch}, Time: {time.time() - self.st:.5f}s, Loss: {self.loss.item():.5f}")
        return self.loss.item()


    def getLayerCenterLoss(self, layCenter, x, label, a0, a1):
        currentCenter = layCenter[label]

        classLoss = 0

        samplesLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        samplesLoss = samplesLoss(x, currentCenter)

        for i in range(data["num_classes"]):
            eachCenter = layCenter[i]
            curCenters = eachCenter.repeat([data["num_classes"] - 1, 1])
            curClassLoss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            itemCount = 0
            for j in range(data["num_classes"]):
                if i != j:
                    if itemCount == 0:
                        otherCenters = torch.unsqueeze(layCenter[j], dim=0)
                    else:
                        curC = torch.unsqueeze(layCenter[j], dim=0)
                        otherCenters = torch.cat((otherCenters, curC), dim=0)
                    itemCount += 1
            curClassLoss = curClassLoss(curCenters, otherCenters)
            classLoss += curClassLoss
        classLoss = classLoss / data["num_classes"]

        centerLoss = a0 * samplesLoss + a1 * classLoss

        return centerLoss


    def val(self, clientId, epoch):
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(self.localSModel, self.X, self.augAdj, self.lbls, self.val_mask)
                self.valAccuracy.append(val_res)
            if val_res > self.best_val:
                print(f"clientId:{clientId}, update best: {val_res:.5f}")
                self.best_epoch = epoch
                self.best_val = val_res
                self.best_state = deepcopy(self.localSModel.state_dict())
            return val_res

    def test(self, clientId):
        print(f"clientId:{clientId}, best val: {self.best_val:.5f}")
        # test
        print(f"clientId:{clientId}, test...")
        self.localSModel.load_state_dict(self.best_state)
        res = infer(self.localSModel, self.X, self.augAdj, self.lbls, self.test_mask, test=True)
        print(f"clientId:{clientId}, final result: epoch: {self.best_epoch}")
        print(res)

    def saveFinalEmbedding(self, saveName):
        self.localSModel.eval()
        outs, outlist = self.localSModel(self.X, self.augAdj)
        outs = outs.detach().numpy()
        # valAcc = np.array(self.valAccuracy)
        # #------save  embedding---------------#
        # valFileOuts = saveName + "-valAcc.npy"
        # np.save(valFileOuts, valAcc)
        fileOuts = saveName + "-outs.npy"
        np.save(fileOuts, outs)
        #------save sampling index-----------#
        # fileIndex = saveName + "-samplingIndex.npy"
        # np.save(fileIndex, self.data["sampling_idx_range"])
        print(saveName + "  save success")



def load_graph(edges):
    G = collections.defaultdict(dict)
    for edge in edges:
        w = 1.0  # 数据集有权重的话则读取数据集中的权重
        G[edge[0]][edge[1]] = w
    return G

def get_global_graph(all_edges, all_features, nodes_id):
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in all_edges]
    df['target'] = [edge[1] for edge in all_edges]

    nodes = sg.IndexedArray(all_features, nodes_id)
    G = StellarGraph(nodes=nodes, edges=df)

    return G

def louvain_graph_cut(whole_graph: StellarGraph, node_subjects, num_owners):   #对数据的处理
    delta = 70
    edges = np.copy(whole_graph.edges())
    df = pd.DataFrame()
    df['source'] = [edge[0] for edge in edges]
    df['target'] = [edge[1] for edge in edges]
    G = StellarGraph.to_networkx(whole_graph)

    partition = community_louvain.best_partition(G)

    groups = []

    for key in partition.keys():   #这个是找到有多少组
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():  #初始化set，将相应的节点对应到不同的组中
        partition_groups[partition[key]].append(key)

    group_len_max = len(list(whole_graph.nodes()))//num_owners-delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups)+1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1],reverse=True)}

    owner_node_ids = {owner_id: [] for owner_id in range(num_owners)}

    owner_nodes_len = len(list(G.nodes()))//num_owners
    owner_list = [i for i in range(num_owners)]
    owner_ind = 0


    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) >= owner_nodes_len + delta:
            owner_ind = (owner_ind + 1) % len(owner_list)
            if owner_ind + 1 == len(owner_list):
               break
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    for owner_i in owner_node_ids.keys():
        print('nodes len for ' + str(owner_i) + ' = '+str(len(owner_node_ids[owner_i])))

    subj_set = list(set(node_subjects.values))  #类别
    local_node_subj_0 = []
    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = whole_graph.node_ids_to_ilocs(partition_i)
        sbj_i = node_subjects.copy(deep=True)
        sbj_i.values[:] = "" if node_subjects.values[0].__class__ == str else 0
        sbj_i.values[locs_i] = node_subjects.values[locs_i]
        local_node_subj_0.append(sbj_i)
    count = []
    for owner_i in range(num_owners):
        count_i = {k: [] for k in subj_set}
        sbj_i = local_node_subj_0[owner_i]
        for i in sbj_i.index:
            if sbj_i[i] != 0 and sbj_i[i] != "":
                count_i[sbj_i[i]].append(i)
        count.append(count_i)
    for k in subj_set:
        for owner_i in range(num_owners):
            if len(count[owner_i][k]) < 2:
                for j in range(num_owners):
                    if len(count[j][k]) > 2:
                        id = count[j][k][-1]
                        count[j][k].remove(id)
                        count[owner_i][k].append(id)
                        owner_node_ids[owner_i].append(id)
                        owner_node_ids[j].remove(id)
                        j = num_owners

    return owner_node_ids



if __name__ == "__main__":
    set_seed(2022)# set_seed(2023) #BlogCatalog #Flickr
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    facebook_feature_dim = 4714
    dataDict = {"num_classes": data["num_classes"],
                "num_vertices": data["num_vertices"],
                "num_edges": data["num_edges"],
                "dim_features": data["dim_features"], #data["dim_features"], #data["dim_features"], #facebook_feature_dim, #data["dim_features"], #facebook_feature_dim,
                "features": data["features"],
                "edge_list": data["edge_list"],
                "labels": data["labels"]}


    clientNum = 3

    G = get_global_graph(data["edge_list"], data["features"].numpy(), range(0, data["num_vertices"]))

    node_subjects = pd.Series(data["labels"])

    communities = louvain_graph_cut(G, node_subjects, clientNum)

    # valFileOuts = "BlogCatalog_communites_c.npy"
    # np.save(valFileOuts, communities)

    dataList, train_mask_list, val_mask_list, test_mask_list, \
    test_idx_list, trainN_list, val_idx_list = getCommunitData(data, communities, node_subjects, clientNum)

    #--------------splitting data and create client net----------------#
    train_local = False
    train_fedavg = False
    center_flag = True

    clientList = []

    # samplingRate = [0.3, 0.4, 0.5, 0.5, 0.6, 0.7]
    samplingRate = [0.3, 0.4, 0.6]
    # samplingRate = [1, 1, 1, 1, 1, 1]
    # samplingRate = [0.7, 0.4, 0.5, 0.5, 0.6, 0.7]
    classNum = data["num_classes"]
    # sampleNumEachClass = [5, 5, 10, 10, 20, 30]                #Citeseer 200  others_10
    sampleNumEachClass = [20, 20, 10, 10, 5, 5]                   #
    filePath = 'cora/Community3/'              #"BlogCatalog/"  'flickr/'   #BlogCatalog   #facebook_200
    overlapRate = 0.15
    # clientSamplingIndex = random_dataset_overlap(data, clientNum, overlapRate)
    globalKModels = []
    n_clusters = 60
    hid_channels = 16
    globalNet = GlobalGCN(data, test_idx_list)
    valist = {clientId: [] for clientId in range(clientNum)}

    for i in range(clientNum):
        saveName = filePath + 'client' + str(i)
        #------------fix----overlapNodes-----------------#
        # samplingIndex = clientSamplingIndex[i]
        # clientData, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum = getOverlapClientData(data, samplingIndex, sampleNumEachClass, classNum, saveName)
        #---------random---------------------------------#
        # clientData, train_mask, val_mask, test_mask, samplingMatrix, sampling_idx_range, trainNum = getClientData(data, samplingRate[i], sampleNumEachClass, classNum, saveName)
        # net = ClientGCN(clientData, train_mask, val_mask, test_mask, samplingMatrix, trainNum)
        clientData = dataList[i]
        train_mask = train_mask_list[i]
        val_mask = val_mask_list[i]
        test_mask = test_mask_list[i]
        train_Num = trainN_list[i]

        net = ClientGCN(clientData, train_mask, val_mask, test_mask, train_Num)


        #---------------保存初始化参数-----------#
        path0 = "client" + str(i) + "_cora_3_c_localModel_lv.pkl"
        # net.localSModel.load_state_dict(torch.load(path0))
        # torch.save(net.localSModel.state_dict(), path0)
        path1 = "client" + str(i) + "_cora_3_c_globalKModel.pkl"
        # net.globalKModel.load_state_dict(torch.load(path1))
        # torch.save(net.globalKModel.state_dict(), path1)
        path2 = "client" + str(i) + "_cora_3_c_globalCModel.pkl"
        # torch.save(net.globalCModel.state_dict(), path2)
        # net.globalCModel.load_state_dict(torch.load(path1))

        clientList.append(net)

    iteration = 4000
    if center_flag:
        aggParams = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
        for epoch in range(iteration):
            # 1、-------training and aggerating------------------#
            localOuts = []
            localOutlists = {}
            for i in range(clientNum):
                localOut, localOutlist = clientList[i].getLocalTrainOut()
                localOuts.append(localOut)
                localOutlists[i] = localOutlist

            # getClusterAdj(localOuts, clientList, n_clusters)

            for i in range(clientNum):
                clientList[i].trainNet(localOuts[i], localOutlists[i], i, epoch)
                valll = clientList[i].val(i, epoch)
                valist[i].append(valll)
                #####################################################
                for pid, param in enumerate(list(clientList[i].parameters())):
                    aggParams[pid] += param.detach().numpy()

            # 2、--------getting average parameters-------------#
            for id, aggParam in aggParams.items():
                aggParams[id] = aggParam / clientNum

            expDis = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
                      6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: []}  # , 4: [], 5: [], 6: [], 7: [], 8: []
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    currentParam = param.detach().numpy()
                    meanParam = aggParams[pid]
                    dist = np.linalg.norm(currentParam - meanParam)
                    if dist > 700:
                        currentDis = math.exp(dist / 100)  # math.exp(dist/100)*math.exp(100)
                    else:
                        currentDis = math.exp(dist)
                    expDis[pid].append(currentDis)

            globalAggParam = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                              6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}  # , 4: 0, 5: 0, 6: 0, 7: 0, 8: 0
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    globalAggParam[pid] += (expDis[pid][i] / sum(expDis[pid])) * param.detach().numpy()
            #
            # # 3、-----------setting aggerated parameters for each client-----------#
            u = 0.01
            b = 0.3
            for i in range(clientNum):
                for pid, param in enumerate(list(clientList[i].parameters())):
                    # if pid == 1:
                    #     param.data = b * param.data + (1-b) * torch.tensor(globalAggParam[pid])
                    # else:
                        param.data = torch.tensor(globalAggParam[pid])
                    # param.data = torch.tensor(globalAggParam[pid])  # aggParams[pid] + u * (aggParams[pid] - param.data) #aggParams[pid] #finalAggParam#aggParams[pid]
                    # param.data = torch.tensor(aggParams[pid] + u * (
                    #         aggParams[pid] - param.detach().numpy()))
            # 4、----------clear aggParams--------------------------#
            for key in aggParams.keys():
                aggParams[key] = 0

            # if epoch == 1999:
            globalNet.test(globalAggParam, epoch, iteration)


    if train_local:
        aggParams = {}
        for pid, param in enumerate(list(clientList[0].parameters())):
            aggParams[pid] = 0

        for epoch in range(iteration):
            #1、-------training and aggerating------------------#
            localOuts = []
            localOutlists = {}
            for i in range(clientNum):
                localOut, localOutlist = clientList[i].getLocalTrainOut()
                localOuts.append(localOut)
                localOutlists[i] = localOutlist

            for i in range(clientNum):
                clientList[i].trainNet(localOuts[i], localOutlists[i], i, epoch)
                valll = clientList[i].val(i, epoch)
                valist[i].append(valll)
               #####################################################
                if train_fedavg:
                   for pid, param in enumerate(list(clientList[i].parameters())):
                       aggParams[pid] += param.detach().numpy()

            #2、--------getting average parameters-------------#
            if train_fedavg:
                for id, aggParam in aggParams.items():
                    aggParams[id] = aggParam/clientNum

                #3、-----------setting aggerated parameters for each client-----------#
                u = 0.05
                for i in range(clientNum):
                    for pid, param in enumerate(list(clientList[i].parameters())):
                        param.data = torch.tensor(aggParams[pid])#aggParams[pid] + u * (aggParams[pid] - param.data) #aggParams[pid] #finalAggParam#aggParams[pid]
                        # param.data = torch.tensor(aggParams[pid] + u * (
                        #         aggParams[pid] - param.detach().numpy()))
                # if epoch == 999:
                globalNet.test(aggParams, epoch, iteration)

                #4、----------clear aggParams--------------------------#
                for key in aggParams.keys():
                    aggParams[key] = 0
    ###-------plot-------
    color = 'blue'
    plt.figure(1)
    plt.plot(range(iteration), globalNet.acc, '-', color=color)

    plt.figure(2)
    plt.plot(range(iteration), valist[0], '-', color='red')
    plt.figure(3)
    plt.plot(range(iteration), valist[1], '-', color='green')
    plt.figure(4)
    plt.plot(range(iteration), valist[2], '-', color='yellow')

    plt.show()
    dataframe0 = pd.DataFrame(
        {'x': range(iteration), 'acc': globalNet.acc})  # save data
    dataframe1 = pd.DataFrame(
        {'x': range(iteration), 'f1': globalNet.f1})  # save data
    dataframe2 = pd.DataFrame(
        {'x': range(iteration), 'precision': globalNet.pre})  # save data
    dataframe3 = pd.DataFrame(
        {'x': range(iteration), 'recall': globalNet.recall})  # save data
    dataframe0.to_csv('cora/Community3/3_fedsala_FedLNA_Intra_0.5_inter_0.1_acc_acc.csv', index=False, sep=',')
    dataframe1.to_csv('cora/Community3/3_fedsala_FedLNA_Intra_0.5_inter_0.1_f1_acc.csv', index=False, sep=',')
    dataframe2.to_csv('cora/Community3/3_fedsala_FedLNA_Intra_0.5_inter_0.1_precision_acc.csv', index=False, sep=',')
    dataframe3.to_csv('cora/Community3/3_fedsala_FedLNA_Intra_0.5_inter_0.1_recall_acc.csv', index=False, sep=',')


    # print("\nsave loss")
    # np.save('BlogCatalog/loss.npy', clientLoss)  # 注意带上后缀名
    #
    # print("\nsave acc")
    # np.save('BlogCatalog/acc.npy', clientAcc)  # 注意带上后缀名

    print("\ntrain finished!")
    for i in range(clientNum):
        clientList[i].test(i)


    # print("\nsave final embedding")
    # for i in range(clientNum):
    #     saveName = filePath + "client" + str(i)
    #     clientList[i].saveFinalEmbedding(saveName)


