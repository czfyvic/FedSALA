import torch
import torch.nn as nn
import pyro
import dhg
from sklearn.neighbors import kneighbors_graph
from dhg.nn import GCNConv
from sklearn.cluster import KMeans
import torch.nn.functional as F
import numpy as np
import math

class MultiGCN(nn.Module):
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.originGraph = GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.knnGraph = GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.augGraph = GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.temperature = 0.5


    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
                ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
                ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        embeding0 = self.originGraph(X, g)
        outs.append(embeding0)

        knng = self.getKnnGraph(embeding0)
        aug = self.getAugGraph(embeding0)

        embeding1 = self.knnGraph(X, knng)
        outs.append(embeding1)

        embeding2 = self.augGraph(X, aug)
        outs.append(embeding2)

        return outs

    def getCrossLoss(self, x_l, x_g, label):
        curItem = x_l[0]
        curLbl = label[0]
        posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
        negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
        posItems = x_g[posIndex]
        negItems = x_g[negIndex]

        curItemList = curItem.repeat([len(posIndex), 1])
        posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
        posSim = torch.exp(((posSim / self.temperature) / 100))
        posSim = posSim.sum(dim=0, keepdim=True)

        curItemList = curItem.repeat([len(negIndex), 1])
        negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
        negSim = torch.exp(((negSim / self.temperature) / 100))
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, len(x_l)):
            curItem = x_l[i]
            curLbl = label[i]
            posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
            negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
            posItems = x_g[posIndex]
            negItems = x_g[negIndex]

            curItemList = curItem.repeat([len(posIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, posItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature) / 100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            posSim = torch.cat((posSim, eachSim), dim=0)

            curItemList = curItem.repeat([len(negIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature) / 100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        loss = (-torch.log(posSim / negSim)).mean()
        return loss



class GCNS(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        # self.train_len = train_len
        self.layers = nn.ModuleList()
        self.k = 6

        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(200, 100, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(200, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.layers.append(GCNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        # self.outputLayer = nn.Linear(hid_channels, num_classes)
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))

        # self.reduce = nn.Linear(self.k * hid_channels, hid_channels)


        # self.centers = torch.nn.Parameter(torch.randn(num_classes, hid_channels))
        # self.neighCenters = torch.nn.Parameter(torch.randn(n_clusters, hid_channels))
        self.tc = 0.5
        self.tn = 0.5
        self.temperature = 0.5

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        layerCount = 0
        for layer in self.layers:
            X = layer(X, g)
            outs.append(X)
            layerCount += 1
            mean = 0
            # if layerCount == len(self.layers) - 1:
            #     for i in range(len(outs)):
            #         mean += outs[i].detach().numpy()
            #     mean = mean / len(outs)
            #     expDis = []
            #     for i in range(len(outs)):
            #         currentNodeEmd = outs[i].detach().numpy()
            #         dist = np.linalg.norm(currentNodeEmd - mean)
            #
            #         try:
            #             expDis.append(math.exp(dist))
            #         except OverflowError:
            #             expDis.append(math.exp(700))
            #
            #     finalNodeEmb = 0
            #     for i in range(len(outs)):
            #         currentNodeEmd = outs[i].detach().numpy()
            #         finalNodeEmb += (expDis[i] / sum(expDis)) * currentNodeEmd
            #     X.data = torch.Tensor(finalNodeEmb)

        # lastOuts = self.outputLayer(X)
        # outs.append(lastOuts)

            # if layerCount == len(self.layers) - 1:
            #     X.data = outs[1].detach()
            # if layerCount == len(self.layers) - 1:
            #     stackOuts = torch.cat(outs, dim=1)
            #     X.data = self.reduce(stackOuts)
            # elif layerCount > 1 and layerCount != len(self.layers):
            #     updateData = 0.0001 * outs[layerCount-2].detach() + outs[layerCount-1].detach()
            #     X.data = updateData

            # if layerCount == 2:
            #     X.data = outs[0].detach() + 0 * outs[1].detach()

            # if layerCount == len(self.layers) - 1:
            #     mean = 0
            #     for i in range(len(outs)):
            #         mean += outs[i].detach().numpy()
            #     mean = mean / layerCount
            #     expDis = []
            #     for i in range(len(outs)):
            #         currentNodeEmd = outs[i].detach().numpy()
            #         dist = np.linalg.norm(currentNodeEmd - mean)
            #
            #         try:
            #             expDis.append(math.exp(dist))
            #         except OverflowError:
            #             expDis.append(math.exp(700))
            #
            #     finalNodeEmb = 0
            #     for i in range(len(outs)):
            #         currentNodeEmd = outs[i].detach().numpy()
            #         finalNodeEmb += (expDis[i] / sum(expDis)) * currentNodeEmd
            #     X.data = torch.Tensor(finalNodeEmb)

        return outs

    # def getIntraInterCenterLoss(self, x, label, a, b):
    #     batch_size = x.size(0)
    #     currentCenter = self.centers[label]
    #     intraDis = x - currentCenter  #((x - currentCenter)**2).sum()/2
    #     intraLoss = torch.norm(intraDis, p=2) * a
    #
    #     interLoss = torch.zeros()
    #     for i in range(1, batch_size):
    #         c_j = self.centers[label[i]]
    #         c_j = c_j.repeat([self.num_classes, 1])
    #         eachDis = c_j - self.centers #((c_j - self.centers)**2).sum()/2
    #         eachDis = torch.norm(eachDis, p=2)
    #         interLoss += eachDis
    #     interLoss = interLoss.mean() * b
    #
    #     loss = intraLoss + interLoss
    #     return loss

    # def getCenterAndCorNeighCenters(self, neighCenters, neighCenterPos, train_index, lbls):
    #     centerAndNeighCenters = []
    #     trainNeighCenterPos = torch.tensor(neighCenterPos)
    #     clsIndex = []
    #     for i in range(self.num_classes):
    #         # for curlbl, ind in enumerate(lbls):
    #         #     if curlbl == i:
    #         #         clsIndex.append(ind)
    #         clsIndex = [ind for ind, curlbl  in enumerate(lbls) if (curlbl.item() == i)]
    #         curNeighCenterPos = trainNeighCenterPos[clsIndex]
    #         curNeighCenterPos = curNeighCenterPos.numpy().tolist()
    #         if len(clsIndex) != 1:
    #            curNeighCenterPos = list(set(curNeighCenterPos))
    #         curNeighCenterPos = torch.tensor(curNeighCenterPos)
    #         curNeighCenters = neighCenters[curNeighCenterPos]
    #         centerAndNeighCenters.append(curNeighCenters)
    #     return centerAndNeighCenters
    #
    # def getNeighCenterAndCenterLoss(self, neighCenters):
    #     curCenters = self.centers[0]
    #     curCenters = curCenters.repeat([self.num_classes, 1])
    #
    #     posSim = torch.cosine_similarity(neighCenters, curCenters, dim=1)
    #     posSim = torch.exp(posSim / self.temperature)
    #
    #     curNeighCenter = neighCenters[0]
    #     eachX = curNeighCenter.repeat([self.num_classes, 1])
    #     negSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #     negSim = negSim.sum(dim=0, keepdim=True)
    #
    #     for i in range(1, self.n_clusters):
    #         curNeighCenter = neighCenters[i]
    #
    #         eachX = curNeighCenter.repeat([self.num_classes, 1])
    #         eachSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #         negSim = torch.cat((negSim, eachSim), dim=0)
    #
    #     negSim = torch.exp((negSim / self.temperature) / 100)
    #     loss = (-torch.log(posSim / negSim)).mean()
    #
    #     return loss
    #
    # def getCorCenters(self, neighIndex, lbls):
    #     corCenters = []
    #     for i in range(self.n_clusters):
    #         tt = []
    #         for ind, val in enumerate(neighIndex):
    #             if val == i:
    #                 curLbl = lbls[ind]
    #                 tt.append(curLbl.item())
    #         corCenters.append(tt)
    #     return corCenters

    # def getCenterLoss(self, neighIndex, lbls):
    #     corCenters = self.getCorCenters(neighIndex, lbls)
    #
    #     posSim = torch.cosine_similarity(self.neighCenters, corCenters, dim=1)
    #     posSim = torch.exp(posSim / self.tc)
    #
    #     curItem = self.neighCenters[0]
    #     eachX = curItem.repeat([self.num_classes, 1])  # 得到负向的第一项
    #     negSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #     negSim = negSim.sum(dim=0, keepdim=True)
    #
    #     for i in range(1, self.n_clusters):
    #         curItem = self.neighCenters[i]
    #         eachX = curItem.repeat([self.num_classes, 1])  # 得到负向的第一项
    #         eachSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #
    #         negSim = torch.cat((negSim, eachSim), dim=0)
    #
    #     negSim = torch.exp(negSim / self.tc)
    #     centerLoss = (-torch.log(posSim / negSim)).mean()
    #
    #     return centerLoss

    # def getNeighborLoss(self, x, train_index, neighIndex, lbls):
    #     train_index = [index for index, val in enumerate(train_index) if val]
    #     curIndex = train_index[0]
    #     curItem = x[curIndex]    #获取当前的项
    #     curItems = x[train_index]
    #     neighIndex = neighIndex[train_index] #获取train_index对应的邻居中心点
    #
    #     posNeiCenters = self.neighCenters[neighIndex]    #得到正项
    #     posSim = torch.cosine_similarity(curItems, posNeiCenters, dim=1)
    #     posSim = torch.exp(posSim / self.tn)
    #
    #     eachX = curItem.repeat([self.n_clusters, 1])   #得到负向的第一项
    #     negSim = torch.cosine_similarity(eachX, self.neighCenters, dim=1)
    #     negSim = negSim.sum(dim=0, keepdim=True)
    #
    #     for i in range(1, len(train_index)):
    #         curIndex = train_index[i]
    #         curItem = x[curIndex]
    #
    #         eachX = curItem.repeat([self.n_clusters, 1])
    #         eachSim = torch.cosine_similarity(eachX, self.neighCenters, dim=1)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #         negSim = torch.cat((negSim, eachSim), dim=0)
    #
    #     negSim = torch.exp(negSim / self.tn)
    #     neighLoss = (-torch.log(posSim / negSim)).mean()
    #
    #     loss = neighLoss
    #
    #     return loss

    # def getCenterLoss(self, x, label):
    #     """
    #     Args:
    #         x: feature matrix with shape (batch_size, feat_dim).
    #         labels: ground truth labels with shape (batch_size).
    #     """
    #     # print(self.centers)
    #     batch_size = x.size(0)
    #     currentCenter = self.centers[label]
    #     # print(self.centers)
    #
    #     ###############based on similarity##########################
    #     posSim = torch.cosine_similarity(x, currentCenter, dim=1)
    #     posSim = torch.exp((posSim / self.temperature)/100)
    #
    #     eachX = x[0].repeat([self.num_classes, 1])
    #     negSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #     negSim = negSim.sum(dim=0, keepdim=True)
    #
    #     for i in range(1, batch_size):
    #         eachX = x[i].repeat([self.num_classes, 1])
    #         eachSim = torch.cosine_similarity(eachX, self.centers, dim=1)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #         negSim = torch.cat((negSim, eachSim), dim=0)
    #
    #     negSim = torch.exp((negSim / self.temperature)/100)
    #     loss = (-torch.log(posSim / negSim)).mean()
    #
    #     return loss

class GCNT0(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 temperature: float = 0.5,
                 use_bn: bool = False,
                 drop_rate: float = 0.5,
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))
        self.temperature = temperature


    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        layerCount = 0
        for layer in self.layers:
            X = layer(X, g)
            layerCount += 1
            # if layerCount == 1:
            #     X = X + self.feaIntervene   # do intervene for feature
            outs.append(X)
        return outs


    def getNeiContrastLoss(self, x_l, x_g, globalKnnAdj, trainIndex):  # local and global, 同一类中的节点为正例，不同类中的节点为反例
        curItem = x_l[0]
        trainIndex = [i for i, x in enumerate(trainIndex) if x]
        curIndex = trainIndex[0]

        curNeighbor = globalKnnAdj[curIndex]
        posIndex = [i for i, x in enumerate(curNeighbor) if x != 0]
        negIndex = [i for i, x in enumerate(curNeighbor) if x == 0]
        posItems = x_g[posIndex]
        negItems = x_g[negIndex]

        curItemList = curItem.repeat([len(posIndex), 1])
        posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
        posSim = torch.exp(((posSim / self.temperature) / 100))
        posSim = posSim.sum(dim=0, keepdim=True)

        curItemList = curItem.repeat([len(negIndex), 1])
        negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
        negSim = torch.exp(((negSim / self.temperature) / 100))
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, len(x_l)):
            curItem = x_l[i]
            curIndex = trainIndex[i]
            curNeighbor = globalKnnAdj[curIndex]
            posIndex = [i for i, x in enumerate(curNeighbor) if x != 0]
            negIndex = [i for i, x in enumerate(curNeighbor) if x == 0]
            posItems = x_g[posIndex]
            negItems = x_g[negIndex]

            curItemList = curItem.repeat([len(posIndex), 1])
            posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
            posSim = torch.exp(((posSim / self.temperature) / 100))
            posSim = posSim.sum(dim=0, keepdim=True)

            curItemList = curItem.repeat([len(negIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
            eachSim = torch.exp(((eachSim / self.temperature) / 100))
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        loss = (-torch.log(posSim / negSim)).mean()
        return loss



    def getCrossLoss(self, x_l, x_g, label):  # local and global, 同一类中的节点为正例，不同类中的节点为反例
        curItem = x_l[0]
        curLbl = label[0]
        posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
        negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
        posItems = x_g[posIndex]
        negItems = x_g[negIndex]

        curItemList = curItem.repeat([len(posIndex), 1])
        posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
        posSim = torch.exp(((posSim / self.temperature)/100))
        posSim = posSim.sum(dim=0, keepdim=True)

        curItemList = curItem.repeat([len(negIndex), 1])
        negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
        negSim = torch.exp(((negSim / self.temperature)/100))
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, len(x_l)):
            curItem = x_l[i]
            curLbl = label[i]
            posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
            negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
            posItems = x_g[posIndex]
            negItems = x_g[negIndex]

            curItemList = curItem.repeat([len(posIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, posItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            posSim = torch.cat((posSim, eachSim), dim=0)

            curItemList = curItem.repeat([len(negIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        loss = (-torch.log(posSim / negSim)).mean()
        return loss

class GCNT1(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 temperature: float = 0.5,
                 use_bn: bool = False,
                 drop_rate: float = 0.5,
                 ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))
        self.temperature = temperature
        self.x_pre = 0


    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        outs = []
        layerCount = 0
        for layer in self.layers:
            X = layer(X, g)
            layerCount += 1
            # if layerCount == 1:
            #     X = X + self.feaIntervene   # do intervene for feature
            outs.append(X)
        return outs

    def getMoonCrossLoss(self, epoch, x_l, x_g):
        if epoch == 0:
            self.x_pre = x_l.detach()
        posSim = torch.cosine_similarity(x_l, x_g, dim=1)
        posSim = torch.exp((posSim / self.temperature))
        posSim = posSim.sum(dim=0, keepdim=True)

        negSim = torch.cosine_similarity(x_l, self.x_pre, dim=1)
        negSim = negSim.sum(dim=0, keepdim=True)
        negSim = posSim + negSim

        self.x_pre = x_l.detach()
        loss = (-torch.log(posSim / negSim)).mean()
        return loss


    def getCrossLoss(self, x_l, x_g, label):  # local and global, 同一类中的节点为正例，不同类中的节点为反例
        curItem = x_l[0]
        curLbl = label[0]
        posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
        negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
        posItems = x_g[posIndex]
        negItems = x_g[negIndex]

        curItemList = curItem.repeat([len(posIndex), 1])
        posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
        posSim = torch.exp(((posSim / self.temperature)/100))
        posSim = posSim.sum(dim=0, keepdim=True)

        curItemList = curItem.repeat([len(negIndex), 1])
        negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
        negSim = torch.exp(((negSim / self.temperature)/100))
        negSim = negSim.sum(dim=0, keepdim=True)

        for i in range(1, len(x_l)):
            curItem = x_l[i]
            curLbl = label[i]
            posIndex = [i for i, lbl in enumerate(label) if lbl == curLbl]
            negIndex = [i for i, lbl in enumerate(label) if lbl != curLbl]
            posItems = x_g[posIndex]
            negItems = x_g[negIndex]

            curItemList = curItem.repeat([len(posIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, posItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            posSim = torch.cat((posSim, eachSim), dim=0)

            curItemList = curItem.repeat([len(negIndex), 1])
            eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
            eachSim = torch.exp((eachSim / self.temperature)/100)
            eachSim = eachSim.sum(dim=0, keepdim=True)
            negSim = torch.cat((negSim, eachSim), dim=0)

        loss = (-torch.log(posSim / negSim)).mean()
        return loss


    # def getCrossLoss(self, x_l, x_g, label, posIndexs, negIndexs):  # local and global, 同一类中的节点为正例，不同类中的节点为反例
    #     curItem = x_l[0]
    #     curLbl = label[0]
    #     posIndex = posIndexs[0]    #[i for i, lbl in enumerate(label) if lbl == curLbl]
    #     negIndex = negIndexs[0]    #[i for i, lbl in enumerate(label) if lbl != curLbl]
    #     posItems = x_g[posIndex]
    #     negItems = x_g[negIndex]
    #
    #     curItemList = curItem.repeat([len(posIndex), 1])
    #     posSim = torch.cosine_similarity(curItemList, posItems, dim=1)
    #     posSim = torch.exp(((posSim / self.temperature)/100))
    #     posSim = posSim.sum(dim=0, keepdim=True)
    #
    #     curItemList = curItem.repeat([len(negIndex), 1])
    #     negSim = torch.cosine_similarity(curItemList, negItems, dim=1)
    #     negSim = torch.exp(((negSim / self.temperature)/100))
    #     negSim = negSim.sum(dim=0, keepdim=True)
    #
    #     for i in range(1, len(x_l)):
    #         curItem = x_l[i]
    #         curLbl = label[i]
    #         posIndex = posIndexs[i]    #[i for i, lbl in enumerate(label) if lbl == curLbl]
    #         negIndex = negIndexs[i]    #[i for i, lbl in enumerate(label) if lbl != curLbl]
    #         posItems = x_g[posIndex]
    #         negItems = x_g[negIndex]
    #
    #         curItemList = curItem.repeat([len(posIndex), 1])
    #         eachSim = torch.cosine_similarity(curItemList, posItems, dim=1)
    #         eachSim = torch.exp((eachSim / self.temperature)/100)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #         posSim = torch.cat((posSim, eachSim), dim=0)
    #
    #         curItemList = curItem.repeat([len(negIndex), 1])
    #         eachSim = torch.cosine_similarity(curItemList, negItems, dim=1)
    #         eachSim = torch.exp((eachSim / self.temperature)/100)
    #         eachSim = eachSim.sum(dim=0, keepdim=True)
    #         negSim = torch.cat((negSim, eachSim), dim=0)
    #
    #     loss = (-torch.log(posSim / negSim)).mean()
    #     return loss


