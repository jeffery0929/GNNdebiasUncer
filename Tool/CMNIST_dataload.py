import numpy as np
from scipy.spatial.distance import cdist
import torch
import dgl
import pickle


def sigma(dists, kth=8):
    # Get k-nearest neighbors for each node
    knns = np.partition(dists, kth, axis=-1)[:, kth::-1]

    # Compute sigma and reshape
    sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    return sigma + 1e-8 # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=False, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)

    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist/sigma(c_dist))**2 - (f_dist/sigma(f_dist))**2 )
    else:
        A = np.exp(- (c_dist/sigma(c_dist))**2)

    # Convert to symmetric matrix
    A = 0.5 * A * A.T
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node
    if 1==1:
        num_nodes = A.shape[0]
        new_kth = num_nodes - kth
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knns_d = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1]
    else:
        knns = np.argpartition(A, kth, axis=-1)[:, kth::-1]
        knns_d = np.partition(A, kth, axis=-1)[:, kth::-1]
    return knns, knns_d

class newCIFARSuperPix(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 use_mean_px=True,
                 use_coord=True,
                 use_feat_for_graph_construct=False,):

        #self.split = split
        #self.is_test = split.lower() in ['test', 'val']
        with open(data_dir, 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        self.use_mean_px = use_mean_px
        self.use_feat_for_graph = use_feat_for_graph_construct
        self.use_coord = use_coord
        self.n_samples = len(self.labels)
        self.img_size = 32

    def precompute_graph_images(self):
        #print('precompute all data for the %s set...' % self.split.upper())
        self.Adj_matrices, self.node_features, self.edges_lists = [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord, mean_px, use_feat=self.use_feat_for_graph)
            edges_list, _ = compute_edges_list(A)
            N_nodes = A.shape[0]

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features

            self.node_features.append(x)
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        g = dgl.DGLGraph()
        g.add_nodes(self.node_features[index].shape[0])
        g.ndata['feat'] = torch.Tensor(self.node_features[index])
        for src, dsts in enumerate(self.edges_lists[index]):
            g.add_edges(src, dsts[dsts!=src])

        return g, self.labels[index]




#####################################
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset,Data
import os.path as osp


def compute_adjacency_matrix_images_process(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A

def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data

def process(data_file):
  use_mean_px=True
  use_coord=True
  node_gt_att_threshold=0
  transform=None
  pre_transform=None
  pre_filter=None

  #data_file ='/content/drive/MyDrive/Colab_Notebooks/colorMNIST05_2000_75sp_train.pkl'

  with open(osp.join(data_file), 'rb') as f:
      labels,sp_data = pickle.load(f)


  #use_mean_px = self.use_mean_px
  #self.use_coord = self.use_coord
  n_samples = len(labels)
  img_size = 28
  #node_gt_att_threshold = self.node_gt_att_threshold

  edge_indices,xs,edge_attrs,node_gt_atts,edge_gt_atts = [], [], [], [], []
  data_list = []
  for index, sample in enumerate(sp_data):
      mean_px, coord = sample[:2]
      coord = coord / img_size
      A = compute_adjacency_matrix_images_process(coord)
      N_nodes = A.shape[0]

      A = torch.FloatTensor((A > 0.1) * A)
      edge_index, edge_attr = dense_to_sparse(A)

      x = None
      if use_mean_px:
          x = mean_px.reshape(N_nodes, -1)
      if use_coord:
          coord = coord.reshape(N_nodes, 2)
          if use_mean_px:
              x = np.concatenate((x, coord), axis=1)
          else:
              x = coord
      if x is None:
          x = np.ones(N_nodes, 1)  # dummy features

      # replicate features to make it possible to test on colored images
      x = np.pad(x, ((0, 0), (2, 0)), 'edge')
      if node_gt_att_threshold == 0:
          node_gt_att = (mean_px > 0).astype(np.float32)
      else:
          node_gt_att = mean_px.copy()
          node_gt_att[node_gt_att < node_gt_att_threshold] = 0

      node_gt_att = torch.LongTensor(node_gt_att).view(-1)
      row, col = edge_index
      edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

      data_list.append(
          Data(
              x=torch.tensor(x),
              y=torch.LongTensor([labels[index]]),
              edge_index=edge_index,
              edge_attr=edge_attr,
              node_gt_att=node_gt_att,
              edge_gt_att=edge_gt_att

          )
      )

  #torch.save(InMemoryDataset.collate(data_list), '/content/drive/MyDrive/Colab_Notebooks/colorMINST05_2000.pt')
  return data_list

