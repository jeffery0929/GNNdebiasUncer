from torch_geometric.data import Data,Batch,DataLoader
import torch
import math
import numpy as np
from torch_geometric.utils import (remove_self_loops, degree,
                                   batched_negative_sampling)
from torch_geometric.utils.num_nodes import maybe_num_nodes

def split_batch(edge_index,batch):
  split = degree(batch[edge_index[0]], dtype=torch.long).tolist()
  edge_indices = torch.split(edge_index, split, dim=1)
  num_nodes = degree(batch, dtype=torch.long)
  cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
  num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(device)
  cum_edges = torch.cat([batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])
  return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

def split_graph_bybatch(edge_weight_o,xo_node_att,xc_node_att,edge_index,batch,ratio=0.5):

  causal_edge_index = torch.LongTensor([[],[]]).to(device)
  causal_edge_weight = torch.tensor([]).to(device)
  conf_edge_index = torch.LongTensor([[],[]]).to(device)
  conf_edge_weight = torch.tensor([]).to(device)

  #causal_edge_attr = torch.tensor([])
  topnode=torch.LongTensor([]).to(device)
  notopnode=torch.LongTensor([]).to(device)

  edge_indices, _, _, num_edges, cum_edges = split_batch(edge_index,batch)
  for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
      n_reserve =  int(ratio * N)
      edge_attr = edge_weight_o[C:C+N]
      single_mask = edge_weight_o[C:C+N]
      single_mask_detach = edge_weight_o[C:C+N].detach().cpu().numpy()
      rank = np.argpartition(-single_mask_detach, n_reserve)
      idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

      causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
      conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)
      #print(causal_edge_index)
      causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
      conf_edge_weight = torch.cat([conf_edge_weight, 1-single_mask[idx_drop]])
      causal_sub_nodes = torch.unique(edge_index[:, idx_reserve])
      non_sub_nodes=torch.unique(edge_index[:, idx_drop])
      #print(xo_node_att[causal_sub_nodes])
      #print(f'non_subnode:{non_sub_nodes}')
    # print(f'non weight:{ -1 * single_mask[idx_drop]}')
      #print(f'subnode:{causal_sub_nodes}')
      #print(f'c subnode weight:{single_mask[idx_reserve]}')
      topnon=torch.topk(xc_node_att[non_sub_nodes],int(len(non_sub_nodes)*0.8))[1]

      top=torch.topk(xo_node_att[causal_sub_nodes],int(len(causal_sub_nodes)*0.2))[1]

      topnode = torch.cat([topnode, causal_sub_nodes[top]])
      notopnode=torch.cat([notopnode, non_sub_nodes[topnon]])
      #print(topnode)
      #print(f'top node:{topnode}')
      #print(f'nontop node:{notopnode}')
  return (causal_edge_index,causal_edge_weight,topnode),(conf_edge_index,conf_edge_weight,notopnode)


def k_hop_subgraph(node_idx,num_hops,edge_index,ew,num_nodes = None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)


    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True


    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    edge_weight=torch.masked_select(ew,edge_mask)

    #if relabel_nodes:
    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    redge_index = node_idx[edge_index]

    return subset,edge_index,inv,edge_weight,redge_index