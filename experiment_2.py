from generate_labels import *
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from models.egnn_main import *
from models.mpnn_main import *
from random import shuffle

#2a) Fix Graph, Vary Random Node Features and X. 
#2b) Fix Random Node Features, Vary Graph and X
#2c) Fix X. Vary Random Node Features and Graph
# In which case is data efficiency helped the most. What is the effect of X on the label
# Does random node feature make sense?

#GENERATE DATASET:

n_nodes = 12
p = 0.5
input_nf, output_nf, hidden_nf, coords = 5, 1, 24, 3
mu_x, sigma_x, mu_h, sigma_h = 0, 1, 1.5, 0.01
training_set_size = 10000
batch_size = 32

training_set = []
num_batches = training_set_size // batch_size
nodes_list = list(range(n_nodes))
edge_index = erdos_renyi_graph(n_nodes,p)
x = torch.tensor([[1]]*n_nodes, dtype=torch.float)
data = Data(x=x, edge_index=edge_index.contiguous())
G = to_networkx(data)
adj = np.array(nx.linalg.graphmatrix.adjacency_matrix(G).todense())
for i in range(num_batches):
  edges, edge_attr = get_edges_batch(n_nodes, batch_size)
  adj_list = [adj]*batch_size
  batch_edge_index = [edge_index]*batch_size
  edge_mask = generate_edge_mask(batch_size, n_nodes, nodes_list, batch_edge_index)
  tensor_edge_mask = torch.tensor(edge_mask, dtype=torch.float32).reshape(-1,1)
  training_set.append((edges, edge_attr, tensor_edge_mask, adj_list))

#Generate Features and coordinates
feat_tr = torch.normal(mean=mu_x,std=sigma_x, size=(training_set_size * n_nodes, input_nf))
coord_tr = torch.normal(mean=mu_h,std=sigma_h,size=(training_set_size * n_nodes, coords))


#Geenrate Test Dataset:
test_set_size = 500
test_set = []
edges_test, edge_attr_test = get_edges_batch(n_nodes, test_set_size)
adj_list_test = [adj]*test_set_size
batch_edge_index_test = [edge_index]*test_set_size
edge_mask_test = generate_edge_mask(test_set_size, n_nodes, nodes_list, batch_edge_index_test)
tensor_edge_mask_test = torch.tensor(edge_mask_test, dtype=torch.float32).reshape(-1,1)
test_set.append((edges_test, edge_attr_test, tensor_edge_mask_test, adj_list_test))
feat_test = torch.normal(mean=mu_x,std=sigma_x, size=(test_set_size * n_nodes, input_nf))
temp_coord_test = np.random.normal(mu_h, sigma_h, size=(test_set_size * n_nodes, coords))
#mpnn_test_in = torch.cat([feat_test, temp_coord_test], axis=1)

#Initializr NN:
mpnn = MPNN(in_node_nf=input_nf, coords=coords, in_edge_nf=1, hidden_nf=hidden_nf, n_layers=4)
egnn = EGNN(in_node_nf=input_nf, in_edge_nf=1, hidden_nf=hidden_nf, n_layers=4)


max_epoch = 20
clip_value = 1
optimizer_mpnn = optim.Adam(mpnn.parameters(), lr=0.0001)
mpnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mpnn, max_epoch)
f1 = nn.L1Loss()
f2 = nn.L1Loss()
f3 = nn.L1Loss()
f4 = nn.L1Loss()
g1 = nn.L1Loss()
g2 = nn.L1Loss()
g3 = nn.L1Loss()
g4 = nn.L1Loss()
optimizer_egnn = optim.Adam(egnn.parameters(), lr=0.0001)
egnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_egnn, max_epoch)
loss_mpnn_x_l = []
loss_mpnn_h_l = []
loss_egnn_x_l = []
loss_egnn_h_l = []
test_loss_mpnn_x_l = []
test_loss_mpnn_h_l = []
test_loss_egnn_x_l = []
test_loss_egnn_h_l = []

for i in range(max_epoch):
  shuffle(training_set)
  epoch_loss_mpnn_x_l = []
  epoch_loss_mpnn_h_l = []
  epoch_loss_egnn_x_l = []
  epoch_loss_egnn_h_l = []
  for iters in range(len(training_set)):
    mpnn_scheduler.step()
    egnn_scheduler.step()
    #Each iteration is process #batch_size number of graphs
    coord = torch.normal(mean=mu_x,std=sigma_x,size=(batch_size * n_nodes, coords))
    feat = feat_tr[iters*batch_size*n_nodes:(iters+1)*batch_size*n_nodes,:]
    edges, edge_attr, tensor_edge_mask, adj_list = training_set[iters]
    l_gen = LabelGenerate(coord, feat, edges, edge_attr, batch_size, n_nodes, adj_list)
    x_1, h_1 =  l_gen.get_labels(type_f2 = 'distance')
    h_1 = torch.tensor(h_1)
    x = torch.cat([feat, coord], axis=1)
    out_mpnn = mpnn(x, edges, edge_attr, tensor_edge_mask, n_nodes)
    #h_mpnn, x_mpnn = out_mpnn[:,0], out_mpnn[:,1:]
    h_egnn, x_egnn = egnn(feat, coord, edges, edge_attr, tensor_edge_mask, n_nodes)

    #out_egnn = torch.cat([x_egnn, h_egnn],axis=1)
    #label = torch.cat([x_1, h_1], axis=1)
    loss_mpnn_h = f1(out_mpnn[:,0].reshape(-1,1), h_1)
    loss_egnn_h = f2(h_egnn, h_1)
    loss_mpnn_x = f3(out_mpnn[:,1:], x_1)
    loss_egnn_x = f4(x_egnn, x_1)
    loss_mpnn = loss_mpnn_h + loss_mpnn_x
    loss_egnn = loss_egnn_h + loss_egnn_x

    loss_mpnn.backward()
    loss_egnn.backward()
    torch.nn.utils.clip_grad_norm_(egnn.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(mpnn.parameters(), clip_value)
    optimizer_mpnn.step()
    optimizer_egnn.step()
    epoch_loss_mpnn_x_l.append(loss_mpnn_x.detach().numpy())
    epoch_loss_mpnn_h_l.append(loss_mpnn_h.detach().numpy())
    epoch_loss_egnn_x_l.append(loss_egnn_x.detach().numpy())
    epoch_loss_egnn_h_l.append(loss_egnn_h.detach().numpy())
    
  loss_mpnn_x_l.append(np.mean(np.array(epoch_loss_mpnn_x_l)))
  loss_mpnn_h_l.append(np.mean(np.array(epoch_loss_mpnn_h_l)))
  loss_egnn_x_l.append(np.mean(np.array(epoch_loss_egnn_x_l)))
  loss_egnn_h_l.append(np.mean(np.array(epoch_loss_egnn_h_l)))
  mpnn_epoch_loss_mean = np.mean(np.array(epoch_loss_mpnn_x_l)) + np.mean(np.array(epoch_loss_mpnn_h_l))
  egnn_epoch_loss_mean = np.mean(np.array(epoch_loss_egnn_x_l)) + np.mean(np.array(epoch_loss_egnn_h_l))
    
  coord_test = torch.tensor(temp_coord_test, dtype=torch.float32 ,requires_grad=False)
  mpnn_test_in = torch.cat([feat_test, coord_test], axis=1)

  l_gen_test = LabelGenerate(coord_test, feat_test, edges_test, edge_attr_test, test_set_size, n_nodes, adj_list_test)
  x_test, h_test =  l_gen_test.get_labels(type_f2 = 'distance')
  h_test = torch.tensor(h_test)
  out_mpnn_test = mpnn(mpnn_test_in, edges_test, edge_attr_test, tensor_edge_mask_test, n_nodes)
  h_egnn_test, x_egnn_test = egnn(feat_test, coord_test, edges_test, edge_attr_test, tensor_edge_mask_test, n_nodes)
  test_loss_mpnn_h = g1(out_mpnn_test[:,0].reshape(-1,1), h_test)
  test_loss_egnn_h = g2(h_egnn_test, h_test)
  test_loss_mpnn_x = g3(out_mpnn_test[:,1:], x_test)
  test_loss_egnn_x = g4(x_egnn_test, x_test)
  test_loss_mpnn = test_loss_mpnn_h + test_loss_mpnn_x
  test_loss_egnn = test_loss_egnn_h + test_loss_egnn_x
  test_loss_mpnn_x_l.append(test_loss_mpnn_x)
  test_loss_mpnn_h_l.append(test_loss_mpnn_h)
  test_loss_egnn_x_l.append(test_loss_egnn_x)
  test_loss_egnn_h_l.append(test_loss_egnn_h)

  print("Epoch: {}, MPNN MSE: {:.4f}, EGNN MSE: {:.4f}, MPNN VAL: {:.4f}, EGNN VAL:{:.4f}".format((i+1), mpnn_epoch_loss_mean, egnn_epoch_loss_mean, test_loss_mpnn, test_loss_egnn))


#FIXED FEATURES

#GENERATE DATASET:

n_nodes = 4
p = 0.5
input_nf, output_nf, hidden_nf, coords = 5, 1, 24, 3
mu_x, sigma_x, mu_h, sigma_h = 0, 1, 1.5, 0.01

#Dataset(Train-Test)

#Train:

training_set_size = 30000
batch_size = 32
num_batches = training_set_size // batch_size
training_set = get_graph_batch(training_set_size, batch_size, n_nodes, p)
#edges_tr, edge_attr_tr, tensor_edge_mask_tr, adj_list_tr = get_graph_batch(training_set_size, batch_size, n_nodes, p)

#Generate Node Features and Coordinates
feat_tr = torch.ones((training_set_size * n_nodes, input_nf)) 
coord_tr = torch.normal(mean=mu_h,std=sigma_h,size=(training_set_size * n_nodes, coords))

#Generate Labels: Training Set
#l_gen_tr = LabelGenerate(coord_tr, feat_tr, edges_tr, edge_attr_tr, training_set_size, n_nodes, adj_list_tr)
#x_tr, h_tr =  l_gen_tr.get_labels(type_f2 = 'distance')



#Geenrate Test Dataset:
test_set_size = 500
test_set = get_graph_batch(test_set_size, test_set_size, n_nodes, p)
edges_test, edge_attr_test, tensor_edge_mask_test, adj_list_test = test_set[0]
feat_test = torch.ones((test_set_size * n_nodes, input_nf))
temp_coord_test = np.random.normal(mu, sigma, size=(test_set_size * n_nodes, coords))


#Train
from torch import nn, optim
import matplotlib.pyplot as plt
from random import shuffle


#Initializr NN:
mpnn = MPNN(in_node_nf=input_nf, coords=coords, in_edge_nf=1, hidden_nf=hidden_nf, n_layers=4)
egnn = EGNN(in_node_nf=input_nf, in_edge_nf=1, hidden_nf=hidden_nf, n_layers=4)


max_epoch = 20
clip_value = 1
optimizer_mpnn = optim.Adam(mpnn.parameters(), lr=0.0001)
mpnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mpnn, max_epoch)
f1 = nn.L1Loss()
f2 = nn.L1Loss()
f3 = nn.L1Loss()
f4 = nn.L1Loss()
g1 = nn.L1Loss()
g2 = nn.L1Loss()
g3 = nn.L1Loss()
g4 = nn.L1Loss()
optimizer_egnn = optim.Adam(egnn.parameters(), lr=0.0001)
egnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_egnn, max_epoch)
loss_mpnn_x_l = []
loss_mpnn_h_l = []
loss_egnn_x_l = []
loss_egnn_h_l = []
test_loss_mpnn_x_l = []
test_loss_mpnn_h_l = []
test_loss_egnn_x_l = []
test_loss_egnn_h_l = []

for i in range(max_epoch):
  shuffle(training_set)
  epoch_loss_mpnn_x_l = []
  epoch_loss_mpnn_h_l = []
  epoch_loss_egnn_x_l = []
  epoch_loss_egnn_h_l = []
  for iters in range(len(training_set)):
    mpnn_scheduler.step()
    egnn_scheduler.step()
    #Each iteration is process #batch_size number of graphs
    coord = torch.normal(mean=mu,std=sigma,size=(batch_size * n_nodes, coords))
    feat = feat_tr[iters*batch_size*n_nodes:(iters+1)*batch_size*n_nodes,:]
    edges, edge_attr, tensor_edge_mask, adj_list = training_set[iters]
    l_gen = LabelGenerate(coord, feat, edges, edge_attr, batch_size, n_nodes, adj_list)
    x_1, h_1 =  l_gen.get_labels(type_f2 = 'distance')
    h_1 = torch.tensor(h_1)
    x = torch.cat([feat, coord], axis=1)
    out_mpnn = mpnn(x, edges, edge_attr, tensor_edge_mask, n_nodes)
    #h_mpnn, x_mpnn = out_mpnn[:,0], out_mpnn[:,1:]
    h_egnn, x_egnn = egnn(feat, coord, edges, edge_attr, tensor_edge_mask, n_nodes)

    #out_egnn = torch.cat([x_egnn, h_egnn],axis=1)
    #label = torch.cat([x_1, h_1], axis=1)
    loss_mpnn_h = f1(out_mpnn[:,0].reshape(-1,1), h_1)
    loss_egnn_h = f2(h_egnn, h_1)
    loss_mpnn_x = f3(out_mpnn[:,1:], x_1)
    loss_egnn_x = f4(x_egnn, x_1)
    loss_mpnn = loss_mpnn_h + loss_mpnn_x
    loss_egnn = loss_egnn_h + loss_egnn_x

    loss_mpnn.backward()
    loss_egnn.backward()
    torch.nn.utils.clip_grad_norm_(egnn.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(mpnn.parameters(), clip_value)
    optimizer_mpnn.step()
    optimizer_egnn.step()
    epoch_loss_mpnn_x_l.append(loss_mpnn_x.detach().numpy())
    epoch_loss_mpnn_h_l.append(loss_mpnn_h.detach().numpy())
    epoch_loss_egnn_x_l.append(loss_egnn_x.detach().numpy())
    epoch_loss_egnn_h_l.append(loss_egnn_h.detach().numpy())
    
  loss_mpnn_x_l.append(np.mean(np.array(epoch_loss_mpnn_x_l)))
  loss_mpnn_h_l.append(np.mean(np.array(epoch_loss_mpnn_h_l)))
  loss_egnn_x_l.append(np.mean(np.array(epoch_loss_egnn_x_l)))
  loss_egnn_h_l.append(np.mean(np.array(epoch_loss_egnn_h_l)))
  mpnn_epoch_loss_mean = np.mean(np.array(epoch_loss_mpnn_x_l)) + np.mean(np.array(epoch_loss_mpnn_h_l))
  egnn_epoch_loss_mean = np.mean(np.array(epoch_loss_egnn_x_l)) + np.mean(np.array(epoch_loss_egnn_h_l))

  coord_test = torch.tensor(temp_coord_test, dtype=torch.float32 ,requires_grad=False)
  mpnn_test_in = torch.cat([feat_test, coord_test], axis=1)
  l_gen_test = LabelGenerate(coord_test, feat_test, edges_test, edge_attr_test, test_set_size, n_nodes, adj_list_test)
  x_test, h_test =  l_gen_test.get_labels(type_f2 = 'distance')
  h_test = torch.tensor(h_test)
  out_mpnn_test = mpnn(mpnn_test_in, edges_test, edge_attr_test, tensor_edge_mask_test, n_nodes)
  h_egnn_test, x_egnn_test = egnn(feat_test, coord_test, edges_test, edge_attr_test, tensor_edge_mask_test, n_nodes)
  test_loss_mpnn_h = g1(out_mpnn_test[:,0].reshape(-1,1), h_test)
  test_loss_egnn_h = g2(h_egnn_test, h_test)
  test_loss_mpnn_x = g3(out_mpnn_test[:,1:], x_test)
  test_loss_egnn_x = g4(x_egnn_test, x_test)
  test_loss_mpnn = test_loss_mpnn_h + test_loss_mpnn_x
  test_loss_egnn = test_loss_egnn_h + test_loss_egnn_x
  test_loss_mpnn_x_l.append(test_loss_mpnn_x)
  test_loss_mpnn_h_l.append(test_loss_mpnn_h)
  test_loss_egnn_x_l.append(test_loss_egnn_x)
  test_loss_egnn_h_l.append(test_loss_egnn_h)

  print("Epoch: {}, MPNN MSE: {:.4f}, EGNN MSE: {:.4f}, MPNN VAL: {:.4f}, EGNN VAL:{:.4f}".format((i+1), mpnn_epoch_loss_mean, egnn_epoch_loss_mean, test_loss_mpnn, test_loss_egnn))


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
# ax1.plot(np.array(loss_mpnn_x_l) + np.array(loss_mpnn_h_l))
# ax1.plot(np.array(loss_egnn_x_l) + np.array(loss_egnn_h_l))
# ax1.set_title("30000")
# ax1.legend(['MPNN', 'EGNN'])
# ax1.set(xlabel = "Epochs", ylabel = "Training Loss")
# #plt.savefig("Label Function: distance, Training Set: 100000, total, Expt 1.png")
# ax2.plot(np.array(test_loss_mpnn_x_l) + np.array(test_loss_mpnn_h_l))
# ax2.plot(np.array(test_loss_egnn_x_l) + np.array(test_loss_egnn_h_l))
# ax2.set_title("30000")
# ax2.legend(['MPNN', 'EGNN'])
# ax2.set(xlabel = "Epochs", ylabel = "Test Loss")
# #ax2.ylabel("Test Loss")
# #plt.savefig("Label Function: distance, Test Set:100000 - 500, total, Expt 1.png")
# fig.tight_layout()
# plt.savefig("Expt 2b - 30000.png")

