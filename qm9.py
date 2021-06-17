from qm9 import dataset
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils

from models.egnn_main import *
from models.mpnn_main import * 

def train_qm9(epoch, loader, partition='train', model_type = 'egnn'):
    
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        #Change to required property
        label = data['gap'].to(device, dtype)
        
        if model_type == 'egnn':
          pred, _ = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                      n_nodes=n_nodes)
        else:
          pred, _ = model(x=torch.cat([nodes, atom_positions], axis=1), edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, 
                         n_nodes = n_nodes)


        if partition == 'train':
            
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
        else:
            loss = loss_l1(mad * pred + meann, label)


        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        # if i % 20 == 0:
        #     #print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter']


#SETUP
#-------------------------------------------
input_nf, output_nf, hidden_nf = 5, 1, 128
max_epoch = 10
batch_size = 96
n_nodes = 6
p = 0.5
coord_dim = 3
all_dataloaders = {}
#Change to desired number of samples in training set
sr = 40000
charge_power=2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

dataloaders, charge_scale = dataset.retrieve_dataloaders(batch_size, train_sr = sr)

# compute mean and mean absolute deviation
#Change to property required
meann, mad = qm9_utils.compute_mean_mad(dataloaders, 'gap')

#Choose model as required

model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=hidden_nf, device=device,
             n_layers=7, coords_weight=1.0, attention=False, node_attr=0)

#model = MPNN(in_node_nf=input_nf, coords=coords, in_edge_nf=1, hidden_nf=hidden_nf, n_layers=7)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-16)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
loss_l1 = nn.L1Loss()

#TRAIN 
# -------------------------------------------------
res = {'epochs': [], 'test_losses': [], 'train_losses': [], 'val_losses' : [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
#sr = [10000, 20000, 50000, ]
#max_epoch = 50
val_losses = []

for epoch in range(0, 30):

    train_loss = train_qm9(epoch, dataloaders['train'], partition='train')
    res['train_losses'].append(train_loss)
    if epoch % 1 == 0:
        val_loss = train_qm9(epoch, dataloaders['valid'], partition='valid')
        test_loss = train_qm9(epoch, dataloaders['test'], partition='test')
        res['epochs'].append(epoch)
        res['test_losses'].append(test_loss)
        res['val_losses'].append(val_loss)

        if val_loss < res['best_val']:
            res['best_val'] = val_loss
            res['best_test'] = test_loss
            res['best_epoch'] = epoch
        #print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
        val_losses.append(val_loss)
        #print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
    print("Epoch done: ", epoch+1)