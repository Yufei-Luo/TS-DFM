import os
import torch
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from   .db import num_images

class MyModel(pl.LightningModule):
    def __init__(self, num_pair_embedding, num_grid, e_mean=0.0, e_std=1.0, num_layers=2, num_gru_layers=1, gru_dropout=0.1, learning_rate = 1e-4, factor=[1,1,1,1,1], filename='output.pkl', TTA=False):
        super(MyModel, self).__init__()
        assert len(factor)==5, "length of factor should be 4"
        self.save_hyperparameters()

        self.node_embedding = nn.Embedding(100, num_pair_embedding//4)
        self.edge_embedding = nn.Linear(num_grid, num_pair_embedding//2, bias=False )

        self.update              = nn.ModuleList( [ nn.TransformerEncoder( nn.TransformerEncoderLayer(d_model=(2**i)*num_pair_embedding, nhead=8, dropout=gru_dropout), 2 ) for i in range(num_layers)] )         
        self.gru                 = nn.ModuleList( [ nn.GRU( (2**i)*num_pair_embedding, (2**i)*num_pair_embedding, num_gru_layers, bias=True, batch_first=False, dropout=gru_dropout, bidirectional=True)  for i in range(num_layers)] )         

        self.linear9             = nn.Linear( (2**num_layers)*num_pair_embedding, 9*num_pair_embedding//16) 

        self.readout_d = nn.Sequential(
#                                       nn.LayerNorm(9*num_pair_embedding//16),
#                                       nn.Dropout(dropout_d),
#                                       nn.Linear(9*num_pair_embedding//16, num_pair_embedding//16),
#                                       nn.ReLU(),
                                       nn.Linear(9*num_pair_embedding//16, 1),
                                       nn.ELU(),
                                      )

        self.readout_e = nn.Sequential(
#                                       nn.LayerNorm( 9*num_pair_embedding//16),
#                                       nn.Dropout(dropout_e),
#                                       nn.Linear(9*num_pair_embedding//16, num_pair_embedding//16),
#                                       nn.ReLU(),
                                       nn.Linear( 9*num_pair_embedding//16, 3),
                                      )

        self.learning_rate = learning_rate
        
        self.register_buffer('e_mean', torch.tensor(e_mean))
        self.register_buffer('e_std', torch.tensor(e_std))

        self.factor1=factor[0]
        self.factor2=factor[1]
        self.factor3=factor[2]
        self.factor4=factor[3]
        self.factor5=factor[4]

        self.num_layers = num_layers
        self.filename = filename

        self.TTA = TTA

    def update_pair(self, update, pair_f, mask):
        batch_size = pair_f.size(0)
        num_images_all  = mask.size(1) # 2 num_images +1
        num_edges  = pair_f.size(-2)
        pair_f = update(
                        pair_f.reshape(batch_size*num_images_all,num_edges, -1).transpose(0,1), 
                        src_key_padding_mask= (mask.reshape(batch_size*num_images_all,num_edges)==False)
                       ).transpose(0,1).reshape(batch_size, num_images_all, num_edges, -1)
        return pair_f

    def step(self, atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask):

        batch_size, num_atoms = atomic_numbers.size()
        num_images_all  = edm_mask.size(1) # 2 num_images +1
        num_edges   = edge_f.size(-2)

        edge_f = self.edge_embedding(edge_f)

        triu_indices = torch.triu_indices(num_atoms, num_atoms, device=edm_mask.device)
        node_f = self.node_embedding(atomic_numbers)
        pair_f = torch.cat([node_f.unsqueeze(-2).repeat(1,1,num_atoms,1) , node_f.unsqueeze(-3).repeat(1,num_atoms,1,1) ], -1 )[:,triu_indices[0], triu_indices[1]]

        pair_f = pair_f.unsqueeze(1).repeat(1,num_images_all,1,1)

        pair_f = torch.cat( [pair_f, edge_f], -1 )

        #####  PSI layers  ######
        for i in range(self.num_layers):
            pair_f = self.update_pair( self.update[i], pair_f, edm_mask)
            pair_f = self.gru[i]( pair_f.transpose(0,1).reshape(num_images_all, batch_size*num_edges, -1) )[0].reshape(num_images_all, batch_size, num_edges, -1).transpose(0,1)

        pair_f = self.linear9(pair_f)
        pred_edm = (1+ self.readout_d( pair_f ).squeeze(-1) ) * linear_edm

        ##### Calculate Residue ######
        res = self.calculate_residue(pred_edm[:,num_images_all//2 ], num_atoms)

        ##### Update pair_f  #######

        ##### Hessian prediction from pair_f #####
        # To make 3N-by-3N matrix, build matrix form of pair_f for reactant, product and ts 
        mat_pair_f = torch.zeros( (batch_size, 3, num_atoms, num_atoms, pair_f.size(-1) ), dtype=pair_f.dtype, device=pair_f.device)
        mat_pair_f[:, 0, triu_indices[0], triu_indices[1]] = pair_f[:,0]
        mat_pair_f[:, 0, triu_indices[1], triu_indices[0]] = pair_f[:,0]
        mat_pair_f[:,-1, triu_indices[0], triu_indices[1]] = pair_f[:,-1]
        mat_pair_f[:,-1, triu_indices[1], triu_indices[0]] = pair_f[:,-1]
        mat_pair_f[:, 1, triu_indices[0], triu_indices[1]] = pair_f[:,num_images_all//2]
        mat_pair_f[:, 1, triu_indices[1], triu_indices[0]] = pair_f[:,num_images_all//2]

        mat_pair_f_mask = torch.zeros( (batch_size, 3, num_atoms, num_atoms ), dtype=edm_mask.dtype, device=edm_mask.device)
        mat_pair_f_mask[:, 0, triu_indices[0], triu_indices[1]] = edm_mask[:,0]
        mat_pair_f_mask[:, 0, triu_indices[1], triu_indices[0]] = edm_mask[:,0]
        mat_pair_f_mask[:,-1, triu_indices[0], triu_indices[1]] = edm_mask[:,-1]
        mat_pair_f_mask[:,-1, triu_indices[1], triu_indices[0]] = edm_mask[:,-1]
        mat_pair_f_mask[:, 1, triu_indices[0], triu_indices[1]] = edm_mask[:,num_images_all//2]
        mat_pair_f_mask[:, 1, triu_indices[1], triu_indices[0]] = edm_mask[:,num_images_all//2]

        num_mat_elements = torch.sum(mat_pair_f_mask, dim=[-1,-2])

        energy = self.readout_e(torch.sum(mat_pair_f.masked_fill(mat_pair_f_mask.unsqueeze(-1)==False, 0.0), -2))
        energy = energy.masked_fill(mat_pair_f_mask.unsqueeze(-1)[:,:,0]==False, 0.0)                               
        energy = torch.sum( energy, -1)

        e_r =self.e_mean + self.e_std*( energy[:,1,0]-energy[:,0,0] )
        e_p =self.e_mean + self.e_std*( energy[:,1,0]-energy[:,2,0] )

        # reshape mat_pair_f as (batch_size, 3, 3*num_atoms, 3*num_atoms, feature_size) 
        # mat_pair_f = mat_pair_f.reshape(batch_size, 3, num_atoms, num_atoms, 3, 3, -1 ).transpose(3,4).reshape(batch_size, 3, 3*num_atoms, 3*num_atoms, -1 )

        return pred_edm, e_r, e_p, res, num_mat_elements

    def calculate_residue(self, pred_edm, dim):
        batch_size = pred_edm.size(0)
        target = torch.zeros( (batch_size, dim, dim) , dtype=pred_edm.dtype, device=pred_edm.device)
        triu_indices = torch.triu_indices(dim, dim, device=pred_edm.device)
        target[:, triu_indices[0], triu_indices[1]] = (pred_edm*pred_edm) #.masked_fill(m==False, 0.0)
        target[:, triu_indices[1], triu_indices[0]] = target[:, triu_indices[0], triu_indices[1]]


        # shift; because of numerical stability 
        I = torch.eye(target.size(-1),dtype=target.dtype, device=target.device) 
        try:
            eigvals = torch.linalg.eigvalsh( (target+I.unsqueeze(0)).float() ).type(I.dtype) -1.0
        except RuntimeError:
            with torch.no_grad():
                largest_eig = torch.lobpcg(target.float(), largest=False, method='ortho')[0]
                print(largest_eig)
                largest_eig = (largest_eig+1).reshape(batch_size, 1, 1).to(target.dtype)
            eigvals = torch.linalg.eigvalsh( target - largest_eig*I.unsqueeze(0) ) + largest_eig.squeeze(-1)

        sort_index = torch.sort( torch.abs(eigvals), -1, True)[1]
        eigvals = torch.gather(eigvals, 1, sort_index ) 
        return torch.linalg.vector_norm(eigvals[:,5:], ord=1) + torch.linalg.vector_norm(torch.sum(eigvals[:,:5], dim=-1 ), ord=1)
        #eigvals = torch.sort( torch.abs(eigvals), -1, True)[0][:,5:]
        #return torch.norm(eigvals, p=1)

    def calculate_loss(self, pred_edm, pred_e_r, pred_e_p, num_mat_elements, batch):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, B_matrices, inv_weights, edm_mask, edm_diag_mask = batch 
        
        num_images = linear_edm.size(1)//2

        m = edm_mask[:,num_images]
        num_mat_e = num_mat_elements[:,num_images]

        ts_edm_diff = true_edm[:,num_images] - pred_edm[:,num_images]
        
        ref_loss = torch.sum( 2*ts_edm_diff**2, dim=[-1] ) / num_mat_e  # PCCP (2020)
        
        r_e_loss = torch.abs(pred_e_r-true_e_r )
        p_e_loss = torch.abs(pred_e_p-true_e_p )

        ts_edm_loss = 2*torch.linalg.vector_norm( ts_edm_diff, ord=1, dim=[-1]) / num_mat_e # sum of abs

        ts_edm_mape = 2*torch.nansum( torch.abs(ts_edm_diff) / true_edm[:,num_images], dim=-1) / num_mat_e 

        return  torch.sum(ts_edm_mape), torch.sum(ts_edm_loss), torch.sum(r_e_loss), torch.sum(p_e_loss), torch.sum( ref_loss)

    def training_step(self, batch, batch_idx):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p,  B_matrices, inv_weights, edm_mask, edm_diag_mask = batch 

        batch_size = atomic_numbers.size(0)
        num_atoms  = atomic_numbers.size(1)
         
        pred_edm, pred_e_r, pred_e_p, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)

        ts_edm_mape, ts_edm_loss, r_e_loss, p_e_loss, ref_loss = self.calculate_loss(pred_edm, pred_e_r, pred_e_p,  num_mat_elements, batch)
        
        self.log('ts_edm_mape_t',  ts_edm_mape  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ts_edm_loss_t',  ts_edm_loss  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.log('e_loss_t',(r_e_loss+p_e_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)

        self.log('r_e_loss_t',     r_e_loss  /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_e_loss_t',     p_e_loss  /batch_size,    prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('res_t',      res     /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ref_t',      ref_loss/batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return self.factor1*ts_edm_loss + self.factor2*(r_e_loss + p_e_loss) + self.factor4*res
        #return self.factor1*torch.sum( ts_loss )+e_r_loss+e_p_loss+self.factor1*res*self.current_epoch/args.max_epochs

    def validation_step(self, batch, batch_idx):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, B_matrices, inv_weights, edm_mask, edm_diag_mask = batch 

        batch_size = atomic_numbers.size(0)
        num_atoms  = atomic_numbers.size(1)

        pred_edm, pred_e_r, pred_e_p, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)

        ts_edm_mape, ts_edm_loss, r_e_loss, p_e_loss, ref_loss = self.calculate_loss(pred_edm, pred_e_r, pred_e_p, num_mat_elements, batch)
        
        self.log('ts_edm_mape_v',  ts_edm_mape  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ts_edm_loss_v',  ts_edm_loss  /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.log('e_loss_v',(r_e_loss+p_e_loss)/batch_size,  prog_bar=True,logger=True, on_step=False, on_epoch=True)
        
        self.log('r_e_loss_v',     r_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('p_e_loss_v',     p_e_loss  /batch_size, prog_bar=False,logger=True, on_step=False, on_epoch=True)
        self.log('res_v',      res     /batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('ref_v',      ref_loss/batch_size, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return self.factor1*ts_edm_loss + self.factor2*(r_e_loss + p_e_loss) + self.factor4*res


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-6) } }

    def test_step(self, batch, batch_idx):
        atomic_numbers, true_edm, linear_edm, edge_f, true_e_r, true_e_p, B_matrices, inv_weights, edm_mask, edm_diag_mask, reactant_pos, product_pos, transition_state_pos, trans_energy  = batch
        batch_size = linear_edm.size(0)
        num_atoms  = atomic_numbers.size(1)
        num_images = linear_edm.size(1)//2

        pred_edm, pred_e_r, pred_e_p, res, num_mat_elements= self.step(atomic_numbers, edge_f, linear_edm, edm_mask, edm_diag_mask)

        if(self.TTA):
            original_batch_size = batch_size//2
            for i in range(original_batch_size):
                pred_edm[i] = (pred_edm[i+original_batch_size] + pred_edm[i])/2
                pred_e_r[i] = (pred_e_p[i+original_batch_size] + pred_e_r[i])/2
                pred_e_p[i] = (pred_e_r[i+original_batch_size] + pred_e_p[i])/2
            pred_edm = pred_edm[:original_batch_size]
            pred_e_r = pred_e_r[:original_batch_size]
            pred_e_p = pred_e_p[:original_batch_size]
            edm_mask = edm_mask[:original_batch_size]
        # print([ o[m] for o,m in zip(pred_edm[:,num_images],edm_mask[:,num_images]) ])
        # return [ o[m] for o,m in zip(pred_edm[:,num_images],edm_mask[:,num_images]) ],\
        #        pred_e_r, pred_e_p
        output = [ o[m] for o,m in zip(pred_edm[:,num_images],edm_mask[:,num_images]) ]
        #print(output)
        list_dim =  list(  map( lambda c: int( (-1+(1+8*c)**0.5 )/2) , [ len(o) for o in output ] ) )
        #print(list_dim)
        # list_dim = output

        return_val=[]
        for i, dim in enumerate(list_dim):
            return_val.append( torch.zeros(dim, dim, device=output[i].device, dtype=output[i].dtype) )
            triu_indices = torch.triu_indices(dim,dim, device=output[i].device)
            return_val[-1][ triu_indices[0], triu_indices[1] ] = output[i]
            return_val[-1]=return_val[-1]+return_val[-1].transpose(0,1) # symmetrize
            return_val[-1]=return_val[-1].detach().cpu()

        return_val = (return_val, pred_e_r, pred_e_p)
        return return_val
