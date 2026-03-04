import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from core.models.edge import EdgeEmbedder
from core.models.node import NodeEmbedder
from core.modules.common.layers import sample_from, clampped_one_hot
from core.models.ga import GAEncoder
from core.modules.protein.constants import BBHeavyAtom, max_num_heavyatoms
from core.modules.common.geometry import construct_3d_basis
from core.utils.data import  PaddingCollate
from core.modules.so3.dist import uniform_so3
from core.dataset import all_atom, so3_utils

from core.dataset.residue_constants import ANGLE_MU, ANGLE_SIGMA2, ANGLE_WEIGHT
from core.models.torsion import torsions_mask
from core.models.bfn_base import BFNBase
# from sklearn.mixture import GaussianMixture

collate_fn = PaddingCollate(eight=False)

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class BFNModel(BFNBase):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.model.encoder
        self._interpolant_cfg = cfg.model.interpolant

        self.node_embedder = NodeEmbedder(cfg.model.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.model.encoder.edge_embed_size,max_num_heavyatoms)
        self.ga_encoder = GAEncoder(cfg.model.encoder.ipa)

        self.sample_trans = self._interpolant_cfg.sample_trans
        self.sample_rots = self._interpolant_cfg.sample_rots
        self.sample_torus = self._interpolant_cfg.sample_torus
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        
        # BFN parameters
        self.sigma1_coord=cfg.sigma1_coord
        self.lambda1_rot=cfg.lambda1_rot
        self.sigma1_angle=cfg.sigma1_angle
        self.beta1=cfg.beta1
        self.use_discrete_t=cfg.use_discrete_t
        self.discrete_steps=cfg.discrete_steps
        self.t_min=cfg.t_min
        self.destination_prediction = cfg.destination_prediction
        self.sampling_strategy = cfg.sampling_strategy
    
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if (self.sample_trans or self.sample_rots) else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        
        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed
    
    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def seq_to_onehot(self,seqs):
        return clampped_one_hot(seqs, self.K).float() # (B,L,K)
    
    def forward(self, batch, t=None):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask

        # prepare for denoise
        trans_1_c = trans_1 # already centered when constructing dataset
        seqs_1_onehot = self.seq_to_onehot(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_onehot, dim=-1)

        with torch.no_grad():
            if t is None:
                t = torch.rand((num_batch,1), device=batch['aa'].device) 
            if self.sample_trans:
                # NOTE: 1. Bayesian Flow p_F(θ|x;t), obtain input parameters θ
                # continuous ~ N(μ | γ(t)x, γ(t)(1 − γ(t))I)
                trans_mu, trans_gamma = self.trans_bayesian_update(
                    t, sigma1=self.sigma1_coord, x=trans_1_c
                )  # [N, 3], [N, 1]
                trans_t_c = torch.where(batch['generate_mask'][...,None],trans_mu,trans_1_c)
            else:
                trans_t_c = trans_1_c.detach().clone()
                
            if self.sample_rots:  
                # temp_i = (t**2*self.lambda1_rot/100*4000).int().repeat(1,rotmats_1.shape[1]).flatten().cpu().numpy()  # discrete interval [1,N]
                # sampel_idx = np.random.randint(0,10000, size=batch['aa'].shape).flatten()
                # matrix_fisher = self.rot_ref_array.lookup(temp_i, sampel_idx).to(batch['aa'].device).reshape(num_batch, num_res,3,3)
                matrix_fisher = torch.stack([self.sample_matrix_fisher_mixed(self.get_lambdat(t_value, self.lambda1_rot)**2, n_samples=num_res, device=batch['aa'].device) for t_value in t.squeeze()], dim=0)
                rotmats_mu = torch.matmul(rotmats_1, matrix_fisher)
                # rotmats_mu = self.rots_bayesian_update(
                #     t, self.lambda1_rot, rotmats_1
                # )  # [N, 3, 3], [N, 1, 1]
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_mu,rotmats_1)
            else:
                rotmats_t = rotmats_1.detach().clone()
            
            if self.sample_sequence:
                # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
                seqs_theta = self.discrete_var_bayesian_update(
                    t, beta1=self.beta1, x=seqs_1_onehot, K=self.K
                )  # [N, K]
                seqs_t = torch.where(batch['generate_mask'][...,None],seqs_theta,seqs_1_onehot)
                res_angle_mask = torsions_mask.to(batch['aa'].device)
                res_angle_mask = res_angle_mask[seqs_t.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
                res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1) # (B,L,15)
                res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
            else:
                seqs_t = seqs_1_onehot.detach().clone()
                res_angle_mask = torsions_mask.to(batch['aa'].device)
                res_angle_mask = res_angle_mask[seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
                res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1) # (B,L,15)
                res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)

            
            if self.sample_torus:
                # NOTE: precompute mu_k, sigma_k, pi_k
                i = (t*self.discrete_steps).int().repeat(1,angles_1.shape[1],angles_1.shape[2]).flatten().cpu().numpy()  # discrete interval [1,N]
                sampel_idx = np.random.randint(0,1000, size=angles_1.shape).flatten()
                i_mu_ro_pi = self.gmm_ref_array.lookup((angles_1.detach().clone()*180/torch.pi).int().flatten().cpu().numpy(), i, sampel_idx).to(batch['aa'].device) 
                mu, _, pi = i_mu_ro_pi[:,0], i_mu_ro_pi[:,1], i_mu_ro_pi[:,2]
                
                angles_mu = mu.reshape(num_batch,num_res,-1) #(N, L, 15)
                angles_pi = pi.reshape(num_batch,num_res,-1) #(N, L, 15)
                angles_1_pi = torch.ones_like(angles_pi)/3 # we have 3 components
                
                # # # WARN: unimodal_gaussian4torus
                # angles_mu, angles_pi = self.unimodal_gaussian4torus(t, 0.2, angles_1)
                
                angles_t_mu = torch.where(batch['generate_mask'][...,None],angles_mu,angles_1.repeat_interleave(3, dim=-1))
                angles_t_mu = angles_t_mu*res_angle_mask
                angles_t_pi = torch.where(batch['generate_mask'][...,None],angles_pi,angles_1_pi)
                angles_t_pi = angles_t_pi*res_angle_mask
            else:
                angles_t_mu = angles_1.detach().clone().repeat_interleave(3, dim=-1)  # [N, 15]
                angles_t_mu = angles_t_mu*res_angle_mask
                angles_t_pi = torch.ones_like(angles_t_mu)/3
                angles_t_pi = angles_t_pi*res_angle_mask
            
        # denoise
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob  = self.ga_encoder(t, rotmats_t, trans_t_c, angles_t_mu, angles_t_pi, seqs_t, node_embed, edge_embed, gen_mask, res_mask)
        
        pred_seqs_1 = F.softmax(pred_seqs_1_prob,dim=-1)
        pred_seqs_1 = torch.where(batch['generate_mask'][...,None],pred_seqs_1,seqs_1_prob)

        if self.use_discrete_t:
            # NOTE: 
            i = (t * self.discrete_steps).int() + 1  # discrete interval [1,N]
            
            if self.sample_trans:
                trans_closs = self.dtime4continuous_loss(
                    i=i,
                    N=self.discrete_steps,
                    sigma1=self.sigma1_coord,
                    x_pred=pred_trans_1,
                    x=trans_1_c,
                    mask=batch['generate_mask'],
                )
                # bb aux loss
                gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3] 
                pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
                bb_atom_loss = torch.sum(
                    (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
                    dim=(-1, -2, -3)
                ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
                bb_atom_loss = torch.mean(bb_atom_loss)
            else:
                trans_closs = torch.tensor(0.0, device=batch['aa'].device)
                bb_atom_loss = torch.tensor(0.0, device=batch['aa'].device)
            
            if self.sample_rots:
                # gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
                # pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
                
                rots_closs = self.dtime4so3_loss(
                    i=i,
                    N=self.discrete_steps,
                    lambda1=self.lambda1_rot,
                    x_pred=pred_rotmats_1,
                    x=rotmats_1,
                    mask=batch['generate_mask'],
                )
                # rots_closs = self.dtime4matrix_fisher_kl(
                #     i=i,
                #     N=self.discrete_steps,
                #     lambda1=self.lambda1_rot,
                #     x_pred=pred_rotmats_1,
                #     x=rotmats_1,
                #     mask=batch['generate_mask'],
                # )
            else:
                rots_closs = torch.tensor(0.0, device=batch['aa'].device)
            
            if self.sample_torus:
                angle_mask_loss = torsions_mask.to(batch['aa'].device)
                angle_mask_loss = angle_mask_loss[seqs_1_prob.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
                angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
                
                torsion_closs = self.dtime4continuous_angle_loss(
                    i=i,
                    N=self.discrete_steps,
                    sigma1=self.sigma1_angle,
                    x_pred=pred_angles_1,
                    x=angles_1,
                    mask=angle_mask_loss,
                )
                
                # angle aux loss
                angles_1_vec = torch.cat([torch.sin(angles_1),torch.cos(angles_1)],dim=-1)
                pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1),torch.cos(pred_angles_1)],dim=-1)
                angle_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*(angle_mask_loss.repeat(1,1,2)),dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
                angle_loss = torch.mean(angle_loss)
            else:
                torsion_closs = torch.tensor(0.0, device=batch['aa'].device)
                angle_loss = torch.tensor(0.0, device=batch['aa'].device)
            
            if self.sample_sequence:
                seqs_dloss = self.dtime4discrete_loss_prob(
                    i=i,
                    N=self.discrete_steps,
                    beta1=self.beta1,
                    one_hot_x=seqs_1_onehot,
                    p_0=pred_seqs_1,
                    K=self.K,
                    mask=batch['generate_mask'],
                )
            else:
                seqs_dloss = torch.tensor(0.0, device=batch['aa'].device)
            
        else:
            raise NotImplementedError("Continus time not implemented for BFNModel")
        
        
        return {
            "trans_loss": trans_closs,
            'rot_loss': rots_closs,
            'torsion_loss': torsion_closs,
            'seqs_loss': seqs_dloss,
            'bb_atom_loss': bb_atom_loss,
            'angle_loss': angle_loss,
        }
    
    # @torch.no_grad()
    # def sample(self, batch, num_steps, pos_norm):

    #     num_batch, num_res = batch['aa'].shape
    #     gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
    #     K = self._interpolant_cfg.seqs.num_classes
    #     rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
    #     angles_1 = angles_1.repeat_interleave(3, dim=-1)  # [N, 15]
        
    #     if self.sample_trans:
    #         # trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device)
    #         trans_0 = torch.zeros((num_batch,num_res,3), device=batch['aa'].device)
    #         trans_0,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
    #         trans_0 = torch.where(batch['generate_mask'][...,None],trans_0,trans_1)
    #     else:
    #         trans_0 = trans_1.detach().clone()
            
    #     if self.sample_rots:
    #         rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
    #         rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
    #     else:
    #         rotmats_0 = rotmats_1.detach().clone()
            
    #     seqs_1_onehot = self.seq_to_onehot(seqs_1)
        
        
    #     if self.sample_sequence:
    #         seqs_0_onehot = torch.ones((num_batch,num_res,K), device=batch['aa'].device) / K # (B,L,K)
    #         seqs_0_onehot = torch.where(batch['generate_mask'][...,None],seqs_0_onehot,seqs_1_onehot)
            
    #         res_angle_mask = torsions_mask.to(batch['aa'].device)
    #         seq_idx = torch.randint(0, K, (num_batch, num_res,), device=seqs_0_onehot.device)
    #         seq_idx = torch.where(batch['generate_mask'], seq_idx, seqs_1)
    #         res_angle_mask = res_angle_mask[seq_idx.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
    #         res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
    #         res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
    #     else:
    #         seqs_0_onehot = seqs_1_onehot.detach().clone()
    #         res_angle_mask = torsions_mask.to(batch['aa'].device)
    #         res_angle_mask = res_angle_mask[seqs_0_onehot.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
    #         res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
    #         res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
            
    #     if self.sample_torus:
    #         angles_mu_0 = torch.tensor(ANGLE_MU, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
    #         angles_pi_0 = torch.tensor(ANGLE_WEIGHT, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
            
    #         angles_mu_0 = torch.where(batch['generate_mask'][...,None],angles_mu_0,angles_1)*res_angle_mask
    #         angles_pi_0 = torch.where(batch['generate_mask'][...,None],angles_pi_0,torch.ones_like(angles_pi_0)/3)*res_angle_mask
    #     else:
    #         angles_mu_0 = angles_1.detach().clone()*res_angle_mask
    #         angles_pi_0 = torch.ones_like(angles_mu_0)/3*res_angle_mask

    #     angles_ro_0 = 1/torch.tensor(ANGLE_SIGMA2, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)*res_angle_mask
        
    #     clean_traj = []
    #     rotmats_t, trans_t, angles_mu_t, angles_pi_t, angles_ro_t, seqs_t = rotmats_0, trans_0, angles_mu_0, angles_pi_0, angles_ro_0, seqs_0_onehot

    #     # denoise loop
    #     for i in trange(1, num_steps + 1):
    #         t = torch.ones((num_batch, 1)).to(batch['aa'].device) * (i - 1) / num_steps
    #         if not self.use_discrete_t and not self.destination_prediction:
    #             t = torch.clamp(t, min=self.t_min)

    #         pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
    #         pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
            
    #         t_next = torch.ones((num_batch, 1)).to(batch['aa'].device) * i / num_steps
    #         if self.sample_trans:
    #             trans_t, _ = self.trans_bayesian_update(
    #                 t_next, sigma1=self.sigma1_coord, x=pred_trans_1)
    #             trans_t = torch.where(batch['generate_mask'][...,None], trans_t, trans_1)
    #         else:
    #             trans_t = trans_1.detach().clone()
                
    #         if self.sample_rots:
    #             matrix_fisher = torch.stack([self.sample_matrix_fisher_mixed(self.get_lambdat(t_value, self.lambda1_rot)**2, n_samples=num_res, device=batch['aa'].device) for t_value in t_next.squeeze()], dim=0)
    #             rotmats_t = torch.matmul(pred_rotmats_1, matrix_fisher)
    #             rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
    #         else:   
    #             rotmats_t = rotmats_1.detach().clone()

    #         if self.sample_sequence:
    #             seqs_t = self.discrete_var_bayesian_update(t_next, beta1=self.beta1, x=pred_seqs_1, K=K)
    #             seqs_t = torch.where(batch['generate_mask'][...,None],seqs_t,seqs_1_onehot)
    #         else:
    #             seqs_t = seqs_1_onehot.detach().clone()
                
    #         res_angle_mask = torsions_mask.to(batch['aa'].device)
    #         res_angle_mask = res_angle_mask[seqs_t.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
    #         res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1) # (B,L,15)
    #         res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
                
    #         if self.sample_torus:
    #             angles_mu_t, angles_ro_t, angles_pi_t = self.torus_bayesian_posterior(
    #                 i, num_steps, self.sigma1_angle, pred_angles_1, angles_mu_t, angles_ro_t, angles_pi_t
    #             )
    #             angles_mu_t = torch.where(batch['generate_mask'][...,None], angles_mu_t, angles_1) * res_angle_mask
    #             angles_pi_t = torch.where(batch['generate_mask'][...,None], angles_pi_t, torch.ones_like(angles_pi_t)/3) * res_angle_mask
    #             angles_ro_t = angles_ro_t * res_angle_mask
    #         else:
    #             angles_mu_t = res_angle_mask * angles_1.detach().clone()
    #             angles_pi_t = res_angle_mask * torch.ones_like(angles_mu_t)/3
    #             angles_ro_t = res_angle_mask * (angles_ro_t + self.get_y_ro(i, num_steps, self.sigma1_angle))  # [N, 15]

    #         clean_traj.append({
    #             "trans": pred_trans_1.detach().clone().cpu() * pos_norm,
    #             "rotmats": pred_rotmats_1.detach().clone().cpu(),
    #             "angles": pred_angles_1.detach().clone().cpu(),
    #             "seqs": pred_seqs_1.detach().clone().cpu(),
    #             "param_trans": trans_t.detach().clone().cpu() * pos_norm,
    #             "param_rotmats": rotmats_t.detach().clone().cpu(),
    #             "param_angles_mu": angles_mu_t.detach().clone().cpu(),
    #             "param_angles_ro": angles_ro_t.detach().clone().cpu(),
    #             "param_angles_pi": angles_pi_t.detach().clone().cpu(),
    #             "param_seqs": seqs_t.detach().clone().cpu(),
    #             "t": t.detach().clone().cpu(),
    #         })
        
    #     # final step
    #     t = torch.ones((num_batch, 1)).to(batch['aa'].device)
    #     pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
    #     pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
        
    #     if self.sample_trans:
    #         trans_final = torch.where(batch['generate_mask'][...,None], pred_trans_1, trans_1)
    #     else:
    #         trans_final = trans_1.detach().clone()
            
    #     if self.sample_rots:
    #         rotmats_final = torch.where(batch['generate_mask'][...,None,None], pred_rotmats_1, rotmats_1)
    #     else:   
    #         rotmats_final = rotmats_1.detach().clone()

    #     if self.sample_sequence:
    #         seqs_final = torch.where(batch['generate_mask'][...,None],pred_seqs_1, seqs_1_onehot)
    #     else:
    #         seqs_final = seqs_1_onehot.detach().clone()    

    #     res_angle_mask = torsions_mask.to(batch['aa'].device)
    #     res_angle_mask = res_angle_mask[seqs_final.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
    #     res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
        
    #     angles_1 = angles_1[:,:,0::3]
    #     if self.sample_torus:
    #         angles_final = torch.where(batch['generate_mask'][...,None], pred_angles_1, angles_1) * res_angle_mask
    #     else:
    #         angles_final = res_angle_mask * angles_1.detach().clone()
            
    #     # rescale pos to original scale
    #     batch['pos_heavyatom'] = batch['pos_heavyatom'] * pos_norm
    #     clean_traj.append({
    #             "trans": trans_final.detach().clone().cpu() * pos_norm,
    #             "rotmats": rotmats_final.detach().clone().cpu(),
    #             "angles": angles_final.detach().clone().cpu(),
    #             "seqs": seqs_final.detach().clone().cpu(),
    #             "t": t.detach().clone().cpu(),
    #             "batch": batch
    #         })
    #     return clean_traj
    
    @torch.no_grad()
    def sample(self, batch, num_steps, pos_norm):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
        angles_1 = angles_1.repeat_interleave(3, dim=-1)  # [N, 15]
        
        
        self.sample_trans, self.sample_torus, self.sample_sequence = False, False, False
        
        
        if self.sample_trans:
            # trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device)
            trans_0 = torch.zeros((num_batch,num_res,3), device=batch['aa'].device)
            trans_0,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0 = torch.where(batch['generate_mask'][...,None],trans_0,trans_1)
        else:
            trans_0 = trans_1.detach().clone()
            
        if self.sample_rots:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
        else:
            rotmats_0 = rotmats_1.detach().clone()
        
        # theta_rots = torch.zeros_like(rotmats_0)
            
        seqs_1_onehot = self.seq_to_onehot(seqs_1)
        
        
        if self.sample_sequence:
            seqs_0_onehot = torch.ones((num_batch,num_res,K), device=batch['aa'].device) / K # (B,L,K)
            seqs_0_onehot = torch.where(batch['generate_mask'][...,None],seqs_0_onehot,seqs_1_onehot)
            
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            seq_idx = torch.randint(0, K, (num_batch, num_res,), device=seqs_0_onehot.device)
            seq_idx = torch.where(batch['generate_mask'], seq_idx, seqs_1)
            res_angle_mask = res_angle_mask[seq_idx.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
        else:
            seqs_0_onehot = seqs_1_onehot.detach().clone()
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            res_angle_mask = res_angle_mask[seqs_0_onehot.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
            
        if self.sample_torus:
            angles_mu_0 = torch.tensor(ANGLE_MU, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
            angles_pi_0 = torch.tensor(ANGLE_WEIGHT, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
            
            angles_mu_0 = torch.where(batch['generate_mask'][...,None],angles_mu_0,angles_1)*res_angle_mask
            angles_pi_0 = torch.where(batch['generate_mask'][...,None],angles_pi_0,torch.ones_like(angles_pi_0)/3)*res_angle_mask
        else:
            angles_mu_0 = angles_1.detach().clone()*res_angle_mask
            angles_pi_0 = torch.ones_like(angles_mu_0)/3*res_angle_mask

        angles_ro_0 = 1/torch.tensor(ANGLE_SIGMA2, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)*res_angle_mask
        
        clean_traj = []
        rotmats_t, trans_t, angles_mu_t, angles_pi_t, angles_ro_t, seqs_t = rotmats_0, trans_0, angles_mu_0, angles_pi_0, angles_ro_0, seqs_0_onehot


        # all_errors = []
        # denoise loop
        for i in trange(1, num_steps + 1):
            t = torch.ones((num_batch, 1)).to(batch['aa'].device) * (i - 1) / num_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
            
            t_next = torch.ones((num_batch, 1)).to(batch['aa'].device) * i / num_steps
            if self.sample_trans:
                trans_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_coord, x=pred_trans_1)
                trans_t = torch.where(batch['generate_mask'][...,None], trans_t, trans_1)
            else:
                trans_t = trans_1.detach().clone()
                
            if self.sample_rots:
                matrix_fisher = torch.stack([self.sample_matrix_fisher_mixed(self.get_lambdat(t_value, self.lambda1_rot)**2, n_samples=num_res, device=batch['aa'].device) for t_value in t_next.squeeze()], dim=0)
                rotmats_t = torch.matmul(pred_rotmats_1, matrix_fisher)
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
            else:   
                rotmats_t = rotmats_1.detach().clone()
                
            # theta_rots = theta_rots + rotmats_t*self.get_lambdat(t_next[...,None,None], self.lambda1_rot)**2
            
            
            # U,S, Vh = torch.linalg.svd(theta_rots, full_matrices=False)
            # mode = U@Vh   
            
            # R_rel = torch.matmul(mode.transpose(-2, -1), pred_rotmats_1)
            # rotvec = so3_utils.rotmat_to_rotvec(R_rel) 
            # dist = torch.norm(rotvec, dim=-1)
            # rots_error = (dist*batch['generate_mask']).sum(-1)/batch['generate_mask'].sum(-1)
            
            
            # all_errors.append(rots_error.mean().item())
            
            
            if self.sample_sequence:
                seqs_t = self.discrete_var_bayesian_update(t_next, beta1=self.beta1, x=pred_seqs_1, K=K)
                seqs_t = torch.where(batch['generate_mask'][...,None],seqs_t,seqs_1_onehot)
            else:
                seqs_t = seqs_1_onehot.detach().clone()
                
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            res_angle_mask = res_angle_mask[seqs_t.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1) # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
                
            if self.sample_torus:
                angles_mu_t, angles_ro_t, angles_pi_t = self.torus_bayesian_posterior(
                    i, num_steps, self.sigma1_angle, pred_angles_1, angles_mu_t, angles_ro_t, angles_pi_t
                )
                angles_mu_t = torch.where(batch['generate_mask'][...,None], angles_mu_t, angles_1) * res_angle_mask
                angles_pi_t = torch.where(batch['generate_mask'][...,None], angles_pi_t, torch.ones_like(angles_pi_t)/3) * res_angle_mask
                angles_ro_t = angles_ro_t * res_angle_mask
            else:
                angles_mu_t = res_angle_mask * angles_1.detach().clone()
                angles_pi_t = res_angle_mask * torch.ones_like(angles_mu_t)/3
                angles_ro_t = res_angle_mask * (angles_ro_t + self.get_y_ro(i, num_steps, self.sigma1_angle))  # [N, 15]

            # WARN: add theta_rots
            clean_traj.append({
                "trans": pred_trans_1.detach().clone().cpu() * pos_norm,
                "rotmats": pred_rotmats_1.detach().clone().cpu(),
                "angles": pred_angles_1.detach().clone().cpu(),
                "seqs": pred_seqs_1.detach().clone().cpu(),
                "param_trans": trans_t.detach().clone().cpu() * pos_norm,
                "param_rotmats": rotmats_t.detach().clone().cpu(),
                "param_angles_mu": angles_mu_t.detach().clone().cpu(),
                "param_angles_ro": angles_ro_t.detach().clone().cpu(),
                "param_angles_pi": angles_pi_t.detach().clone().cpu(),
                "param_seqs": seqs_t.detach().clone().cpu(),
                "t": t.detach().clone().cpu(),
            })
        
        # final step
        t = torch.ones((num_batch, 1)).to(batch['aa'].device)
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
        
        if self.sample_trans:
            trans_final = torch.where(batch['generate_mask'][...,None], pred_trans_1, trans_1)
        else:
            trans_final = trans_1.detach().clone()
            
        if self.sample_rots:
            rotmats_final = torch.where(batch['generate_mask'][...,None,None], pred_rotmats_1, rotmats_1)
        else:   
            rotmats_final = rotmats_1.detach().clone()

        if self.sample_sequence:
            seqs_final = torch.where(batch['generate_mask'][...,None],pred_seqs_1, seqs_1_onehot)
        else:
            seqs_final = seqs_1_onehot.detach().clone()    

        res_angle_mask = torsions_mask.to(batch['aa'].device)
        res_angle_mask = res_angle_mask[seqs_final.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
        
        angles_1 = angles_1[:,:,0::3]
        if self.sample_torus:
            angles_final = torch.where(batch['generate_mask'][...,None], pred_angles_1, angles_1) * res_angle_mask
        else:
            angles_final = res_angle_mask * angles_1.detach().clone()
            
        
        # U,S, Vh = torch.linalg.svd(theta_rots, full_matrices=False)
        # mode = U@Vh    
        
        # R_rel = torch.matmul(mode.transpose(-2, -1), rotmats_1)
        # rotvec = so3_utils.rotmat_to_rotvec(R_rel) 
        # dist = torch.norm(rotvec, dim=-1)
        # rots_error = (dist*batch['generate_mask']).sum(-1)/batch['generate_mask'].sum(-1)
            
        # R_rel = torch.matmul(rotmats_final.transpose(-2, -1), rotmats_1)
        # rotvec = so3_utils.rotmat_to_rotvec(R_rel) 
        # dist = torch.norm(rotvec, dim=-1)
        # rots_error2 = (dist*batch['generate_mask']).sum(-1)/batch['generate_mask'].sum(-1)
            
            
            
        # rescale pos to original scale
        batch['pos_heavyatom'] = batch['pos_heavyatom'] * pos_norm
        clean_traj.append({
                "trans": trans_final.detach().clone().cpu() * pos_norm,
                "rotmats": rotmats_final.detach().clone().cpu(),
                "angles": angles_final.detach().clone().cpu(),
                "seqs": seqs_final.detach().clone().cpu(),
                "t": t.detach().clone().cpu(),
                "batch": batch
            })
        return clean_traj
    
    @torch.no_grad()
    def fix_seq_sample(self, batch, num_steps, pos_norm):
        sample_trans=True
        sample_rots=True
        sample_torus=True
        sample_sequence=True
        
        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
        angles_1 = angles_1.repeat_interleave(3, dim=-1)  # [N, 15]
        
        if sample_trans:
            # trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device)
            trans_0 = torch.zeros((num_batch,num_res,3), device=batch['aa'].device)
            trans_0,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0 = torch.where(batch['generate_mask'][...,None],trans_0,trans_1)
        else:
            trans_0 = trans_1.detach().clone()
            
        if sample_rots:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            
        seqs_1_onehot = self.seq_to_onehot(seqs_1)
        
        if sample_sequence:
            seqs_0_onehot = torch.ones((num_batch,num_res,K), device=batch['aa'].device) / K # (B,L,K)
            seqs_0_onehot = torch.where(batch['generate_mask'][...,None],seqs_0_onehot,seqs_1_onehot)
            
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            seq_idx = torch.randint(0, K, (num_batch, num_res,), device=seqs_0_onehot.device)
            seq_idx = torch.where(batch['generate_mask'], seq_idx, seqs_1)
            res_angle_mask = res_angle_mask[seq_idx.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
        else:
            seqs_0_onehot = seqs_1_onehot.detach().clone()
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            res_angle_mask = res_angle_mask[seqs_0_onehot.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
            
        if sample_torus:
            
            angles_mu_0 = torch.tensor(ANGLE_MU, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
            angles_pi_0 = torch.tensor(ANGLE_WEIGHT, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)
            
            angles_mu_0 = torch.where(batch['generate_mask'][...,None],angles_mu_0,angles_1)*res_angle_mask
            angles_pi_0 = torch.where(batch['generate_mask'][...,None],angles_pi_0,torch.ones_like(angles_pi_0)/3)*res_angle_mask
        else:
            angles_mu_0 = angles_1.detach().clone()*res_angle_mask
            angles_pi_0 = torch.ones_like(angles_mu_0)/3*res_angle_mask

        angles_ro_0 = 1/torch.tensor(ANGLE_SIGMA2, device=batch['aa'].device).reshape(1,1,-1).repeat(num_batch, num_res, 5)*res_angle_mask
        
        rotmats_t, trans_t, angles_mu_t, angles_pi_t, angles_ro_t, seqs_t = rotmats_0, trans_0, angles_mu_0, angles_pi_0, angles_ro_0, seqs_0_onehot

        # denoise loop
        for i in range(1, num_steps + 1):
            t = torch.ones((num_batch, 1)).to(batch['aa'].device) * (i - 1) / num_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
            
            t_next = torch.ones((num_batch, 1)).to(batch['aa'].device) * i / num_steps
            if sample_trans:
                trans_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_coord, x=pred_trans_1)
                trans_t = torch.where(batch['generate_mask'][...,None], trans_t, trans_1)
            else:
                trans_t = trans_1.detach().clone()
                
            if sample_rots:
                matrix_fisher = torch.stack([self.sample_matrix_fisher_mixed(self.get_lambdat(t_value, self.lambda1_rot)**2, n_samples=num_res, device=batch['aa'].device) for t_value in t_next.squeeze()], dim=0)
                rotmats_t = torch.matmul(pred_rotmats_1, matrix_fisher)
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
            else:   
                rotmats_t = rotmats_1.detach().clone()

            
            if sample_sequence:
                seqs_t = self.discrete_var_bayesian_update(t_next, beta1=self.beta1, x=pred_seqs_1, K=K)
                seqs_t = torch.where(batch['generate_mask'][...,None],seqs_t,seqs_1_onehot)
            else:
                seqs_t = seqs_1_onehot.detach().clone()
                
            res_angle_mask = torsions_mask.to(batch['aa'].device)
            res_angle_mask = res_angle_mask[seqs_t.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            res_angle_mask = res_angle_mask.repeat_interleave(3, dim=-1).bool() # (B,L,15)
            res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
            
            if sample_torus:
                angles_mu_t, angles_ro_t, angles_pi_t = self.torus_bayesian_posterior(
                    i, num_steps, self.sigma1_angle, pred_angles_1, angles_mu_t, angles_ro_t, angles_pi_t
                )
                
                # # # WARN: unimodal_gaussian4torus
                # y_ro = self.get_y_ro(i, num_steps, self.sigma1_angle)
                # angles_mu_t, angles_pi_t = self.unimodal_gaussian4torus(t, 0.2, pred_angles_1)
                # angles_ro_t = angles_ro_t + y_ro
                
                angles_mu_t = torch.where(batch['generate_mask'][...,None], angles_mu_t, angles_1) * res_angle_mask
                angles_pi_t = torch.where(batch['generate_mask'][...,None], angles_pi_t, torch.ones_like(angles_pi_t)/3) * res_angle_mask
                angles_ro_t = angles_ro_t * res_angle_mask
            else:
                angles_mu_t = res_angle_mask * angles_1.detach().clone()
                angles_pi_t = res_angle_mask * torch.ones_like(angles_mu_t)/3
                angles_ro_t = res_angle_mask * (angles_ro_t + self.get_y_ro(i, num_steps, self.sigma1_angle))  # [N, 15]

        # final step
        t = torch.ones((num_batch, 1)).to(batch['aa'].device)
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1 = self.ga_encoder(t, rotmats_t, trans_t, angles_mu_t, angles_pi_t, seqs_t, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_seqs_1 = F.softmax(pred_seqs_1,dim=-1)
        
        if sample_trans:
            trans_final = torch.where(batch['generate_mask'][...,None], pred_trans_1, trans_1)
        else:
            trans_final = trans_1.detach().clone()
            
        if sample_rots:
            rotmats_final = torch.where(batch['generate_mask'][...,None,None], pred_rotmats_1, rotmats_1)
        else:   
            rotmats_final = rotmats_1.detach().clone()

        if sample_sequence:
            seqs_final = torch.where(batch['generate_mask'][...,None],pred_seqs_1, seqs_1_onehot)
        else:
            seqs_final = seqs_1_onehot.detach().clone()    

        res_angle_mask = torsions_mask.to(batch['aa'].device)
        res_angle_mask = res_angle_mask[seqs_final.argmax(-1).reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        res_angle_mask = torch.logical_and(batch['res_mask'][...,None].bool(),res_angle_mask)
        generate_angle_mask = torch.logical_and(batch['generate_mask'][...,None].bool(),res_angle_mask)
        
        angles_1 = angles_1[:,:,0::3]
        if sample_torus:
            angles_final = torch.where(batch['generate_mask'][...,None], pred_angles_1, angles_1) * res_angle_mask
        else:
            angles_final = res_angle_mask * angles_1.detach().clone()
            
        
        trans_error = (((trans_final-trans_1)**2).sum(dim=-1)*batch['generate_mask']).sum()/batch['generate_mask'].sum()
        
        R_rel = torch.matmul(rotmats_1.transpose(-2, -1), rotmats_final)
        rotvec = so3_utils.rotmat_to_rotvec(R_rel) 
        dist = torch.norm(rotvec, dim=-1)
        rots_error = (dist*batch['generate_mask']).sum(-1)/batch['generate_mask'].sum(-1)
        
        aar = ((seqs_final.argmax(-1) == seqs_1)*batch['generate_mask']).sum()/batch['generate_mask'].sum()
        
        # calculate chi errors
        chi_errors={}
        for i in range(5):
            chi = angles_final[:,:,i]
            chi_gt = angles_1[:,:,i]
            dist = torch.abs(chi - chi_gt)
            chi_error = (torch.min(2*torch.pi - dist, dist) * generate_angle_mask[:,:,i]).sum()/generate_angle_mask[:,:,i].sum()
            chi_errors[f'val/chi{i+1}_error'] = chi_error.item()*180/torch.pi
        
        incorrect_port=((torch.abs(angles_final - angles_1) * 180 / torch.pi > 20)* generate_angle_mask).sum()/generate_angle_mask.sum()
        chi_errors['val/incorrect_portion'] = incorrect_port.item()
        
        error = {
            'val/trans_error': trans_error.item(),
            'val/aars_error': 1-aar.item(),
            'val/rots_error': rots_error.mean().item(),
            'val/chi1_error': chi_errors['val/chi1_error'],
            'val/chi2_error': chi_errors['val/chi2_error'],
            'val/chi3_error': chi_errors['val/chi3_error'],
            'val/chi4_error': chi_errors['val/chi4_error'],
            'val/chi5_error': chi_errors['val/chi5_error'],
            'val/incorrect_portion': chi_errors['val/incorrect_portion'],
        }
        return error
