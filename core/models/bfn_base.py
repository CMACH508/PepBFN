import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean, scatter_sum
import torch.distributions as dist
from core.modules.so3.dist import sample_matrix_fisher_torch, axis_angle_to_matrix
from core.dataset import so3_utils
from core.dataset.residue_constants import ANGLE_MU, ANGLE_SIGMA2, ANGLE_WEIGHT
import core.models.torus as torus

import numpy as np
import math

LOG2PI = np.log(2 * np.pi)


def uniform_SO3_torch(n, device='cpu'):
    """Shoemake算法采样均匀旋转矩阵 [n,3,3]"""
    u1, u2, u3 = torch.rand(3, n, device=device)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    R = torch.zeros((n,3,3), device=device)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

def exp_so3(omega):
    """指数映射: so(3) -> SO(3), omega: [N,3]"""
    theta = torch.norm(omega, dim=1, keepdim=True) + 1e-12
    k = omega / theta
    K = torch.zeros((omega.shape[0], 3, 3), device=omega.device)
    K[:,0,1],K[:,0,2],K[:,1,0],K[:,1,2],K[:,2,0],K[:,2,1] = -k[:,2],k[:,1],k[:,2],-k[:,0],-k[:,1],k[:,0]
    I = torch.eye(3, device=omega.device).unsqueeze(0)
    sin_term = torch.sin(theta)[:,None] * K
    cos_term = (1-torch.cos(theta))[:,None] * torch.bmm(K,K)
    return I + sin_term + cos_term




class BFNBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(BFNBase, self).__init__(*args, **kwargs)

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_l

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)

        return f_

    def trans_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = 1 - torch.pow(sigma1, 2 * t[:,None])  # [B]
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        # WARN: 
        mu = gamma * x
        return mu, gamma
    
    def unimodal_gaussian4torus(self, t, sigma1, x):
        """
        x: [N, D, 15]
        """
        RO_0 = 1/np.mean(ANGLE_SIGMA2)
        MU_0 = float(np.mean(ANGLE_MU))
        BETA_t = torch.pow(RO_0, 1- t[:,None])*torch.pow(sigma1, -t[:,None])  - RO_0
        RO_t = self.get_y_ro((1000*t).int(), 1000, sigma1)[:,None]
        
        mu = (BETA_t * x + MU_0*RO_0)/RO_t+ torch.randn_like(x) * torch.sqrt(BETA_t/ (RO_t**2))
        pi = torch.ones_like(mu)/3
        return mu.repeat_interleave(3, dim=-1), pi.repeat_interleave(3, dim=-1), 
    
    def rots_bayesian_update(self, t, lambda1, rotmats_1):
        """
        rotmats_1: [N, D, 3, 3]
        """
        t = t.repeat(1, rotmats_1.shape[1])
        gamma = lambda1 * t**2 # [B, 3, 3]
        matrix_fisher = sample_matrix_fisher_torch(gamma, device=rotmats_1.device)
        mu = torch.matmul(rotmats_1, matrix_fisher)  # [B, D, 3, 3]
        return mu
    
    def sample_matrix_fisher_mixed(self, lambda_val=25, n_samples=10000, device='cpu'):
        """
        从 Matrix Fisher M(λI) 采样 (混合方法: 小 λ 拒绝采样, 大 λ 李代数近似)
        返回: R_samples [N,3,3]
        """
        lambda_val = float(lambda_val)
        if lambda_val <= 26:
            # 拒绝采样
            samples = []
            batch = max(2000, n_samples * 5)
            max_density = torch.exp(torch.tensor(3.0 * lambda_val, device=device))
            total = 0
            while total < n_samples:
                R = uniform_SO3_torch(batch, device=device)
                tr = torch.einsum('bii->b', R)
                density = torch.exp(lambda_val * tr)
                u = torch.rand(batch, device=device)
                accept = R[u < density / max_density]
                if accept.numel() > 0:
                    samples.append(accept)
                    total += accept.shape[0]
            R_samples = torch.cat(samples, dim=0)[:n_samples]
        else:
            # 李代数高斯近似
            sigma = 1.0 / math.sqrt(2*lambda_val)  # 经验近似
            omega = torch.randn(n_samples, 3, device=device) * sigma
            R_samples = exp_so3(omega)
        return R_samples
    # def rots_bayesian_update(self, t, sigma1, rotmats_1):
    #     """
    #     rotmats_1: [N, D, 3, 3]
    #     """
    #     # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
    #     gamma = 1 - torch.pow(sigma1, 2 * t[:,None])  # [B]
    #     # rotmats_0 = normal_so3(rotmats_1.shape[0], rotmats_1.shape[1], device=rotmats_1.device)
    #     # mu = so3_utils.geodesic_t(gamma[..., None, None], rotmats_1, rotmats_0)
    #     # mu = so3_utils.geodesic_t(gamma, rotmats_1, torch.zeros_like(rotmats_1)) + rotmats_0 * torch.sqrt(gamma * (1 - gamma))[..., None]  # [B, D, 3, 3]
    #     mu = so3_utils.SO3_gaussian(gamma, rotmats_1)
    #     return mu, gamma
    
    def get_torus_beta_t(self, t, sigma1):
        ro_0=1/ANGLE_SIGMA2[0]
        beta_t = (ro_0**(1-t))*(sigma1**(-2*t)) - ro_0
        return beta_t[...,None], ro_0
    
    def get_y_ro(self, i, N, sigma1):
        RO_0 = 1/np.mean(ANGLE_SIGMA2)
        y_ro = (RO_0**(1-i/N))*(sigma1**(-2*i/N))*(1-(RO_0*(sigma1**2))**(1/N))
        return y_ro
    
    def sample_angles_y_given_angles_1(self, i, N, sigma1, angles_1):
        """
        angles_1: [N, D, 15]
        """
        y_ro = self.get_y_ro(i, N, sigma1)
        y = angles_1 + torch.randn_like(angles_1) * np.sqrt(1/y_ro)
        # WARN: remove nosie
        y = angles_1
        return y.repeat_interleave(3,dim=-1), y_ro
    
    def compute_posterior_weights(self, y, prior_mu, prior_ro, prior_pi, y_ro):
        """
        计算高斯混合后验分布的权重。
        
        参数:
            y: 观测值
            prior_mu: 各分量的先验均值 [μ_1, ..., μ_K]
            prior_ro: 各分量的先验精度 [λ_1, ..., λ_K]
            prior_pi: 各分量的先验权重 [π_1, ..., π_K]
            y_ro: 观测精度 y_ro
        
        返回:
            posterior_weights: 后验权重 [π̃_1, ..., π̃_K]
        """
        # 计算边缘似然的方差：σ_k^2 + σ_y^2 = 1/λ_k + 1/λ_y
        marginal_variance = 1.0 / (prior_ro+1e-8) + 1.0 / y_ro
        
        # 计算各分量的边缘似然
        m = dist.normal.Normal(prior_mu, torch.sqrt(marginal_variance))
        logp = m.log_prob(y).reshape(prior_mu.shape[0], prior_mu.shape[1], 5, 3)
        marginal_likelihood = F.softmax(logp, dim=-1)
        
        unnormalized_weights = prior_pi.reshape(prior_mu.shape[0], prior_mu.shape[1], 5, 3) * marginal_likelihood
        # 归一化
        posterior_weights = unnormalized_weights / (unnormalized_weights.sum(dim=-1, keepdim=True)+1e-8)
        
        return posterior_weights.reshape(prior_mu.shape[0], prior_mu.shape[1], 15)
    
    def torus_bayesian_posterior(self, i, N, sigma1, angles_1, prior_mu, prior_ro, prior_pi):
        """
        angles_1: [N, D, 15]
        """
        y, y_ro = self.sample_angles_y_given_angles_1(i, N, sigma1, angles_1)
        posterior_ro = prior_ro+y_ro
        posterior_mu = (prior_mu*prior_ro+y*y_ro)/posterior_ro
        posterior_pi = self.compute_posterior_weights(
            y, prior_mu, prior_ro, prior_pi, y_ro
        )
        return posterior_mu, posterior_ro, posterior_pi

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        # Eq.(182): β(t) = t**2 β(1)
        beta = beta1 * (t[:,None]**2)  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        # WARN:
        y = mean
        theta = F.softmax(y, dim=-1)
        return theta

    def discreteised_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        Note, this is identical to the continuous_var_bayesian_update
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    # def ctime4continuous_loss(self, t, sigma1, x_pred, x, segment_ids=None):
    #     # Eq.(101): L∞(x) = −ln(σ1) * E_{t∼U (0,1), p_F(θ|x;t)} [|x − x_hat(θ,t)|**2 / (σ_1**2)**t]
    #     if segment_ids is not None:
    #         loss = scatter_mean(
    #             torch.pow(sigma1, -2 * t.view(-1))
    #             * ((x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)),
    #             segment_ids,
    #             dim=0,
    #         )
    #     else:
    #         loss = torch.pow(sigma1, -2 * t.view(-1)) * (x_pred - x).view(
    #             x.shape[0], -1
    #         ).abs().pow(2).sum(dim=1)
    #     return -torch.log(sigma1) * loss
    
    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, mask):
        weight = N * (1 - sigma1**(2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        loss = weight.view(-1) * (((x_pred - x) ** 2).sum(-1)*mask).sum(-1)/mask.sum(-1)
        return loss.mean()
    
    # def dtime4so3_loss(self, i, N, lambda1, x_pred, x, mask):
    #     weight = lambda1 * (i / N)**2
    #     R_rel = torch.matmul(x.transpose(-2, -1), x_pred)
    #     rotvec = so3_utils.rotmat_to_rotvec(R_rel) 
    #     dist = torch.norm(rotvec, dim=-1)
    #     dist = weight * dist 
    #     dist = (dist*mask).sum(-1)/mask.sum(-1)
    #     return N * dist.mean()
    def get_lambdat(self, t, lambda1):
        return lambda1 / (math.exp(2) - 1) * (torch.exp(2 * t) - 1)
    
    
    def dtime4so3_loss(self, i, N, lambda1, x_pred, x, mask):
        weight = self.get_lambdat(i/N, lambda1)
        weight = weight * (1-1/(2*weight+1))
        
        R_rel = torch.matmul(x.transpose(-2, -1), x_pred)
        dist = 3-torch.einsum('njii->nj', R_rel)
        dist = weight * dist 
        dist = (dist*mask).sum(-1)/mask.sum(-1)
        return N * dist.mean()

    def dtime4matrix_fisher_kl(self, i, N, lambda1, x_pred, x, mask):
        """
        计算 Matrix Fisher 分布 KL: M(λ R_a) || M(λ R_b)
        R_a, R_b: shape [..., 3, 3]
        lambda_val: scalar
        """
        lambda_val = lambda1 * (i / N)**2  # [B, 1, 1]
        def approx_mu(lam):
            return 1 - 1.5 / lam + 3.75 / lam**2
        
        mu = approx_mu(lambda_val)
        Rt = torch.matmul(x_pred.transpose(-2, -1), x)
        trace = torch.einsum('...ii->...', Rt)  # trace(R_b^T R_a)
        kl = lambda_val * mu * (3 - trace)
        kl = N * kl * mask / mask.sum()
        return kl.mean()

    def dtime4continuous_angle_loss(self, i, N, sigma1, x_pred, x, mask):
        weight = N/2 * self.get_y_ro(i, N, sigma1)
        loss = weight.squeeze() * (((x_pred - x) ** 2)*mask).sum((-1,-2))/(mask.sum((-1,-2))+1e-8)
        return loss.mean()
        

    # def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K, segment_ids=None):
    #     # Eq.(205): L∞(x) = Kβ(1) E_{t∼U (0,1), p_F (θ|x,t)} [t|e_x − e_hat(θ, t)|**2,
    #     # where e_hat(θ, t) = (\sum_k p_O^(1) (k | θ; t)e_k, ..., \sum_k p_O^(D) (k | θ; t)e_k)
    #     e_x = one_hot_x  # [N, K]
    #     e_hat = p_0  # (N, K)
    #     assert e_x.size() == e_hat.size()
    #     if segment_ids is not None:
    #         L_infinity = scatter_mean(
    #             K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1),
    #             segment_ids,
    #             dim=0,
    #         )
    #     else:
    #         L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
    #     return L_infinity

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, mask=None
    ):
        # this is based on the official implementation of BFN.
        target_x = one_hot_x  # [B, T, K]
        e_hat = p_0  # (B, T, K)
        alpha = beta1 * (2 * i - 1) / N**2  # [B, 1]
        alpha = alpha.view(-1, 1) # [B, 1]
        classes = torch.arange(K, device=target_x.device).long()[None,None,...]  # [ 1, 1, K]
        e_x = F.one_hot(classes.long(), K) #[1, 1, K, K]
        # print(e_x.shape)
        receiver_components = dist.Independent(
            dist.Normal(
                alpha[...,None,None] * ((K * e_x) - 1), # [B, 1, K, K]
                (K * alpha[...,None,None]) ** 0.5, # [B, 1, 1, 1]
            ),
            1,
        )  # [B,T, K, K]
        receiver_mix_distribution = dist.Categorical(probs=e_hat)  # [B, T, K]
        receiver_dist = dist.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [B, T, K, K]
        sender_dist = dist.Independent( dist.Normal(
            alpha[...,None]* ((K * target_x) - 1), ((K * alpha)[...,None] ** 0.5)
        ),1)  # [B, T, K, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        loss = N * ((sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0)*mask).sum() / mask.sum()
        # loss = (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0)[mask].sum() / mask.sum()
        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
    #     # i in {1,n}
    #     # Algorithm 7 in BFN
    #     e_x = one_hot_x  # [D, K]
    #     e_hat = p_0  # (D, K)
    #     assert e_x.size() == e_hat.size()
    #     alpha = beta1 * (2 * i - 1) / N**2  # [D]

    #     # print(alpha.shape)
    #     mean_ = alpha * (K * e_x - 1)  # [D, K]
    #     std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
    #     eps = torch.randn_like(mean_)  # [D,K,]
    #     y_ = mean_ + std_ * eps
    #     # modify this line:
    #     matrix_ek = torch.eye(K, K).unsqueeze(0).to(e_x.device)
    #     matrix_ek.repeat(alpha.size(0), 1, 1)  # [D,K,K]
    #     mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [D,K,K]
    #     std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
    #     likelihood = (
    #         torch.exp(
    #             -((y_.unsqueeze(1).repeat(1, K, 1) - mean_matrix) ** 2)
    #             / (2 * std_matrix**2)
    #         )
    #         / (std_matrix * np.sqrt(2 * np.pi))
    #     ).prod(
    #         -1
    #     )  # [D,K]

    #     if segment_ids is not None:
    #         L_N = -scatter_mean(
    #             torch.log((likelihood * e_hat).sum(dim=-1)), segment_ids, dim=0
    #         )
    #     else:
    #         L_N = -torch.log((likelihood * e_hat).sum(dim=-1))  # [D]
    #     # print(L_N.shape)
    #     #
    #     return N * L_N

    # def dtime4discrete_loss_gjj(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
    #     # i in {1,n}
    #     # Algorithm 7 in BFN
    #     e_x = one_hot_x  # [D, K]
    #     e_hat = p_0  # (D, K)
    #     assert e_x.size() == e_hat.size()
    #     alpha = beta1 * (2 * i - 1) / N**2  # [D]

    #     # print(alpha.shape)
    #     mean_ = alpha * (K * e_x - 1)  # [D, K]
    #     std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
    #     eps = torch.randn_like(mean_)  # [D,K,]
    #     y_ = mean_ + std_ * eps  # [D, K]
    #     # modify this line:
    #     matrix_ek = torch.eye(K, K).to(e_x.device)  # [K, K]
    #     mean_matrix = K * matrix_ek - 1  # [K, K]
    #     std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
    #     _log_gaussians = (  # [D, K]
    #         (-0.5 * LOG2PI - torch.log(std_matrix))
    #         - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
    #     ).sum(-1)

    #     _inner_log_likelihood = torch.log(
    #         torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1)
    #     )  # (D,)

    #     _inner_log_likelihood = torch.log(e_hat) + _log_gaussians  # [D, K]
    #     log_likelihood = torch.logsumexp(_inner_log_likelihood, dim=-1)  # [D]

    #     if segment_ids is not None:
    #         L_N = -scatter_mean(log_likelihood, segment_ids, dim=0)
    #     else:
    #         L_N = -log_likelihood.sum(dim=-1)  # [D]
    #     # print(L_N.shape)
    #     #
    #     return N * L_N

    # def ctime4discreteised_loss(self, t, sigma1, x_pred, x, segment_ids=None):
    #     if segment_ids is not None:
    #         loss = scatter_sum(
    #             (x_pred - x).view(x.shape[0], -1).abs().pow(2), segment_ids, dim=0
    #         )
    #     else:
    #         raise NotImplementedError
    #         loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
    #     return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

