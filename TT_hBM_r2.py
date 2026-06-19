import torch

dtype = torch.float32
device = 'cuda'
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)

def mpo_vec(mpo,vec):
    res = torch.clone(vec).reshape(1,-1,1)
    dim = len(mpo)
    for i in range(dim):
        res = res.reshape(res.shape[0],2**i,2,2**(dim-i-1),res.shape[-1])
        res = torch.einsum('inmj,lompi->lonpj',mpo[i],res)
    return res.flatten()

def mpo_matrix(mpo,M):
    res = torch.clone(M).reshape(1,M.shape[0],M.shape[1],1)
    dim = len(mpo)
    for i in range(dim):
        res = res.reshape(res.shape[0],2**i,2,2**(dim-i-1),res.shape[-2],res.shape[-1])
        res = torch.einsum('inmj,lompbi->lonpbj',mpo[i],res)
    return res.reshape(M.shape)

def SDP_r2(mpo,h,alpha,alpha0):
    coupling = torch.cos(alpha)@mpo_vec(mpo,torch.cos(alpha)) + torch.sin(alpha)@mpo_vec(mpo,torch.sin(alpha))
    field = h@torch.cos(alpha-alpha0)
    return coupling + field

def Ising_mpo(mpo,h,S):
    coupling = torch.einsum('in,in->n',S,mpo_matrix(mpo,S))
    field = torch.einsum('i,in->n',h,S)
    return coupling + field

def gradE_r2(mpo,h,alpha,alpha0):
    coupling = 2*torch.cos(alpha)*mpo_vec(mpo,torch.sin(alpha)) - 2*torch.sin(alpha)*mpo_vec(mpo,torch.cos(alpha))
    field = -h*torch.sin(alpha-alpha0)
    return coupling+field

def armijo(mpo,h,alpha,alpha0, p, grad, eta=1.0, c=0.5, tau=0.5, max_iter=40):
    f0    = SDP_r2(mpo,h,alpha,alpha0).item()
    slope = (grad * p).sum().item()                  # directional derivative
    for _ in range(max_iter):
        if SDP_r2(mpo,h,alpha + eta * p,alpha0).item() <= f0 + c * eta * slope:
            return eta
        eta *= tau
    return eta

def cg_r2_iter(mpo,h,alpha,alpha0,p,grad):
    eta   = armijo(mpo,h,alpha,alpha0, p, grad)
    alpha = alpha + eta * p
    grad_new = gradE_r2(mpo,h,alpha,alpha0)
    delta_g  = grad_new - grad
    beta = (grad_new @ delta_g) / (grad @ grad)
    beta = max(beta.item(), 0.0)
    p = -grad_new + beta * p
    return alpha,p,grad_new,beta

def opt_loop(mpo,h,alpha,alpha0,n_iter,tol=1e-7,eps=1e-4):
    grad = gradE_r2(mpo,h,alpha,alpha0)
    p = -grad.clone()
    obj = []
    count = 100
    for i in range(n_iter):
        alpha,p,grad,beta = cg_r2_iter(mpo,h,alpha,alpha0,p,grad)

        grad_norm = grad.norm().item()
        obj.append(SDP_r2(mpo,h,alpha,alpha0).item())
        if i % 100 == 0:
            print(f"iter {i:4d} | obj {SDP_r2(mpo,h,alpha,alpha0).item():10.4f} "
                  f"| |g| {grad_norm:.2e} | beta {beta:.3f}")
        if grad_norm < tol:
            print(f"Converged at iter {i}")
            break
        if i>count and (np.abs(obj[i] - obj[i-count]))<eps:
            print(f"Converged at iter {i}")
            break
    return alpha,obj

def rounding(alpha,alpha0,nrounds):
    alpha_ref = torch.rand(nrounds)*2*np.pi
    scoupling = (torch.cos(alpha.reshape(-1,1))*torch.cos(alpha_ref.reshape(1,-1)) + torch.sin(alpha.reshape(-1,1))*torch.sin(alpha_ref.reshape(1,-1))).sign()
    sfield = (torch.cos(alpha0)*torch.cos(alpha_ref) + torch.sin(alpha0)*torch.sin(alpha_ref)).sign()
    return scoupling*sfield.reshape(1,-1)