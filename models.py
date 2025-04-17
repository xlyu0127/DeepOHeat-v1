import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from kan import ChebyKAN

# einsum keys
# support up to 8 dimensions for now
# probably there is a more elegant way to do this

c2 = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
c1 = [c+'yz' for c in c2]
c3 = [c+'xyz' for c in c2]


#########################################################################
# activations
#########################################################################
@jax.jit
def sine(x):
    return jnp.sin(x)

@jax.jit
def identity(x):
    return x

#########################################################################
# DeepOHeat models
#########################################################################
class DeepOHeat(eqx.Module):
    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module
    branch: eqx.Module
    B: jax.Array

    def __init__(self, 
                 dim,
                 branch_dim, # number of measurements in function space
                             # currently assuming that there is only 1 function
                 field_dim=1,
                 branch_depth=3,
                 branch_hidden=64,
                 trunk_depth=3,
                 trunk_hidden=64,
                 rank=64,
                 branch_activation=jax.nn.swish,
                 branch_final_activation=identity,
                 trunk_activation = jax.nn.swish,
                 trunk_final_activation = identity,
                 key=None,
                 ):

        super().__init__()
        key, subkey = jax.random.split(key)
        subkey1, subkey2 = jax.random.split(subkey)
        self.B = 2*jnp.pi*jax.random.normal(key,shape=(dim,64))
        self.trunk = eqx.filter_vmap(nn.MLP(128, rank*field_dim, trunk_hidden, trunk_depth, 
                                            activation=trunk_activation, 
                                            final_activation=trunk_final_activation, 
                                            key=subkey1)) # vmap over number of points

        self.branch = nn.MLP(branch_dim, rank*field_dim, branch_hidden, branch_depth, 
                             activation=branch_activation, 
                             final_activation=branch_final_activation, 
                             key=subkey2) # branch_dim are number of features measured in function space

        self.dim = dim
        self.branch_dim = branch_dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(self, x__f):
        # x, f = x__f
        # x should be sequence of length dim each with shape [N**num_dims, 1] (before vmap)
        # f should be shape [branch_dim] (before vmap)
        x, f = x__f
        x = jnp.concatenate(x, axis=-1)
        x = jnp.concatenate((jnp.cos(x@self.B),jnp.sin(x@self.B)),1)
        t = self.trunk(x).reshape(-1, self.field_dim, self.rank) # shape [N**num_dims, field_dim, rank]
        b = self.branch(f).reshape(self.field_dim, self.rank) # shape [field_dim, rank]
        return jnp.einsum('ijk,jk->ij', t, b, optimize='optimal') # shape [N**num_dims, field_dim]
    

class DeepOHeat_KAN(eqx.Module):
   
    dim: int
    branch_dim: int
    field_dim: int
    rank: int
    trunk: eqx.Module
    branch: eqx.Module

    def __init__(self, 
                 dim,
                 branch_dim, # number of measurements in function space
                             # currently assuming that there is only 1 function
                 field_dim=1,
                 branch_depth=3,
                 branch_hidden=64,
                 trunk_depth=3,
                 trunk_hidden=64,
                 rank=64,
                 branch_activation=jax.nn.swish,
                 branch_final_activation=identity,
                 key=None,
                 ):

        super().__init__()

        subkey1, subkey2 = jax.random.split(key)
        
        self.trunk = eqx.filter_vmap(ChebyKAN(in_size=dim, out_size=rank*field_dim, width_size=trunk_hidden, depth=trunk_depth, 
                                            key=subkey1)) # vmap over number of points

        self.branch = nn.MLP(branch_dim, rank*field_dim, branch_hidden, branch_depth, 
                             activation=branch_activation, 
                             final_activation=branch_final_activation, 
                             key=subkey2) # branch_dim are number of features measured in function space

        self.dim = dim
        self.branch_dim = branch_dim
        self.field_dim = field_dim
        self.rank = rank

    def __call__(self, x__f):
        # x, f = x__f
        # x should be sequence of length dim each with shape [N**num_dims, 1] (before vmap)
        # f should be shape [branch_dim] (before vmap)
        x, f = x__f
        x = jnp.concatenate(x, axis=-1)
        t = self.trunk(x).reshape(-1, self.field_dim, self.rank) # shape [N**num_dims, field_dim, rank]
        b = self.branch(f).reshape(self.field_dim, self.rank) # shape [field_dim, rank]
        return jnp.einsum('ijk,jk->ij', t, b, optimize='optimal') # shape [N**num_dims, field_dim]



class DeepOHeat_ST(eqx.Module):
   
    dim: int
    branch_dim: int
    field_dim: int
    trunk: eqx.Module
    branch: eqx.Module
    rank: int
    outer_product_string: str
    B: jax.Array

    def __init__(self, 
                 dim,
                 branch_dim,
                 field_dim=1,
                 branch_depth=3,
                 branch_hidden=64,
                 trunk_depth=3,
                 trunk_hidden=64,
                 rank=64,
                 branch_activation=jax.nn.swish,
                 branch_final_activation=identity,
                 trunk_activation = jax.nn.swish,
                 trunk_final_activation = identity,
                 key=None,
                 ):
        super().__init__()

        def make_ensemble(keys):
            # probably should be able to make this more efficient with vmap
            # but the issue is that each mlp might see different sized data
            mlps = []
            for i in range(len(keys)):
                mlp = eqx.filter_vmap(nn.MLP(128, rank*field_dim, trunk_hidden, trunk_depth, 
                                             activation=trunk_activation, 
                                             final_activation=trunk_final_activation, 
                                             key=keys[i])) # vmap over number of points per dim
                
                mlps.append(mlp)
            return mlps

        subkeys = jax.random.split(key, num=dim+2) # need dim separate mlps
        
        trunk = make_ensemble(subkeys[:-2])
        self.B = 2*jnp.pi*jax.random.normal(subkeys[-2],shape=(1,64))

        branch = eqx.filter_vmap(nn.MLP(branch_dim, rank*field_dim, branch_hidden, branch_depth, 
                        activation=branch_activation, 
                        final_activation=branch_final_activation, 
                        key=subkeys[-1]))# branch_dim are number of features measured in function space
        
       

        self.dim = dim
        self.field_dim = field_dim
        self.branch_dim = branch_dim
        self.trunk = trunk
        self.branch = branch
        self.rank = rank

        s1 = ''
        s2 = ''
        for i in range(dim):
            s1 = s1 + c1[i] + ','
            s2 = s2 + c2[i]
        self.outer_product_string = s1+'byz'+'->'+'b'+s2+'y' # e.g. 'iyz,jyz,byz->bijy'
        print(self.outer_product_string)

    def __call__(self, x__f, return_basis=False):
        # x, f = x__f
        # x is a sequence of length dim each with shape [N, 1] (before vmap)
        # f should be shape [branch_dim] (before vmap)
        x, f = x__f
        ts = []
        for i in range(len(x)):
            
            xi = jnp.concatenate((jnp.cos(x[i]@self.B),jnp.sin(x[i]@self.B)),1)
            ts.append(self.trunk[i](xi).reshape(-1, self.field_dim, self.rank)) # [Nx, field_dim, rank]
           
        b = self.branch(f).reshape(-1, self.field_dim, self.rank) # [Nf, field_dim, rank]

        #path_info = jnp.einsum_path(self.outer_product_string, *ts, b, optimize='optimal')
        #print(path_info)
        if return_basis == False:
            return jnp.einsum(self.outer_product_string, *ts, b, optimize='optimal') # shape [*([N]*num_dims), field_dim]
        else:
            return ts, b, jnp.einsum(self.outer_product_string, *ts, b, optimize='optimal')
        
        
class DeepOHeat_v1(eqx.Module):
   
    dim: int
    branch_dim: int
    field_dim: int
    trunk: eqx.Module
    branch: eqx.Module
    rank: int
    outer_product_string: str

    def __init__(self, 
                 dim,
                 branch_dim,
                 field_dim=1,
                 branch_depth=3,
                 branch_hidden=64,
                 trunk_depth=3,
                 trunk_hidden=64,
                 rank=64,
                 branch_activation=jax.nn.swish,
                 branch_final_activation=identity,
                 key=None,
                 ):
        super().__init__()

        def make_ensemble(keys):
            # probably should be able to make this more efficient with vmap
            # but the issue is that each mlp might see different sized data
            kans = []
            for i in range(len(keys)):
                
                kan = eqx.filter_vmap(ChebyKAN(in_size=1,out_size=rank*field_dim, width_size=trunk_hidden, depth=trunk_depth,key=keys[i]))  
             
                kans.append(kan)
            return kans

        subkeys = jax.random.split(key, num=dim+2) # need dim separate mlps
        
        trunk = make_ensemble(subkeys[:-2])

        branch = eqx.filter_vmap(nn.MLP(branch_dim, rank*field_dim, branch_hidden, branch_depth, 
                        activation=branch_activation, 
                        final_activation=branch_final_activation, 
                        key=subkeys[-1]))# branch_dim are number of features measured in function space
        # branch = eqx.filter_vmap(ChebyKAN(branch_dim,rank*field_dim,branch_hidden,branch_depth,key=subkeys[-1]))
        
       

        self.dim = dim
        self.field_dim = field_dim
        self.branch_dim = branch_dim
        self.trunk = trunk
        self.branch = branch
        self.rank = rank

        s1 = ''
        s2 = ''
        for i in range(dim):
            s1 = s1 + c1[i] + ','
            s2 = s2 + c2[i]
        self.outer_product_string = s1+'byz'+'->'+'b'+s2+'y' # e.g. 'iyz,jyz,byz->bijy'
        print(self.outer_product_string)

    def __call__(self, x__f, return_basis=False):
        # x, f = x__f
        # x is a sequence of length dim each with shape [N, 1] (before vmap)
        # f should be shape [branch_dim] (before vmap)
        x, f = x__f
        ts = []
        for i in range(len(x)):
            
            ts.append(self.trunk[i](x[i]).reshape(-1, self.field_dim, self.rank)) # [Nx, field_dim, rank]
           
        b = self.branch(f).reshape(-1, self.field_dim, self.rank) # [Nf, field_dim, rank]

        #path_info = jnp.einsum_path(self.outer_product_string, *ts, b, optimize='optimal')
        #print(path_info)
        if return_basis == False:
            return jnp.einsum(self.outer_product_string, *ts, b, optimize='optimal') # shape [*([N]*num_dims), field_dim]
        else:
            return ts, b, jnp.einsum(self.outer_product_string, *ts, b, optimize='optimal')

