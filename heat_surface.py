import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import argparse
import optax
from functools import partial
from models import DeepOHeat, DeepOHeat_KAN, DeepOHeat_ST, DeepOHeat_v1
from jax import vmap
from hvp import hvp_fwdfwd, hvp_fwdrev
from train import train_loop, update
from eval import eval_heat3d

@jax.jit
def create_mesh(xi_batch, yi_batch, zi_batch):
    return jnp.meshgrid(xi_batch.ravel(), yi_batch.ravel(), zi_batch.ravel(), indexing='ij')

batch = 50
x = jnp.stack([jnp.linspace(0, 1, 21).reshape(-1,1)]*batch)
y = jnp.stack([jnp.linspace(0, 1, 21).reshape(-1,1)]*batch)
z = jnp.stack([jnp.linspace(0, 0.5, 11).reshape(-1,1)]*batch)
x_mesh, y_mesh, z_mesh = vmap(create_mesh, in_axes=(0, 0, 0))(x, y, z)
x = x_mesh.reshape(batch, 21*21*11, 1)
y = y_mesh.reshape(batch, 21*21*11, 1)
z = z_mesh.reshape(batch, 21*21*11, 1)
top_idx = z == 0.5
bottom_idx = z == 0
left_idx = x == 0
right_idx = x == 1
front_idx = y == 0
back_idx = y == 1
inside_idx = ~(top_idx | bottom_idx | left_idx | right_idx | front_idx | back_idx)


#########################################################################
# Loss functions
#########################################################################
@eqx.filter_jit
def apply_model_deepoheat(model, xc, yc, zc, fc, lam_b=1.):
    
    def PDE_loss(model, x, y, z, f):
        # N_f
        nf = f.shape[0]
        # compute u
        u = model(((x, y, z), f))
        # tangent vector du/du
        v = jnp.ones(u.shape)
        # 1st, 2nd derivatives of u
        ux, uxx = hvp_fwdrev(lambda x: model(((x, y, z), f)), (x,), (v,), True)
        uy, uyy = hvp_fwdrev(lambda y: model(((x, y, z), f)), (y,), (v,), True)
        uz, uzz = hvp_fwdrev(lambda z: model(((x, y, z), f)), (z,), (v,), True)
        # PDE residual
        pde_res = jnp.mean((uxx[inside_idx] + uyy[inside_idx] + uzz[inside_idx])**2)
        # top surface
        bc_top = jnp.mean((uz[top_idx].reshape(nf,-1) - f)**2)
        # bottom surface
        bc_bottom = jnp.mean((u[bottom_idx] - 0.2- 0.2*uz[bottom_idx])**2)
        # other surfaces
        bc_other = jnp.mean((uy[front_idx])**2) + jnp.mean((uy[back_idx])**2) + jnp.mean((ux[left_idx])**2) + jnp.mean((ux[right_idx])**2)
        
        return pde_res + lam_b*(bc_top + bc_bottom + bc_other)

    
    

    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
                       

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient



@eqx.filter_jit
def apply_model_deepoheat_st(model, xc, yc, zc, fc, lam_b=1.):

    def PDE_loss(model, x, y, z, f):
        # compute u
        u = model(((x, y, z), f))
    
        # tangent vector dx/dx dy/dy dz/dz
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        v_z = jnp.ones(z.shape)

        # 1st, 2nd derivatives of u
        ux, uxx = hvp_fwdfwd(lambda x: model(((x, y, z), f)), (x,), (v_x,), True)
        uy, uyy = hvp_fwdfwd(lambda y: model(((x, y, z), f)), (y,), (v_y,), True)
        uz, uzz = hvp_fwdfwd(lambda z: model(((x, y, z), f)), (z,), (v_z,), True)

        # PDE residual, only interior points are used
        pde_res = jnp.mean((uxx + uyy + uzz)**2)

        # top surface
        bc_top = jnp.mean((uz[:,:,:,-1,:] - f.reshape(-1,21,21,1))**2)
        # bottom surface
        bc_bottom = jnp.mean((u[:,:,:,0,:] - 0.2 - 0.2*uz[:,:,:,0,:])**2)
        # other_surfaces
        bc_other = jnp.mean((uy[:,:,0,:,:])**2) + jnp.mean((uy[:,:,-1,:,:])**2) + jnp.mean((ux[:,0,:,:,:])**2) + jnp.mean((ux[:,-1,:,:,:])**2)
      

        return pde_res + lam_b*(bc_top + bc_bottom + bc_other)


    # isolate loss func from redundant arguments
    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
                       

    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient


#########################################################################
# Train generators
#########################################################################
@partial(jax.jit, static_argnums=(1,2))
def deepoheat_train_generator(fs, batch, nc, key):
    
    nx = nc
    ny = nc
    nz = (nx // 2) + 1
    key, _ = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx,:]
    x = jnp.stack([jnp.linspace(0, 1, nx).reshape(-1,1)]*batch)
    y = jnp.stack([jnp.linspace(0, 1, ny).reshape(-1,1)]*batch)
    z = jnp.stack([jnp.linspace(0, 0.5, nz).reshape(-1,1)]*batch)
    x_mesh, y_mesh, z_mesh = vmap(create_mesh, in_axes=(0, 0, 0))(x, y, z)
    xc = x_mesh.reshape(batch, nx*ny*nz, 1)
    yc = y_mesh.reshape(batch, nx*ny*nz, 1)
    zc = z_mesh.reshape(batch, nx*ny*nz, 1)
   
    return xc, yc, zc, fc


@partial(jax.jit, static_argnums=(1,2))
def deepoheat_st_train_generator(fs, batch, nc, key):
    
    nx = nc
    ny = nc
    nz = (nx // 2) + 1
    key, _ = jax.random.split(key)
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False)
    fc = fs[idx,:]
    xc = jnp.linspace(0, 1, nx).reshape(-1,1)
    yc = jnp.linspace(0, 1, ny).reshape(-1,1)
    zc = jnp.linspace(0, 0.5, nz).reshape(-1,1)
    
    return xc, yc, zc, fc


#########################################################################
# Test generators
#########################################################################
@jax.jit
def deepoheat_test_generator(fs, u):
    nf = fs.shape[0]
    x = jnp.stack([jnp.linspace(0, 1, 101).reshape(-1,1)]*nf)
    y = jnp.stack([jnp.linspace(0, 1, 101).reshape(-1,1)]*nf)
    z = jnp.stack([jnp.linspace(0, 0.5, 51).reshape(-1,1)]*nf)
    x_mesh, y_mesh, z_mesh = vmap(create_mesh, in_axes=(0, 0, 0))(x, y, z)
    x = x_mesh.reshape(nf, 101*101*51, 1)
    y = y_mesh.reshape(nf, 101*101*51, 1)
    z = z_mesh.reshape(nf, 101*101*51, 1)
    return x, y, z, fs, u


@jax.jit
def deepoheat_st_test_generator(fs, u):

    x = jnp.linspace(0, 1, 101).reshape(-1,1)
    y = jnp.linspace(0, 1, 101).reshape(-1,1)
    z = jnp.linspace(0, 0.5, 51).reshape(-1,1)
    return x, y, z, fs, u


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')
    parser.add_argument('--model_name', type=str, default='DeepOHeat_v1', choices=['DeepOHeat', 'DeepOHeat_KAN', 
                                                                                          'DeepOHeat_ST', 'DeepOHeat_v1'])
    parser.add_argument('--device_name', type=int, default=0, choices=[0, 1], help='GPU device')

    # training data settings
    parser.add_argument('--nc', type=int, default=21, help='the number of input points for each axis')
    parser.add_argument('--batch', type=int, default=50, help='the number of train functions')
    
    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='training epochs')
    parser.add_argument('--log_epoch', type=int, default=100, help='log the loss every chosen epochs')

    # model settings
    parser.add_argument('--dim', type=int, default=3, help='the input size')
    parser.add_argument('--branch_dim', type=int, default=21**2, help='the number of sensors for indentifying an input function')
    parser.add_argument('--field_dim', type=int, default=1, help='the dimension of the output field')
    parser.add_argument('--branch_depth', type=int, default=8, help='the number of hidden layers, including the output layer')
    parser.add_argument('--branch_hidden', type=int, default=256, help='the size of each hidden layer')
    parser.add_argument('--trunk_depth', type=int, default=3, help='the number of hidden layers, including the output layer')
    parser.add_argument('--trunk_hidden', type=int, default=64, help='the size of each hidden layer')
    parser.add_argument('--r', type=int, default=128, help='rank*field_dim equals the output size')
    
    args = parser.parse_args()

    # set up the device
    fs_train = jnp.load('data/fs_train_surface.npy').reshape(-1,21**2)
    fs_test = jnp.load('data/fs_test_surface.npy').reshape(-1,21**2)
    u_test = jnp.load('data/u_test_surface.npy')

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', 'results_surface', args.model_name)
    result_dir = os.path.join(root_dir, 'nf'+str(args.batch)+'_nc'+str(args.nc) + '_branch_' + str(args.branch_depth) + 
                              '_'+str(args.branch_hidden)+'_trunk_' + str(args.trunk_depth) + 
                              '_'+str(args.trunk_hidden)+'_r'+ str(args.r))
    
    # make dir
    os.makedirs(result_dir, exist_ok=True)
    
    # logs
    if os.path.exists(os.path.join(result_dir, 'log (loss).csv')):
        os.remove(os.path.join(result_dir, 'log (loss).csv'))

    if os.path.exists(os.path.join(result_dir, 'log (eval metrics).csv')):
        os.remove(os.path.join(result_dir, 'log (eval metrics).csv'))
    
    if os.path.exists(os.path.join(result_dir, 'total parameters.csv')):
        os.remove(os.path.join(result_dir, 'total parameters.csv'))
    
    if os.path.exists(os.path.join(result_dir, 'total runtime (sec).csv')):
        os.remove(os.path.join(result_dir, 'total runtime (sec).csv'))

    if os.path.exists(os.path.join(result_dir, 'memory usage (mb).csv')):
        os.remove(os.path.join(result_dir, 'memory usage (mb).csv'))
        
    
    # update function
    update_fn = update

    # define the optimizer
    schedule = optax.exponential_decay(args.lr,500,0.9)
    optimizer = optax.adam(schedule)

    # random key
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    # init model
    if args.model_name == 'DeepOHeat':
        model = eqx.filter_jit(eqx.filter_vmap(DeepOHeat(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim, 
                                                           branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                                           trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey)))
        
    
    elif args.model_name == 'DeepOHeat_KAN':
        model = eqx.filter_jit(eqx.filter_vmap(DeepOHeat_KAN(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim, 
                                                        branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                                        trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey)))
    
    elif args.model_name == 'DeepOHeat_ST':
        model = eqx.filter_jit(DeepOHeat_ST(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim, 
                                                           branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                                           trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey))
    elif args.model_name == 'DeepOHeat_v1':
        model = eqx.filter_jit(DeepOHeat_v1(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim,
                                                        branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                                        trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey))
    
    
    # Filter the model to get only the trainable parameters
    params = eqx.filter(model, eqx.is_array)
    # Count the total number of parameters by summing the size of each array
    num_params = sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size, params)))
    print(f'Total number of parameters: {num_params}')
    
    
    # init state
    key, subkey = jax.random.split(key)
    opt_state = optimizer.init(params)

    # train/test generator
    if args.model_name == "DeepOHeat" or args.model_name == "DeepOHeat_KAN":
        train_generator = jax.jit(lambda key: deepoheat_train_generator(fs_train, args.batch, args.nc, key))
        test_generator = jax.jit(deepoheat_test_generator)
        loss_fn = apply_model_deepoheat
    
    elif args.model_name == "DeepOHeat_ST" or args.model_name == "DeepOHeat_v1":
        train_generator = jax.jit(lambda key: deepoheat_st_train_generator(fs_train, args.batch, args.nc, key))
        test_generator = jax.jit(deepoheat_st_test_generator)
        loss_fn = apply_model_deepoheat_st
    
    
    # train the model
    model, optimizer, opt_state, runtime = train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, args.epochs, args.log_epoch, result_dir, args.device_name, subkey)
    
    # save the model
    eqx.tree_serialise_leaves(os.path.join(result_dir,args.model_name+'_trained_model.eqx'),model)
    
    
    # eval the trained model
    rel_l2_mean, rel_l2_std, rmse_mean, rmse_std, max_l1_mean, max_l1_std, mape_mean, mape_std, pape_mean, pape_std = eval_heat3d(model,test_generator,fs_test,u_test,result_dir)
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    print(f'rel_l2 --> mean: {rel_l2_mean:.8f} (std: {rel_l2_std: 8f})')
    print(f'rmse --> mean: {rmse_mean:.8f} (std: {rmse_std: 8f})')
    print(f'max_l1 --> mean: {max_l1_mean:.8f} (std: {max_l1_std: 8f})')
    print(f'mape --> mean: {mape_mean:.8f} (std: {mape_std: 8f})')
    print(f'pape --> mean: {pape_mean:.8f} (std: {pape_std: 8f})')
    
    
    # # save runtime and eval metrics
    runtime = np.array([runtime])
    num_params = np.array([num_params])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')
    np.savetxt(os.path.join(result_dir, 'total parameters.csv'), num_params, delimiter=',')
    with open(os.path.join(result_dir, 'log (eval metrics).csv'), 'a') as f:
        f.write(f'rel_l2_mean: {rel_l2_mean}\n')
        f.write(f'rel_l2_std: {rel_l2_std}\n')
        f.write(f'rmse_mean: {rmse_mean}\n')
        f.write(f'rmse_std: {rmse_std}\n')
        f.write(f'max_l1_mean: {max_l1_mean}\n')
        f.write(f'max_l1_std: {max_l1_std}\n')
        f.write(f'mape_mean: {mape_mean}\n')
        f.write(f'mape_std: {mape_std}\n')
        f.write(f'pape_mean: {pape_mean}\n')
        f.write(f'pape_std: {pape_std}\n')