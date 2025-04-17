import jax
import jax.numpy as jnp
from jax import vmap
from hvp import hvp_fwdfwd, hvp_fwdrev

@jax.jit
def create_mesh(xi_batch, yi_batch, zi_batch):
    return jnp.meshgrid(xi_batch.ravel(), yi_batch.ravel(), zi_batch.ravel(), indexing='ij')

batch = 10
x = jnp.stack([jnp.linspace(0, 1, 21).reshape(-1,1)]*batch)
y = jnp.stack([jnp.linspace(0, 1, 21).reshape(-1,1)]*batch)
z = jnp.stack([jnp.linspace(0, 0.5, 11).reshape(-1,1)]*batch)
x_mesh, y_mesh, z_mesh = vmap(create_mesh, in_axes=(0, 0, 0))(x, y, z)
x = x_mesh.reshape(batch, 21*21*11, 1)
y = y_mesh.reshape(batch, 21*21*11, 1)
z  = z_mesh.reshape(batch, 21*21*11, 1)
top_idx = z == 0.5
bottom_idx = z == 0
left_idx = x == 0
right_idx = x == 1
front_idx = y == 0
back_idx = y == 1
inside_idx = ~(top_idx | bottom_idx | left_idx | right_idx | front_idx | back_idx)

# use this for PI-DeepPNet when eval if having OOM issue
def process_batch(model, batch):
    (t, x, y), f = batch
    return model(((t, x, y), f))

def process_all_data(model, data, num_batches):
    (t, x, y), f = data
    batch_size = 100 // num_batches
    results = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = ((t[start:end], x[start:end], y[start:end]), f[start:end])
        result = process_batch(model, batch)
        results.append(result)
    return jnp.concatenate(results, axis=0)

@jax.jit
def rel_l2(u, u_pred):
    u_norm = jnp.linalg.norm(u.reshape(-1,1))
    diff_norm = jnp.linalg.norm(u.reshape(-1,1)-u_pred.reshape(-1,1))
    return diff_norm / u_norm

@jax.jit
def rmse(u, u_pred):
    return jnp.sqrt(jnp.mean((u_pred.reshape(-1,1)-u.reshape(-1,1))**2))


@jax.jit
def mape(u, u_pred):
    return jnp.mean(jnp.abs((u.reshape(-1,1)-u_pred.reshape(-1,1))/u.reshape(-1,1)))

@jax.jit
def pape(u, u_pred):
    return jnp.max(jnp.abs((u.reshape(-1,1)-u_pred.reshape(-1,1))/u.reshape(-1,1)))

@jax.jit
def max_l1(u, u_pred):
    u_max = jnp.max(u)
    u_pred_max = jnp.max(u_pred)
    return jnp.abs(u_max - u_pred_max)


def eval_heat3d(model, test_generator, fs, u, result_dir):
    x, y, z, f, u = test_generator(fs, u)
    u_pred = model(((x,y,z),f))
    jnp.save(result_dir+'/u_pred_heat3d.npy', u_pred)
    rel_l2_u = vmap(rel_l2, in_axes=(0, 0))(u,u_pred)
    rmse_u = vmap(rmse, in_axes=(0, 0))(u,u_pred)
    max_l1_u = vmap(max_l1, in_axes=(0, 0))(u,u_pred)
    
    u = 25*u + 293.15
    u_pred  = 25*u_pred + 293.15
    mape_u = vmap(mape, in_axes=(0, 0))(u,u_pred)
    pape_u = vmap(pape, in_axes=(0, 0))(u,u_pred)
    
    return (jnp.mean(rel_l2_u), jnp.std(rel_l2_u), 
            jnp.mean(rmse_u), jnp.std(rmse_u), 
            jnp.mean(max_l1_u), jnp.std(max_l1_u),
            jnp.mean(mape_u), jnp.std(mape_u),
            jnp.mean(pape_u), jnp.std(pape_u))
    
