import jax
import equinox as eqx
import time
import os
import GPUtil

def get_gpu_memory(device_name):
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[device_name].memoryUsed 
    return None


# Define your update function
@eqx.filter_jit
def update(grads, optimizer, opt_state, model):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

# Define your training loop
def train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, num_epochs, log_epoch, result_dir, device_name, key):
    # key, subkey = jax.random.split(key)
    # inputs = train_generator(subkey)
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        inputs = train_generator(subkey)
        

        loss, grads = loss_fn(model, *inputs)
        model, opt_state = update_fn(grads, optimizer, opt_state, model)

        if epoch == 1:
            gpu_memory = get_gpu_memory(device_name)
            with open(os.path.join(result_dir, 'memory usage (mb).csv'), 'a') as f:
                f.write(f'{gpu_memory}\n')
            start = time.time()

        if epoch % log_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss}")
            with open(os.path.join(result_dir, 'log (loss).csv'), 'a') as f:
                f.write(f'{loss}\n')
                
            
            
    runtime = time.time() - start

    return model, optimizer, opt_state, runtime