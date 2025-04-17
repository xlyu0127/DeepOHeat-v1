import equinox as eqx
import jax
import jax.random as jr, jax.numpy as jnp


class ChebyKANLayer(eqx.Module):
    input_dim: int
    output_dim: int
    degree: int
    cheby_coeffs: jax.Array
    arange: jax.Array

    def __init__(self, input_dim: int, output_dim: int, degree: int, key: jax.random.PRNGKey):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        
        cheby_coeffs_key, _ = jr.split(key)
        self.cheby_coeffs = jr.normal(
            cheby_coeffs_key, 
            (input_dim, output_dim, degree + 1)
        ) / jnp.sqrt(input_dim * (degree + 1))
        
        self.arange = jnp.arange(0, degree + 1, 1)

    def __call__(self, x):
        # Normalize x to [-1, 1] using tanh
        x = jnp.tanh(x)
      
        # Expand dimensions
        x = jnp.expand_dims(x, axis=-1)
      
        x = jnp.repeat(x, self.degree + 1, axis=-1)

        # Apply acos
        x = jnp.arccos(x)
        
        # Multiply by arange [0 .. degree]
        x = x * self.arange
        
        # Apply cos
        x = jnp.cos(x)
       
        
        # Compute the Chebyshev interpolation
        y = jnp.einsum('id,iod->o', x, self.cheby_coeffs)
        
        return y
    
class ChebyKAN(eqx.Module):
    layers: list

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, degree: int=3, *, key: jax.random.PRNGKey):
        super().__init__()
        keys = jr.split(key, depth + 1)
        
        layers = []
        for i in range(depth + 1): 
            if i == 0:
                layer = ChebyKANLayer(in_size, width_size, degree, key=keys[i])
                layer_norm = eqx.nn.LayerNorm(width_size)
                layers.append(layer)
                layers.append(layer_norm)
            elif i == depth:
                layer = ChebyKANLayer(width_size, out_size, degree, key=keys[i])
                layers.append(layer)
            else:
                layer = ChebyKANLayer(width_size, width_size, degree, key=keys[i])
                layer_norm = eqx.nn.LayerNorm(width_size)
                layers.append(layer)
                layers.append(layer_norm)
            
        
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x