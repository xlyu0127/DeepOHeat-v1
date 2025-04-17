import jax

# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over reverse
def hvp_revrev(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.vjp(f, primals)[1](tangents)
    primals_out, vjp_fn = jax.vjp(g, primals)
    tangents_out = vjp_fn((tangents,))[0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# forward over reverse
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.vjp(f, primals)[1](tangents[0])[0]
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over forward
def hvp_revfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, primals, tangents)[1]
    primals_out, vjp_fn = jax.vjp(g, primals)
    tangents_out = vjp_fn(tangents[0])[0][0]
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out