from jax import grad, jit


def autograd(func):
    def wrapper(*args):
        return func(*args), grad(func)(*args)

    return wrapper


def jit_autograd(func):
    return jit(autograd(func))
