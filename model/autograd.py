#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/10/17 0:47
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : autograd.py
# @Description    : ******
"""


__all__ = ["gradient", "jacobian", "hessian"]
# from model import bkd
import torch
backend_name = "pytorch"

def gradient(y, xs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`

    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if bkd.is_tensor(xs):
        inputs = [xs]
    else:
        inputs = list(xs)
    grads = torch.autograd.grad(y, xs, grad_outputs,
                              allow_unused=True,
                              retain_graph=retain_graph,
                              create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return grads


class Jacobian(object):
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i = 0, ..., dim_y-1 and
    j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes J[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, ..., dim_y).
        xs: Input Tensor of shape (batch_size, ..., dim_x).
    """

    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs

        if backend_name in ["pytorch", "paddle"]:
            self.dim_y = ys.shape[-1]
        elif backend_name == "jax":
            # For backend jax, a tuple of a jax array and a callable is passed as one of
            # the arguments, since jax does not support computational graph explicitly.
            # The array is used to control the dimensions and the callable is used to
            # obtain the derivative function, which can be used to compute the
            # derivatives.
            self.dim_y = ys[0].shape[-1]
        elif backend_name == "taichi":
            assert False, "taichi will be supported in the future"
            pass

        self.dim_x = xs.shape[-1]

        self.J = {}

    def __call__(self, i=None, j=None, create_graph=True, retain_graph=True):
        """Returns J[`i`][`j`]. If `j` is ``None``, returns the gradient of y_i, i.e.,
        J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))
        # Compute J[i]
        if i not in self.J:
            if backend_name == "pytorch":
                y = self.ys[..., i: i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = torch.autograd.grad(
                    y, self.xs, grad_outputs=torch.ones_like(y),
                    create_graph=create_graph,
                    retain_graph=retain_graph,
                    allow_unused=True,
                )[0]
            elif backend_name == "paddle":
                y = self.ys[..., i: i + 1] if self.dim_y > 1 else self.ys
                self.J[i] = paddle.grad(y, self.xs,
                                        create_graph=create_graph,
                                        retain_graph=retain_graph,
                                        allow_unused=True,
                                        )[0]
            elif backend_name == "jax":
                # Here, we use jax.grad to compute the gradient of a function. This is
                # different from TensorFlow and PyTorch that the input of a function is
                # no longer a batch. Instead, it is a single point. Formally, backend
                # jax computes gradients pointwisely and then vectorizes to batch, by
                # jax.vmap. However, computationally, this is in fact done batchwisely
                # and efficiently. It is very important to note that, without jax.vmap,
                # this can only deal with functions whose output is a scalar and input
                # is a single point.
                # Other options are jax.jacrev + jax.vmap or jax.jacfwd + jax.vmap,
                # which could be used to compute the full Jacobian matrix efficiently,
                # if needed. Also, jax.vjp, jax.jvp will bring more flexibility and
                # efficiency. jax.vjp + jax.vmap or jax.jvp + jax.vmap will be
                # implemented in the future.
                grad_fn = jax.grad(lambda x: self.ys[1](x)[i])
                self.J[i] = (jax.vmap(grad_fn)(self.xs), grad_fn)
            elif backend_name == "taichi":
                assert False, "taichi will be supported in the future"
                pass

        if backend_name in ["pytorch", "paddle"]:
            return (
                self.J[i] if j is None or self.dim_x == 1 else self.J[i][..., j: j + 1]
            )
        if backend_name == "jax":
            # Unlike other backends, in backend jax, a tuple of a jax array and a callable is returned, so that
            # it is consistent with the argument, which is also a tuple. This may be useful for further computation,
            # e.g. Hessian.
            return (
                self.J[i]
                if j is None or self.dim_x == 1
                else (
                    self.J[i][0][..., j: j + 1],
                    lambda inputs: self.J[i][1](inputs)[..., j: j + 1],
                )
            )


class Jacobians(object):
    """Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Js = {}

    def __call__(self, ys, xs, i=0, j=None, create_graph=True, retain_graph=True):
        # For backend tensorflow and pytorch, self.Js cannot be reused across iteration.
        # For backend pytorch, we need to reset self.Js in each iteration to avoid
        # memory leak.
        #
        # For backend tensorflow, in each iteration, self.Js is reset to {}.
        #
        # Example:
        #
        #
        #
        # For backend pytorch, in each iteration, ys and xs are new tensors
        # converted from np.ndarray, so self.Js will increase over iteration.
        #
        # Example:
        #
        # mydict = {}
        #
        # def f(x):
        #     print(mydict)
        #     y = 1 * x
        #     print(hash(y), hash(x))
        #     mydict[(y, x)] = 1
        #     print(mydict)
        #
        # for i in range(2):
        #     x = np.random.random((3, 4))
        #     x = torch.from_numpy(x)
        #     x.requires_grad_()
        #     f(x)
        if backend_name in ["pytorch", "paddle"]:
            key = (ys, xs)
        elif backend_name == "jax":
            key = (id(ys[0]), id(xs))
        if key not in self.Js:
            self.Js[key] = Jacobian(ys, xs)
        return self.Js[key](i, j)

    def clear(self):
        """Clear cached Jacobians."""
        self.Js = {}


def jacobian(ys, xs, i=0, j=None, create_graph=True, retain_graph=True):
    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int):
        j (int or None):

    Returns:
        J[`i`][`j`] in Jacobian matrix J. If `j` is ``None``, returns the gradient of
        y_i, i.e., J[`i`].
    """
    return jacobian._Jacobians(ys, xs, i=i, j=j, create_graph=create_graph, retain_graph=retain_graph)


jacobian._Jacobians = Jacobians()


class Hessian(object):
    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j = 0,..., dim_x-1.

    It is lazy evaluation, i.e., it only computes H[i][j] when needed.

    Args:
        y: Output Tensor of shape (batch_size, 1) or (batch_size, dim_y > 1).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: If `y` has the shape (batch_size, dim_y > 1), then `y[:, component]`
            is used to compute the Hessian. Do not use if `y` has the shape (batch_size,
            1).
        grad_y: The gradient of `y` w.r.t. `xs`. Provide `grad_y` if known to avoid
            duplicate computation. `grad_y` can be computed from ``Jacobian``.
    """

    def __init__(self, y, xs, component=None, grad_y=None):
        if backend_name in ["pytorch", "paddle"]:
            dim_y = y.shape[-1]
        elif backend_name == "jax":
            dim_y = y[0].shape[0]
        elif backend_name == "taichi":
            assert False, "taichi will be supported in the future"
            pass

        if dim_y > 1:
            if component is None:
                raise ValueError("The component of y is missing.")
            if component >= dim_y:
                raise ValueError(
                    "The component of y={} cannot be larger than the dimension={}.".format(
                        component, dim_y
                    )
                )
        else:
            if component is not None:
                raise ValueError("Do not use component for 1D y.")
            component = 0

        if grad_y is None:
            grad_y = jacobian(y, xs, i=component, j=None)
        self.H = Jacobian(grad_y, xs)

    def __call__(self, i=0, j=0):
        """Returns H[`i`][`j`]."""
        return self.H(i, j)


class Hessians(object):
    """Compute multiple Hessians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Hs = {}

    def __call__(self, y, xs, component=None, i=0, j=0, grad_y=None):
        if backend_name in ["pytorch", "paddle"]:
            key = (y, xs, component)
        elif backend_name == "jax":
            key = (id(y[0]), id(xs), component)
        if key not in self.Hs:
            self.Hs[key] = Hessian(y, xs, component=component, grad_y=grad_y)
        return self.Hs[key](i, j)

    def clear(self):
        """Clear cached Hessians."""
        self.Hs = {}


def hessian(ys, xs, component=None, i=0, j=0, grad_y=None):
    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: If dim_y > 1, then `ys[:, component]` is used as y to compute the
            Hessian. If dim_y = 1, `component` must be ``None``.
        i (int):
        j (int):
        grad_y: The gradient of y w.r.t. `xs`. Provide `grad_y` if known to avoid
            duplicate computation. `grad_y` can be computed from ``jacobian``. Even if
            you do not provide `grad_y`, there is no duplicate computation if you use
            ``jacobian`` to compute first-order derivatives.

    Returns:
        H[`i`][`j`].
    """
    return hessian._Hessians(ys, xs, component=component, i=i, j=j, grad_y=grad_y)


hessian._Hessians = Hessians()


def clear():
    """Clear cached Jacobians and Hessians."""
    jacobian._Jacobians.clear()
    hessian._Hessians.clear()


if __name__ == "__main__":

    import numpy as np
    import torch

    net = torch.nn.Linear(50, 5)
    x = torch.ones((300, 20, 50), requires_grad=True)
    y = net(x)

    c = jacobian(y, x, 2, None)
    print(c.shape)


# auto-gradients in the old version for pytorch
# import torch
#
# def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
#     '''
#     Compute the gradient of `outputs` with respect to `inputs`
#
#     gradient(x.sum(), x)
#     gradient((x * y).sum(), [x, y])
#     '''
#     if torch.is_tensor(inputs):
#         inputs = [inputs]
#     else:
#         inputs = list(inputs)
#     grads = torch.autograd.grad(outputs, inputs, grad_outputs,
#                                 allow_unused=True,
#                                 retain_graph=retain_graph,
#                                 create_graph=create_graph)
#     grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
#     return torch.cat([x.contiguous().view(-1) for x in grads])
#
#
# def hessian(output, inputs, out=None, allow_unused=False, create_graph=False, return_inputs=False):
#     '''
#     Compute the Hessian of `output` with respect to `inputs`
#
#     hessian((x * y).sum(), [x, y])
#     '''
#     assert output.ndimension() == 0
#
#     if torch.is_tensor(inputs):
#         inputs = [inputs]
#     else:
#         inputs = list(inputs)
#
#     n = sum(p.numel() for p in inputs)
#     if out is None:
#         out = output.new_zeros(n, n)
#
#     ai = 0
#     for i, inp in enumerate(inputs):
#         [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
#         grad = torch.zeros_like(inp) if grad is None else grad
#         grad = grad.contiguous().view(-1)
#
#         for j in range(inp.numel()):
#             if grad[j].requires_grad:
#                 row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
#             else:
#                 row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)
#
#             out[ai, ai:].add_(row.type_as(out))  # ai's row
#             if ai + 1 < n:
#                 out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
#             del row
#             ai += 1
#         del grad
#     # https://github.com/pytorch/pytorch/issues/16532
#     if return_inputs:
#         return out, inputs
#     else:
#         return out
#
#
# def jacobian(outputs, inputs, create_graph=False, return_inputs=False):
#     '''
#     Compute the Jacobian of `outputs` with respect to `inputs`
#
#     jacobian(x, x)
#     jacobian(x * y, [x, y])
#     jacobian([x * y, x.sqrt()], [x, y])
#     '''
#     if torch.is_tensor(outputs):
#         outputs = [outputs]
#     else:
#         outputs = list(outputs)
#
#     if torch.is_tensor(inputs):
#         inputs = [inputs]
#     else:
#         inputs = list(inputs)
#
#     jac = []
#     for output in outputs:
#         output_flat = output.view(-1)
#         output_grad = torch.zeros_like(output_flat)
#         for i in range(len(output_flat)):
#             output_grad[i] = 1
#             jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
#             output_grad[i] = 0
#     if return_inputs:
#         return torch.stack(jac), inputs
#     else:
#         return torch.stack(jac)
