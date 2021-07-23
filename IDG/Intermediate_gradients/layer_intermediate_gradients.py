"""
Class for obtaining the gradients used to calculate Layer Integrated Gradients.
"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.scatter_gather import scatter

from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_input_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.attr._utils.gradient import _forward_layer_eval, _run_forward

from .intermediate_gradients import IntermediateGradients

class LayerIntermediateGradients(LayerIntegratedGradients):
    """
    Layer Intermediate Gradients is a variant of Intermediate Gradients that
    retrieves gradients for a layer's inputs or outputs, depending on whether
    we are looking at the former or the latter.
    Integrated Gradients is a model interpretability algorithm that assigns
    an importance score to each input feature by appproximating the integral
    of gradients between the model's output with respect to the inputs along
    the path between baselines and inputs.
    Intermediate Gradients is a modification of the Integrated Gradients algorithm
    that returns the gradients used to approximate the integral of gradients.
    These gradients can then be used for further attribution exploration.
    Returned from Layer Intermediate Gradients is a tensor containing each of the
    gradients and a tensor of the step sizes used to calculate those gradients.
    """
    def __init__(self, forward_func, layer, device_ids=None):
        """
        Parameters
        ----------
        forward_func: callable
            The model's forward function.
        layer: torch.nn.Module
            Layer for which intermediate gradients are computed.
            Output will be (n_steps, forward_func_output, layer_output_dimensions).
        device_ids: list(int)
            List of Device IDs if running a DataParallel model.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)
        self.device_ids = device_ids
        self.interm_grad = IntermediateGradients(forward_func)

    def attribute(self,
                  inputs,
                  baselines,
                  additional_forward_args,
                  target=None,
                  n_steps=50,
                  method="gausslegendre",
                  attribute_to_layer_input=False):
        """
        Set-up hooks on each of the layers and then compute the intermediate gradients.
        Parameters
        ----------
        inputs: torch.tensor(num_ids) or torch.tensor([batch_size, num_ids])
            Input for which layered intermediate gradients are computed.
            This is the tensor that would be passed to the forward_func.
        baselines: torch.tensor(num_ids) or torch.tensor([batch_size, num_ids])
            Baselines to define the starting point for gradient calculations.
            Should be the same length as inputs.
        additional_forward_args: any, tuple(any), or None
            If the forward function takes any additional arguments,
            they can be provided here.  If there are multiple forward args,
            a tuple of the forward arguments can be provided.
        target: int, tuple, tensor or list
            Output indices for which gradients are computed (for classification
            cases, this is the target class). For 2D batched inputs, the targets
            can be a singl einteger which is applied to all input examples or a
            1D tensor with length matching the number of examples in inputs (dim 0).
        n_steps: int
            The number of steps used by the approximation method. Default: 50.
            The article suggests between 20 and 300 steps are enough to
            approximate the integral.
        method: str
            Method for determining step sizes for the gradients.
            One of `riemann_right`, `riemann_left`, `riemann_middle`,
            `riemann_trapezoid` or `gausslegendre`.
        attribute_to_layer_input: bool
            Indicates whether to compute the gradients with respect to
            the layer input or output. If `attribute_to_layer_input` is
            set to True then the gradients will be computed with respect to
            layer input, otherwise it will be computed with respect
            to layer output.
        Returns
        -------
        grads: torch.tensor(num_steps*batch_size, num_embeddings, num_ids), dtype=float32
            Tensor of the gradients for each input.
            Multiplying this value by step_sizes, summing along the
            num_steps dimension, and then multiplying by the (inputs - baseline)
            produces the attributions that are returned from Layer Integrated
            Gradients.
        step_sizes: torch.tensor(num_steps), dtype=float32
            Tensor of the step sizes used to calculate each of the gradients.
        intermediates: torch.tensor(num_steps*batch_size, num_embeddings, num_ids), dtype=float32
            Tensor of the intermediate x values used to calculate each of the gradients.
        """
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)


        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)
        inputs_layer, is_layer_tuple = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        baselines_layer, _ = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        # inputs -> these inputs are scaled
        def gradient_func(
                forward_fn,
                inputs,
                target_ind=None,
                additional_forward_args=None,
        ):
            if self.device_ids is None:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type:ignore
                    inputs, target_gpus=self.device_ids
                )

            scattered_inputs_dict = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(module, hook_inputs, hook_outputs=None):
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    if is_layer_tuple:
                        return scattered_inputs_dict[device]
                    return scattered_inputs_dict[device][0]

                if attribute_to_layer_input:
                    hook = self.layer.register_forward_pre_hook(layer_forward_hook)
                else:
                    hook = self.layer.register_forward_hook(layer_forward_hook)

                output = _run_forward(
                    self.forward_func, tuple(), target_ind, additional_forward_args
                )
                hook.remove()
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(torch.unbind(output), inputs)
            return grads

        self.interm_grad.gradient_func = gradient_func
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )
        grads, step_sizes, intermediates, input_forward, baseline_forward = self.interm_grad.attribute(
            inputs_layer,
            baselines=baselines_layer,
            target=target,
            additional_forward_args=all_inputs,
            n_steps=n_steps,
            method=method,
        )
        return grads, step_sizes, intermediates, input_forward, baseline_forward