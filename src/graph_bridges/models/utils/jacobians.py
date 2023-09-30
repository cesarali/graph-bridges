import torch

def compute_jacobian(model, x):
    x.requires_grad_(True)
    output = model(x)
    batch_size, num_outputs = output.size(0), output.size(1)
    jacobian = torch.zeros(batch_size, num_outputs, x.size(1))

    for i in range(num_outputs):
        output_i = output[:, i]
        gradient = torch.zeros_like(output_i)
        gradient.fill_(1.0)
        output_i.backward(gradient, retain_graph=True)
        jacobian[:, i, :] = x.grad.clone()
        x.grad.zero_()
    return jacobian

def compute_jacobian_with_fix_time(model, x,t):
    model_t = lambda x:model(x,t)
    jacobian = compute_jacobian(model_t,x)
    return jacobian
