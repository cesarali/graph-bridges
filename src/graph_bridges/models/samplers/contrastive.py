import torch

import torch
import torch.nn.functional as F


def gibbs_with_gradients(unnormalized_log_prob, current_sample, num_samples=1):
    samples = []
    x = current_sample.clone()  # Make a copy of the current sample

    for _ in range(num_samples):
        # Compute d_tilde(x)
        d_tilde = torch.sigmoid(x)  # For binary data, Eq. 3

        # Compute q(i|x)
        q_i_given_x = F.softmax(d_tilde, dim=0)

        # Sample i ~ q(i|x)
        i = torch.multinomial(q_i_given_x, 1).item()

        # Flip dimension i
        x_flipped = x.clone()
        x_flipped[i] = 1 - x[i]

        # Compute d_tilde(x_flipped)
        d_tilde_flipped = torch.sigmoid(x_flipped)  # For binary data, Eq. 3

        # Compute q(i|x_flipped)
        q_i_given_x_flipped = F.softmax(d_tilde_flipped, dim=0)

        # Compute the acceptance probability
        acceptance_prob = min(1, torch.exp(unnormalized_log_prob(x_flipped) - unnormalized_log_prob(x)) *
                              q_i_given_x_flipped[i] / q_i_given_x[i])

        # Accept the new sample with probability acceptance_prob
        if torch.rand(1).item() < acceptance_prob:
            x = x_flipped

        samples.append(x.clone())

    return torch.stack(samples)


# Example usage:
# Define your unnormalized log-prob function unnormalized_log_prob(x)
# Initialize a current sample x_current
# Call the gibbs_with_gradients function to perform Gibbs sampling
# sampled_samples = gibbs_with_gradients(unnormalized_log_prob, x_current, num_samples=1000)



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def contrastive_divergence(rbm, input_data, k=1, learning_rate=0.1, num_epochs=100):
    num_samples, num_visible_units = input_data.size()
    num_hidden_units = rbm.num_hidden_units

    for epoch in range(num_epochs):
        positive_associations = torch.matmul(input_data.t(), rbm.sample_hidden_given_visible(input_data))
        negative_visible_activations = rbm.sample_visible_given_hidden(rbm.sample_hidden_given_visible(input_data))
        negative_hidden_activations = rbm.sample_hidden_given_visible(negative_visible_activations)
        negative_associations = torch.matmul(negative_visible_activations.t(), negative_hidden_activations)

        rbm.weights += learning_rate * ((positive_associations - negative_associations) / num_samples)
        rbm.visible_bias += learning_rate * torch.sum(input_data - negative_visible_activations, dim=0) / num_samples
        rbm.hidden_bias += learning_rate * torch.sum(
            rbm.sample_hidden_given_visible(input_data) - negative_hidden_activations, dim=0) / num_samples

        if epoch % 10 == 0:
            reconstruction_error = torch.mean(torch.square(input_data - negative_visible_activations))
            print(f"Epoch {epoch}: Reconstruction Error = {reconstruction_error.item()}")


class RBM:
    def __init__(self, num_visible_units, num_hidden_units):
        self.num_visible_units = num_visible_units
        self.num_hidden_units = num_hidden_units
        self.weights = torch.randn(num_visible_units, num_hidden_units)
        self.visible_bias = torch.zeros(num_visible_units)
        self.hidden_bias = torch.zeros(num_hidden_units)

    def sample_hidden_given_visible(self, visible):
        hidden_prob = sigmoid(torch.matmul(visible, self.weights) + self.hidden_bias)
        return torch.bernoulli(hidden_prob)

    def sample_visible_given_hidden(self, hidden):
        visible_prob = sigmoid(torch.matmul(hidden, self.weights.t()) + self.visible_bias)
        return torch.bernoulli(visible_prob)


# Example usage:
# Define the number of visible and hidden units
num_visible_units = 6
num_hidden_units = 2

# Create an RBM model
rbm = RBM(num_visible_units, num_hidden_units)

# Generate some example training data
input_data = torch.randint(2, size=(100, num_visible_units), dtype=torch.float32)

# Train the RBM using Contrastive Divergence
contrastive_divergence(rbm, input_data, k=1, learning_rate=0.1, num_epochs=1000)
