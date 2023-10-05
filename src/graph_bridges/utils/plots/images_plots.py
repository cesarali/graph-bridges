# Display some generated samples
import matplotlib.pyplot as plt

def plot_sample(sample):
    if sample.shape[0] >= 64:
        plt.figure(figsize=(8, 8))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(sample[i].view(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()
    else:
        print("Not Enough Images in Sample")