import matplotlib.pyplot as plt
import torch

def plot_history(history):
    epochs = list(range(len(history['loss'])))
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(epochs, history['loss'], 'ro-', label='Train Loss')
    ax[0].plot(epochs, history['val_loss'], 'b^-', label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, history['accuracy'], 'ro-', label='Train Accuracy')
    ax[1].plot(epochs, history['val_accuracy'], 'b^-', label='Val Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    fig.suptitle("Training Curves", fontsize=16)
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)
