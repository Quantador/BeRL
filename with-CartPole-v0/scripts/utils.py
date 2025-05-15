import pickle
import numpy as np
import matplotlib.pyplot as plt

def pref_save(pref_data, name):
    with open(name, 'wb') as f:
        pickle.dump(pref_data, f, pickle.HIGHEST_PROTOCOL)

def pref_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def plot_loss_rewards(x, losses, mean_returns, std_returns, title, label_loss="Loss", label_mean_return="Mean Return", xlabel="Epoch"):
    fig, ax1 = plt.subplots(figsize=(10,5))

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(label_loss, color=color)
    ax1.plot(x, losses, color=color, label=label_loss)
    ax1.fill_between(x, losses, losses, color=color, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(label_mean_return, color=color)
    ax2.plot(x, mean_returns, color=color, label=label_mean_return)
    ax2.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.3)

    fig.tight_layout()
    plt.title(title)
    plt.show()