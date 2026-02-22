import os
import matplotlib.pyplot as plt

def plot_metric(history, metric, lr, save_dir, model_name):
    plt.figure()

    plt.plot(history.history[metric], label=f"train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} (LR={lr})")
    plt.legend()
    plt.grid(alpha=0.3)

    filename = f"{model_name}_{metric}_lr_{lr}.png"
    plt.savefig(os.path.join(save_dir, filename),
                dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()