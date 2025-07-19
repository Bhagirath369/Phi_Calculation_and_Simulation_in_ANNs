import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

connectivity = np.zeros((6, 6), dtype=int)
node_labels = [f'N{i}' for i in range(6)]

plt.figure(figsize=(6, 5))
sns.heatmap(connectivity, annot=True, cmap='Blues', xticklabels=node_labels, yticklabels=node_labels, cbar=True)

plt.title("Connectivity Matrix (Feed Forward)")
plt.xlabel("To Node")
plt.ylabel("From Node")

plt.savefig('connectivity_mnist_classification_only_feed_forward.png', dpi=300, bbox_inches='tight')
plt.show()