# import matplotlib.pyplot as plt
#
# x = [1,2,3,]
# y_train = [0.1295,0.0333, 0.0271]
# y_val = [0.0373, 0.0245, 0.0203]
#
# plt.plot(x, y_train, label="Training", marker='o')
# plt.plot(x, y_val, label="Validation", marker='o')
# plt.xticks(x)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Train vs Validation Loss")
# plt.savefig("save_dir//loss.png")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the confusion matrix
confusion_matrix = np.array([
    [2131, 3, 1],
    [7, 3068, 5],
    [2, 0, 2077]
])

# Class labels
class_labels = ['Class 0', 'Class 1', 'Class 2']

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix - CNN-LSTM Fusion Model', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Display the plot
plt.savefig("save_dir//confusion_matrix.png")
plt.show()

# Optional: Save the figure
# plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')