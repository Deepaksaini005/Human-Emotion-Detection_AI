import os
import matplotlib.pyplot as plt

# This is the path to your dataset
train_path = "archive/train"
test_path  = "archive/test"

# Get all emotion folder names
emotions = os.listdir(train_path)

# Count images in each emotion folder
train_counts = []
test_counts  = []

for emotion in emotions:
    train_folder = os.path.join(train_path, emotion)
    test_folder  = os.path.join(test_path, emotion)
    train_counts.append(len(os.listdir(train_folder)))
    test_counts.append(len(os.listdir(test_folder)))
    print(f"{emotion:12} → Train: {train_counts[-1]:5}  |  Test: {test_counts[-1]:5}")

# Draw a bar chart
plt.figure(figsize=(10, 5))
x = range(len(emotions))
plt.bar(x, train_counts, width=0.4, label="Train", align="center")
plt.bar([i + 0.4 for i in x], test_counts, width=0.4, label="Test", align="center")
plt.xticks([i + 0.2 for i in x], emotions, rotation=15)
plt.title("Number of Images per Emotion")
plt.xlabel("Emotion")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
plt.show()
