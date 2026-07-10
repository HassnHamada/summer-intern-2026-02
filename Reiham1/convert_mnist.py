import idx2numpy
from PIL import Image
import os
from tqdm import tqdm

# Folder containing the IDX files
folder = r"C:\Users\GM\Desktop\MINST"

# Output folders
train_output = os.path.join(folder, "train")
test_output = os.path.join(folder, "test")

# Read IDX files
X_train = idx2numpy.convert_from_file(
    os.path.join(folder, "train-images.idx3-ubyte")
)
y_train = idx2numpy.convert_from_file(
    os.path.join(folder, "train-labels.idx1-ubyte")
)

X_test = idx2numpy.convert_from_file(
    os.path.join(folder, "t10k-images.idx3-ubyte")
)
y_test = idx2numpy.convert_from_file(
    os.path.join(folder, "t10k-labels.idx1-ubyte")
)

# Create folders for digits 0-9
for i in range(10):
    os.makedirs(os.path.join(train_output, str(i)), exist_ok=True)
    os.makedirs(os.path.join(test_output, str(i)), exist_ok=True)

# Save training images
print("Saving training images...")
for i in tqdm(range(len(X_train))):
    label = str(y_train[i])
    img = Image.fromarray(X_train[i])
    img.save(os.path.join(train_output, label, f"{i}.png"))

# Save testing images
print("Saving testing images...")
for i in tqdm(range(len(X_test))):
    label = str(y_test[i])
    img = Image.fromarray(X_test[i])
    img.save(os.path.join(test_output, label, f"{i}.png"))

print("Finished!")
print("Training images:", len(X_train))
print("Testing images:", len(X_test))
