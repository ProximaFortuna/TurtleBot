#!/usr/bin/env python3

# Run with python3 {path}/train resnet.py -p {path to image folder}

import csv, random, argparse
import cv2, numpy as np, torch, torch.nn as nn
from torchvision import models, transforms as T

W, H = 224, 168 # Width and Height of the input images for resnet18
MEAN = np.array([0.485, 0.456, 0.406], np.float32) # Mean value for normalizing images to match the distribution of the ImageNet dataset
STD = np.array([0.229, 0.224, 0.225], np.float32) # Standard deviation value for normalizing images to match the distribution of the ImageNet dataset
NUM_CLASSES = 6 # Number of classes in the dataset


# Takes in path to data folder
# Returns a uint8 RGB array of images [N, H, W, 3] and an array of labels [N]
def load_data(path):
    rows = [r for r in csv.reader(open(path + 'labels.txt')) if r] # Read the labels.txt file and store each row as a list in the rows variable, filtering out any empty rows
    X, y = [], []
    for name, label in rows:
        img = cv2.imread(f"{path}{name.strip()}.png") # Read the image file corresponding to the name in the row
        img = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) # Resize the image to the specified width and height
        X.append(img); y.append(int(label)) # Append the images to the X list and the labels to the y list
    return np.stack(X), np.array(y)

# Takes in an array of labels, a fraction for validation, and a random seed
# Returns two lists of indices for training and validation data split so that each class is represented in both sets
def split_data(y, val_frac, seed):
    rng = random.Random(seed) # Create a random number generator with the specified seed
    train, val = [], []
    for cls in np.unique(y): # Iterate over each unique class in the labels array
        idx = [i for i in range(len(y)) if y[i] == cls] # Create a list of indices for the current class
        rng.shuffle(idx) # Shuffle the indices for the current class to randomize the order
        val_size = max(1, round(len(idx) * val_frac)) # Calculate the number of samples to include in the validation set for the current class, ensuring at least one sample is included
        val += idx[:val_size]; train += idx[val_size:] # Add the first val_size indices to the validation set and the remaining indices to the training set
    return train, val

# Data augmentation transforms for training
data_transforms = T.Compose([
    T.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6), # Apply random affine transformations to the image, including rotation, translation, scaling, and shearing
    T.ColorJitter(0.4, 0.4, 0.4, 0.03), # Apply random changes to the brightness, contrast, saturation, and hue of the image
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5) # Randomly apply Gaussian blur to the image with a probability of 0.5
])

# Applies data transforms to the batch of uint8 images and converts them to expected format for resnet18
def apply_transforms(u8, augment):
    x = torch.from_numpy(u8).permute(0, 3, 1, 2) # Convert the input uint8 images to a PyTorch tensor and rearrange the dimensions from (N, H, W, C) to (N, C, H, W)
    if augment:
        x = torch.stack([data_transforms(i) for i in x]) # Apply the data augmentation transforms to each image in the batch if augment is True
    x = x.float() / 255.0 # Convert the images to float and normalize the pixel values to the range [0, 1]
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1) # Normalize the images using the mean and standard deviation values for the ImageNet dataset, which is expected by the ResNet model

# Evaluates the model on the given data and returns accuracy and confusion matrix
def evaluate(model, X, y, device, batch_size=32):
    model.eval() # Set the model to evaluation mode, which disables dropout and batch normalization layers
    preds = []
    with torch.no_grad(): # Disable gradient calcs
        for i in range(0, len(X), batch_size):
            preds += model(apply_transforms(X[i:i+batch_size], augment=False).to(device)).argmax(1).cpu().tolist() # Apply transforms, pass throught the model, get predicted classes through argmax
    preds = np.array(preds)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return (preds == y).mean(), cm

# Trains the model for the given number of epochs on the training data and evaluates on validation data every 5 epochs
def run_epochs(model, params, lr, epochs, X_train, y_train, X_val, y_val, device):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4) # AdamW optimizer with weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) # Adjusts learning rate to get lower as training goes on
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # Cross-entropy loss with label smoothing

    for epoch in range(epochs):
        model.train() # Set model to train mode
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() # Set batch norm layers to eval mode to avoid updating running stats during training
        perm = np.random.permutation(len(X_train)) # Randomly permute the indices of the training data to create mini-batches
        for i in range(0, len(perm), 16):
            batch = perm[i:i+16] # Get little batch
            if len(batch) < 2: continue
            x, t = apply_transforms(X_train[batch], augment=True).to(device), torch.from_numpy(y_train[batch]).to(device) # Apply transforms
            opt.zero_grad(); loss_fn(model(x), t).backward(); opt.step() # Zero the gradients, compute the loss, backpropagate, and update the model parameters
        scheduler.step() # Update learning rate according to the scheduler
        if X_val is not None and (epoch +1) % 5 == 0:
            acc, _ = evaluate(model, X_val, y_val, device) # Evaluate the model on the validation data every 5 epochs and print the accuracy
            print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {acc:.1%}")

# Main function to parse arguments, load data, train the model, evaluate it, and save the trained model in both PyTorch and ONNX formats
def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, required=True, help='Path to data folder')
    parser.add_argument('--full', action='store_true', help='Use full dataset for training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y = load_data(args.data_path) # Load data. X is images, y is labels
    if args.full:
        X_train, y_train, X_val, y_val = X, y, None, None
    else:
        train_idx, val_idx = split_data(y, val_frac=0.2, seed=args.seed) # Split the data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    print(f"Training on {len(X_train)} samples, validating on {len(X_val) if X_val is not None else 0} samples")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Get pretrained resnet18 trained on ImageNet
    for param in model.parameters(): param.requires_grad = False # Freeze every layer
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES) # NEW RESNET HEAD (This is what we are training)
    model.to(device)

    # PARAMS
    lr = 1e-3 # Learning rate
    epochs1 = 20 # Epochs to run with just training final classifier
    epochs2 = 20 # Epochs to run with an unfrozen layer 4 (Last layer before head)

    # Train only final Layer
    run_epochs(model, model.fc.parameters(), lr=lr, epochs=epochs1, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, device=device)

    # Un freeze layer 4 and Train again
    for param in model.layer4.parameters(): param.requires_grad = True
    run_epochs(model, [p for p in model.layer4.parameters() if p.requires_grad], lr=lr/10, epochs=epochs2, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, device=device)

    if X_val is not None:
        acc, cm = evaluate(model, X_val, y_val, device) # Get final accuracy and confusion matrix
        print(f"Final Validation Accuracy: {acc:.1%}")
        print("Confusion Matrix:")
        print(cm)

    # Save the model
    model.eval().cpu()
    torch.save(model.state_dict(), 'signs_resnet18.pth')
    torch.onnx.export(model, torch.zeros(1, 3, H, W), 'signs_resnet18.onnx', input_names=['input'], output_names=['output'], opset_version=12)
    print("Model saved as signs_resnet18.pth and signs_resnet18.onnx")

if __name__ == '__main__':
    main()