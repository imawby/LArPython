import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from ivysaurus_model import IvysaurusModel

# RUN ON GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

########################################################################################################

class IvysaurusDataset(Dataset):
    def __init__(self, grids, trackVars, showerVars, y):
        # grids: dict of name -> (N, H, W, 1) numpy arrays
        self.grids = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in grids.items()}   #
        self.trackVars = torch.from_numpy(trackVars.astype(np.float32))
        self.showerVars = torch.from_numpy(showerVars.astype(np.float32))
        # CrossEntropyLoss wants class indices, not one-hot
        self.labels = torch.from_numpy(np.argmax(y, axis=1).astype(np.int64))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        sample = {k: v[i] for k, v in self.grids.items()}
        sample["trackVars"] = self.trackVars[i]
        sample["showerVars"] = self.showerVars[i]
        return sample, self.labels[i]

########################################################################################################    

GRID_KEYS = [
    "startU", "startU_mask", "endU", "endU_mask",
    "startV", "startV_mask", "endV", "endV_mask",
    "startW", "startW_mask", "endW", "endW_mask",
]

########################################################################################################

def run_model(model, batch, device):
    args = [batch[key].to(device) for key in GRID_KEYS]
    args.append(batch["trackVars"].to(device))
    args.append(batch["showerVars"].to(device))
    return model(*args)

########################################################################################################    

def main(args):

    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    suffix = "Contained" if args.is_contained else "Exiting"
    trainFileNames = glob.glob(f'{args.input_dir}/*_{suffix}.npz')

    grids_train = {key: [] for key in GRID_KEYS}
    grids_test  = {key: [] for key in GRID_KEYS}
    trackVars_train = []
    showerVars_train = []
    trackVars_test = []
    showerVars_test = []
    y_train = []
    y_test = []

    for fname in trainFileNames:
        print(f"Reading file: {fname}")

        with np.load(fname) as data:
            for key in GRID_KEYS:
                grids_train[key].append(data[f"{key}_train"])
                grids_test[key].append(data[f"{key}_test"])

            trackVars_train.append(data['trackVars_train'])
            trackVars_test.append(data['trackVars_test'])
            showerVars_train.append(data['showerVars_train'])
            showerVars_test.append(data['showerVars_test'])
            y_train.append(data['y_train'])
            y_test.append(data['y_test'])

    for key in GRID_KEYS:
        grids_train[key] = np.concatenate(grids_train[key], axis=0)
        grids_test[key]  = np.concatenate(grids_test[key], axis=0)
        
    trackVars_train = np.concatenate(trackVars_train, axis=0)
    trackVars_test = np.concatenate(trackVars_test, axis=0)
    showerVars_train = np.concatenate(showerVars_train, axis=0)
    showerVars_test = np.concatenate(showerVars_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Work out some network shapes
    n_classes = y_train.shape[1]
    n_track_vars = trackVars_train.shape[1]
    n_shower_vars = showerVars_train.shape[1]
    dimensions = grids_train[GRID_KEYS[0]].shape[1]

    print('n_classes:', n_classes)
    print('n_track_vars:', n_track_vars)
    print('n_shower_vars:', n_shower_vars)
    print("y_train:", y_train.shape, "y_test:", y_test.shape)

    print('Train')
    print(np.unique(np.argmax(y_train, axis=1), return_counts=True))
    print('Test')    
    print(np.unique(np.argmax(y_test, axis=1), return_counts=True))

    train_ds = IvysaurusDataset(grids_train, trackVars_train, showerVars_train, y_train)    
    test_ds = IvysaurusDataset(grids_test, trackVars_test, showerVars_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Class weights
    particle_type = np.argmax(y_train, axis=1)
    counts = [np.count_nonzero(particle_type == c) for c in range(n_classes)]
    maxParticle = max(counts)
    classWeights = np.sqrt(np.array([maxParticle / c for c in counts], dtype=np.float32))
    print("Class Weights:", classWeights)
    class_weights_t = torch.from_numpy(classWeights).to(device)

    # Model stuff
    model = IvysaurusModel(dimensions, n_classes, n_track_vars, n_shower_vars).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    #criterion = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.n_epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=1)
    model_path = f'{args.output_dir}/my_model_{"contained" if args.is_contained else "exiting"}.pt'

    # Training loop
    best_val_acc = -1.0    
    for epoch in range(args.n_epochs):
        # Train
        model.train()
        all_preds_train = []
        all_labels_train = []
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch, labels in train_loader:
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = run_model(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            preds = logits.argmax(1)
            all_preds_train.append(preds.cpu().numpy())
            all_labels_train.append(labels.cpu().numpy())
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        all_preds_train = np.concatenate(all_preds_train)
        all_labels_train = np.concatenate(all_labels_train)
        train_bal_acc = balanced_accuracy_score(all_preds_train, all_labels_train)
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        val_loss, val_correct, val_total = 0.0, 0, 0        

        with torch.no_grad():
            for batch, labels in test_loader:
                labels = labels.to(device)
                logits = run_model(model, batch, device)
                loss = criterion(logits, labels)

                preds = logits.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{args.n_epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - train_bal_acc: {train_bal_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_bal_acc: {val_bal_acc:.4f}")
        class_names = ["muon", "proton", "pion", "electron", "photon"]
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))        
        cm = confusion_matrix(all_labels, all_preds)

        print("Confusion Matrix:")
        print("                 Predicted")
        print("             " + "  ".join(f"{name:>9}" for name in class_names))

        for i, row in enumerate(cm):
            print(f"{class_names[i]:>10} " + "  ".join(f"{x:9d}" for x in row))

        scheduler.step(val_bal_acc)
        #scheduler.step(val_loss)

        # checkpoint: save best on val_acc
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            model_cpu = model.cpu()
            model_cpu.eval() # just to make sure
            scripted = torch.jit.script(model_cpu)
            scripted.save(model_path)
            model.to(device)
            print(f"  val_acc improved to {val_bal_acc:.4f}, saved model to {model_path}")

########################################################################################################            


def parse_cli():
    parser = argparse.ArgumentParser(description="Ivysaurus PID")

    parser.add_argument("--input_dir", type=str, required=True, help="Input file directory")    
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save model")
    
    parser.add_argument("--is_contained", action="store_true", help="Training for contained particles?")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs, default=10")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size, default=64")    
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate, default=1e-3")    
           
    return parser.parse_args()

########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    main(args)
