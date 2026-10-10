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
import MapKeys

# Run on GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Weight constants for loss function
BALANCE_STRENGTH = 0.5
CONTAINED_FACTOR = 1.5
UNCONTAINED_FACTOR = 1.0

########################################################################################################

class IvysaurusDataset(Dataset):
    def __init__(self, grids, trackVars, showerVars, y, containment):
        # grids: dict of name -> (N, H, W, 1) numpy arrays
        self.grids = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in grids.items()}   #
        self.trackVars = torch.from_numpy(trackVars.astype(np.float32))
        self.showerVars = torch.from_numpy(showerVars.astype(np.float32))
        self.labels = torch.from_numpy(np.argmax(y, axis=1).astype(np.int64))
        self.containment = containment

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        sample = {k: v[i] for k, v in self.grids.items()}
        sample["trackVars"] = self.trackVars[i]
        sample["showerVars"] = self.showerVars[i]
        return sample, self.labels[i], self.containment[i]

########################################################################################################

def run_model(model, batch, device):
    args = [batch[key].to(device) for key in MapKeys.GRID_KEYS]
    args.append(batch["trackVars"].to(device))
    args.append(batch["showerVars"].to(device))
    return model(*args)

########################################################################################################

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    trainFileNames = glob.glob(f'{args.input_dir}/*.npz')
    grids_train = {key: [] for key in MapKeys.GRID_KEYS}
    grids_test  = {key: [] for key in MapKeys.GRID_KEYS}
    trackVars_train = []
    showerVars_train = []
    trackVars_test = []
    showerVars_test = []
    y_train = []
    y_test = []
    distToEdge_train = []
    distToEdge_test = []

    for fname in trainFileNames:
        print(f"Reading file: {fname}")

        with np.load(fname) as data:
            for key in MapKeys.GRID_KEYS:
                grids_train[key].append(data[f"{key}_train"])
                grids_test[key].append(data[f"{key}_test"])
            trackVars_train.append(data['trackVars_train'])
            trackVars_test.append(data['trackVars_test'])
            showerVars_train.append(data['showerVars_train'])
            showerVars_test.append(data['showerVars_test'])
            y_train.append(data['y_train'])
            y_test.append(data['y_test'])
            distToEdge_train.append(data['distToEdge_train'])
            distToEdge_test.append(data['distToEdge_test'])

    for key in MapKeys.GRID_KEYS:
        grids_train[key] = np.concatenate(grids_train[key], axis=0)
        grids_test[key]  = np.concatenate(grids_test[key], axis=0)
    trackVars_train = np.concatenate(trackVars_train, axis=0)
    trackVars_test = np.concatenate(trackVars_test, axis=0)
    showerVars_train = np.concatenate(showerVars_train, axis=0)
    showerVars_test = np.concatenate(showerVars_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    distToEdge_train = np.concatenate(distToEdge_train, axis=0)
    distToEdge_test = np.concatenate(distToEdge_test, axis=0)
    isContained_train = (distToEdge_train > 10.0)
    isContained_test = (distToEdge_test > 10.0)

    # Work out some network shapes
    n_classes = y_train.shape[1]
    n_track_vars = trackVars_train.shape[1]
    n_shower_vars = showerVars_train.shape[1]
    dimensions = grids_train[MapKeys.GRID_KEYS[0]].shape[1]
    print('n_classes:', n_classes)
    print('n_track_vars:', n_track_vars)
    print('n_shower_vars:', n_shower_vars)
    print(f'y (train, test): {y_train.shape}, {y_test.shape}')
    print(f'Particle counts:')
    print(f' Train: {np.unique(np.argmax(y_train, axis=1), return_counts=True)}')
    print(f' Test: {np.unique(np.argmax(y_test, axis=1), return_counts=True)}')

    # Datasets
    train_ds = IvysaurusDataset(grids_train, trackVars_train, showerVars_train, y_train, isContained_train)    
    test_ds = IvysaurusDataset(grids_test, trackVars_test, showerVars_test, y_test, isContained_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    # Get non-energy based weights
    class_weights_t = [[], []]
    for is_contained in [False, True] :
        containment_mask = isContained_train if is_contained else ~isContained_train
        particle_type = np.argmax(y_train[containment_mask], axis=1)
        counts = np.array([np.count_nonzero(particle_type == c) for c in range(n_classes)])
        classWeights = (1.0 + BALANCE_STRENGTH * (np.sqrt(np.clip(counts.max() / counts, 1.0, 4.0)) - 1.0))
        
        if is_contained:
            classWeights *= CONTAINED_FACTOR
        else:
            classWeights *= UNCONTAINED_FACTOR
            
        class_weights_t[(1 if is_contained else 0)] = classWeights
        print("Class Counts:", counts)        
        print("Class Weights:", classWeights)        
    class_weights_t = torch.from_numpy(np.array(class_weights_t)).to(device)    
        
    # Model setup
    model = IvysaurusModel(dimensions, n_classes, n_track_vars, n_shower_vars).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=1)
    model_path = f'{args.output_dir}/my_model_all.pt'

    # Training loop
    best_val_acc = -1.0    
    for epoch in range(args.n_epochs):
        # Train
        model.train()
        all_preds_train = []
        all_labels_train = []
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch, labels, is_contained in train_loader:
            labels = labels.to(device)
            is_contained = is_contained.to(device)
            optimiser.zero_grad()
            logits = run_model(model, batch, device)
            losses = criterion(logits, labels)
            weights = class_weights_t[is_contained.long(), labels]
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        all_containment = []
        val_loss, val_correct, val_total = 0.0, 0, 0        

        with torch.no_grad():
            for batch, labels, is_contained in test_loader:
                labels = labels.to(device)
                is_contained = is_contained.to(device)
                logits = run_model(model, batch, device)
                losses = criterion(logits, labels)
                weights = class_weights_t[is_contained.long(), labels]
                loss = (losses * weights).sum() / weights.sum()                

                preds = logits.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_containment.append(is_contained.cpu().numpy())
                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_containment = np.concatenate(all_containment)
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{args.n_epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - train_bal_acc: {train_bal_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_bal_acc: {val_bal_acc:.4f}")

        for i_containment in [True, False, None] :
            print('----------------------------------------')
            print(f'--{"-Contained-" if (i_containment==True) else "Uncontained" if (i_containment==False) else "All"}--')
            print('----------------------------------------')            
            class_names = ["muon", "proton", "pion", "electron", "michel/DR", "photon"]

            if (i_containment != None) :
                mask = (all_containment == i_containment)
            else :
                mask = np.ones_like(all_containment, dtype=bool)
            
            if (np.count_nonzero(mask) == 0) : 
                continue;
            
            print(classification_report(all_labels[mask], all_preds[mask], target_names=class_names, digits=4))        
            cm = confusion_matrix(all_labels[mask], all_preds[mask])

            print("Confusion Matrix:")
            print("                 Predicted")
            print("             " + "  ".join(f"{name:>9}" for name in class_names))
            for i, row in enumerate(cm):
                print(f"{class_names[i]:>10} " + "  ".join(f"{x:9d}" for x in row))

        scheduler.step(val_bal_acc)

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
    
