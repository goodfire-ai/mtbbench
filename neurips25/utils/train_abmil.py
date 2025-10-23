import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import h5py
import os
from sklearn.model_selection import train_test_split
from trident.slide_encoder_models import ABMILSlideEncoder
from omegaconf import OmegaConf

conf = OmegaConf.load("neurips25/configs/base.yaml")

class RegressionModel(nn.Module):
    def __init__(self, input_feature_dim=1536, n_heads=1, head_dim=512, dropout=0.2, gated=True, hidden_dim=512):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.regressor = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_encoder(x)
        output = self.regressor(features).squeeze(1)
        return output


def train_model(
    train_df: pd.DataFrame,
    input_feature_dim: int,
    n_heads: int,
    head_dim: int,
    dropout: float,
    gated: bool,
    hidden_dim: int,
    SEED: int,
    batch_size: int,
    epochs: int
) -> torch.nn.Module:    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = RegressionModel(input_feature_dim=input_feature_dim, n_heads=n_heads, head_dim=head_dim, 
                            dropout=dropout, gated=gated, hidden_dim=hidden_dim).to(device)

    train_loader = DataLoader(H5Dataset(train_df, "train", num_features=2048, seed=SEED), num_workers=10,
                              batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        for features, labels in train_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model


def evaluate_model(model, test_df, SEED):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test_loader = DataLoader(H5Dataset(test_df, "test", num_features=2048, seed=SEED), num_workers=10,
                             batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))
    model.eval()
    all_labels, all_outputs = [], []
    total_loss = 0.

    criterion = nn.MSELoss()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    print(np.concatenate(all_outputs))
    print(np.concatenate(all_labels))
    mse = total_loss / len(test_loader)
    print(f"Test MSE: {mse:.4f}")
    return np.concatenate(all_outputs)


class H5Dataset(Dataset):
    def __init__(self, df, split, num_features=2048, seed=42):
        self.df = df
        self.num_features = num_features
        self.split = split
        self.seed = seed
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(row["file_name"], "r") as f:
            features = torch.from_numpy(f["features"][:])
            if len(features.shape) == 3:
                features = features.squeeze(0)

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(self.seed))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(self.seed))
            features = features[indices]

        label = torch.tensor(row["value"], dtype=torch.float)
        return features, label

def train():

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    files = []
    for file in os.listdir(conf.hancock.block2unifeatures):
        files.append(file.replace(".h5", ".png"))

    values = pd.read_csv(conf.hancock.block2celldensities)
    values = values[values["file_name"].isin(files)]

    # Prepare features and labels table
    result = values[["file_name", "positive_percent"]]
    result["file_name"] = result["file_name"].str.replace(".png", ".h5")
    result["file_name"] = conf.hancock.block2unifeatures + "/" + result["file_name"]
    result['value'] = result['positive_percent']

    # Split in train and test
    train_df, test_df = train_test_split(result, test_size=0.2, random_state=42, shuffle=True)
    print(len(train_df), len(test_df))

    # Train model
    model = train_model(train_df, input_feature_dim=1536, n_heads=2, head_dim=512, dropout=0.3, gated=True, 
                        hidden_dim=1536, SEED=SEED, batch_size=64, epochs=70)
    # Evaluate model
    evaluate_model(model, test_df, SEED)
    # Save model weights
    torch.save(model.state_dict(), f"ABMIL_checkpoint_regression_new.pt")


# Main part of the script
train()

# Load measurements
test_df = pd.read_csv(conf.hancock.block1celldensities)
test_df = test_df.rename(columns={"filename": "file_name"})
filenames = os.listdir(conf.hancock.block1unifeatures)
filenames = [filename.replace(".h5", ".png") for filename in filenames]
test_df = test_df[test_df["file_name"].isin(filenames)]
test_df = test_df[["file_name", "positive_percent"]]
test_df["file_name"] = test_df["file_name"].str.replace(".png", ".h5")
test_df["file_name"] = conf.hancock.block1unifeatures + "/" + test_df["file_name"]
# Turn from regression to classification in 10 classes for positive_percent
test_df['value'] = test_df['positive_percent']


# Evaluate model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RegressionModel(input_feature_dim=1536, n_heads=2, head_dim=512, 
                            dropout=0.3, gated=True, hidden_dim=1024).to(device)

model.load_state_dict(torch.load("data/ABMIL_checkpoint_regression.pt"))
model.eval()

# Prepare for IHC Tool
predictions = evaluate_model(model, test_df.fillna(0), SEED=42)

file_names = test_df["file_name"]
file_names = [os.path.basename(file).replace("_lvl1_dwnsmpl4.h5", "") for file in file_names]
file_names = [os.path.basename(file).replace("_TumorCenter", "_TMA_IHC_TumorCenter") for file in file_names]
file_names = [os.path.basename(file).replace("_InvasionFront", "_TMA_IHC_InvasionFront") for file in file_names]
file_names = [f"{"_".join(file.rsplit("_")[:-1])}_{int(file.rsplit("_")[-1]) % 2}.png" for file in file_names]
test_df["file_name"] = file_names
test_df["value"] = predictions

test_df = test_df[["file_name", "value"]]
print(test_df.head())
test_df.to_csv("data/hancock/cell_density_measurements.csv", index=False)