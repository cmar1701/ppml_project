import os
import torch
import tenseal
import pandas as pd
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models import resnet50

# Source code imports
from data.create_dataset import SkinCancerMNISTDataset

MODEL_PATH = "trained_models"

def tenseal_context():
    context = tenseal.context(tenseal.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


def train_resnet():
    context = tenseal_context()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with device: {device}")

    df = pd.read_csv("data/HAM10000_metadata.csv")

    # Send 3/4 of the data to the training set
    num_samples = len(df)
    samples_to_use = 10
    training_set_size = int(samples_to_use * .75)
    df_train = df.iloc[:training_set_size, :]
    df_test = df.iloc[training_set_size:samples_to_use, :]

    training_set = SkinCancerMNISTDataset(metadata=df_train, device=device, tenseal_context=context)
    testing_set = SkinCancerMNISTDataset(metadata=df_test, device=device, tenseal_context=context)

    batch_size = 5

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=True)

    model = resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7)
    model.to(device)

    criterion = nn.MSELoss();  # Use MSE since we're classifying
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Now train the network.
    model.train() # switches modes

    num_epochs = 3


    # Train the model
    losses = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device=device).float()
            labels = labels.to(device=device)
            
            optimizer.zero_grad()
            outputs = model(images).to(device=device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss)
            if (i+1) % 1 == 0:
                print(f"Epoch: {epoch}/{num_epochs}, Iteration: {i}/{len(training_set)//batch_size}, Loss: {loss}")

    now_datetime_without_colons = datetime.now().isoformat().replace(":", "_")          
    torch.save(model, os.path.join(MODEL_PATH, now_datetime_without_colons + ".pt"))
                
if __name__ == "__main__":
    train_resnet()
