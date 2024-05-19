import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from create_dataset import SkinCancerMNISTDataset


def main():
    df = pd.read_csv("data/HAM10000_metadata.csv")

    # Send 3/4 of the data to the training set
    num_samples = len(df)
    samples_to_use = 40  # I keep this pretty small since I set my environment up on my laptop and these images are pretty large
    training_set_size = int(samples_to_use * .75)
    df_train = df.iloc[:training_set_size, :]
    df_test = df.iloc[training_set_size:samples_to_use, :]

    training_set = SkinCancerMNISTDataset(df_train)
    testing_set = SkinCancerMNISTDataset(df_test)

    batch_size = 10

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=True)

    model = resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7)
    criterion = nn.MSELoss();  # Use MSE since we're classifying
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def evaluate_model():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, ground_truth = torch.max(labels.data, 1)
            total += labels.size(0)
            for label, prediction in zip(ground_truth, predicted):
                if label == prediction:
                    correct += 1
        print(f'Test accuracy of the model on {len(df_test)} test images: {100 * correct / total}%')
        model.train()

    evaluate_model()


    # Now train the network.
    model.train() # switches modes

    # Look at the entire training set 10 times
    num_epochs = 3

    losses = [];
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = labels
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss);
            if (i+1) % 1 == 0:
                
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f        \r' 
                                %(epoch+1, num_epochs, i+1, len(training_set)//batch_size, loss))
                
if __name__ == "__main__":
    main()
