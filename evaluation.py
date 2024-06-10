import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

# Source code imports
from training import MODEL_PATH


def evaluate_model(model: Module, test_set_dataloader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_set_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        _, ground_truth = torch.max(labels.data, 1)
        total += labels.size(0)
        for label, prediction in zip(ground_truth, predicted):
            if label == prediction:
                correct += 1

    accuracy = 100 * correct / total

    return accuracy

def evaluate_model_by_name(model_name: str, test_set_dataloader: DataLoader):
    model = torch.load(os.path.join(MODEL_PATH, model_name))
    accuracy = evaluate_model(model=model, test_set_dataloader=test_set_dataloader)
    print(f'Test accuracy of the model: {accuracy}%')
