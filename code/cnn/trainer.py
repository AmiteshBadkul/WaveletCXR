import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import torch.optim as optim
from torchvision.models import resnet50
from utils import measure_computational_load

def initialize_model(num_classes=3):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Adjust the final layer to the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

@measure_computational_load
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device...', device)
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    y_true = all_labels
    y_scores = [x[1] for x in all_probs]  # Assuming binary classification, taking the score of class 1

    accuracy = sum([1 if round(score) == label else 0 for score, label in zip(y_scores, y_true)]) / len(y_true)
    precision_val = precision_score(y_true, [round(score) for score in y_scores])
    recall_val = recall_score(y_true, [round(score) for score in y_scores])
    confusion_mat = confusion_matrix(y_true, [round(score) for score in y_scores])
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, [round(score) for score in y_scores])

    return {"Accuracy": accuracy, "Precision": precision_val, "Recall": recall_val, "Confusion Matrix": confusion_mat, "PR AUC": pr_auc, "ROC AUC": roc_auc, "F1 Score": f1}
