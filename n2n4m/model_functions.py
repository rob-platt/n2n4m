import torch


def train(model, optimizer, criterion, data_loader, device):
    model.train()
    train_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def validate(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs[0].unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions, dim=0).squeeze(1)
    return predictions
