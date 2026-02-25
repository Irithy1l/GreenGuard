from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, num_classes),
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    torch.manual_seed(42)

    root = Path("dataset")
    train_dir = root / "train"
    val_dir = root / "val"
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    best_acc = 0.0
    best_state = None

    print("classes:", train_ds.classes)
    print("class_to_idx:", train_ds.class_to_idx)
    print("train size:", len(train_ds), "val size:", len(val_ds))
    print("device:", device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "input_size": [128, 128],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            }

    if best_state is None:
        raise RuntimeError("training failed to produce a model")

    out_path = out_dir / "soil_water_classifier.pt"
    torch.save(best_state, out_path)
    print("saved:", out_path)
    print("best_val_acc:", round(best_acc, 4))


if __name__ == "__main__":
    main()
