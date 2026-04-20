"""
Train MS-DSCCNet for brain tumor classification.
Usage: python train.py --data_dir data/brain_tumor
"""
import argparse, torch, torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import MSDSCCNet

CLASSES = ["glioma","meningioma","notumor","pituitary"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/brain_tumor")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tfm = {
        "train": transforms.Compose([
            transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), transforms.ColorJitter(0.2,0.2),
            transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)]),
        "val": transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)]),
    }
    loaders = {s: DataLoader(datasets.ImageFolder(f"{args.data_dir}/{s}", tfm[s]),
               args.batch_size, shuffle=(s=="train"), num_workers=4, pin_memory=True)
               for s in ("train","val")}

    model     = MSDSCCNet(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        for phase in ("train","val"):
            model.train() if phase=="train" else model.eval()
            correct = total = 0
            for imgs, labels in loaders[phase]:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.set_grad_enabled(phase=="train"):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    if phase=="train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                correct += (out.argmax(1)==labels).sum().item()
                total   += len(labels)
            if phase=="val":
                acc = correct/total
                print(f"Epoch {epoch:3d} | val_acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
                    print(f"  ✅ Best={best_acc:.4f}")
        scheduler.step()

if __name__=="__main__": main()
