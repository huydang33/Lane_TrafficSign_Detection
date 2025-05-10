import argparse
import torch
import torch_utils.utils
from torch_utils.engine import train_one_epoch, evaluate
from dataset import get_dataloader
from model import faster_rcnn

def parse_args():
    parser = argparse.ArgumentParser(description="Argument")
    # Add arguments
    parser.add_argument("--resume", type=bool, help="Resume training", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--num_epochs", type=int, help="Number of epoch", default=20)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = faster_rcnn(num_classes=43)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_loader, val_loader, test_loader = get_dataloader(args.batch_size)

    best_loss = float('inf')
    if args.resume:
        checkpoint = torch.load('checkpoint_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()

        val_loss = evaluate(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoint_best.pth')
            print(f"✅ Saved best model at epoch {epoch}, val_loss = {val_loss:.4f}")


if __name__ == "__main__":
    main()