# from model import YourGNNModel # Build your model in model.py
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn

from data_loader import load_data
from model import GCN

warnings.filterwarnings("ignore")


def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(
    g,
    features,
    train_labels,
    val_labels,
    train_mask,
    val_mask,
    model,
    epochs,
    es_iters=None,
):
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    gamma = 0.1
    # training loop
    for epoch in range(epochs):
        model.train()

        lam = (float(epoch + 1) / float(epochs)) ** gamma if gamma is not None else 0.0

        logits = model(g, features)
        pesudo_label = logits.max(1)[1]
        pesudo_label[train_mask] = train_labels
        loss = loss_fcn(logits[train_mask], pesudo_label[train_mask])
        loss += lam * loss_fcn(logits[~train_mask], pesudo_label[~train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(g, features, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

        val_loss = loss_fcn(logits[val_mask], val_labels).item()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch + 1}")
                break


if __name__ == "__main__":
    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--es_iters", type=int, help="num of iters to trigger early stopping"
    )
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load data
    (
        features,
        graph,
        num_classes,
        train_labels,
        val_labels,
        test_labels,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data()

    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    in_size = features.shape[1]
    out_size = num_classes
    model = GCN(in_size, 64, out_size, dropout=0.3).to(device)

    graph = graph.to(device)
    features = features.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)

    # model training
    print("Training...")
    train(
        graph,
        features,
        train_labels,
        val_labels,
        train_mask,
        val_mask,
        model,
        args.epochs,
        args.es_iters,
    )

    print("Testing...")
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[test_mask]
        _, indices = torch.max(logits, dim=1)

    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open("output.csv", "w") as f:
        f.write("ID,Predict\n")
        for idx, pred in enumerate(indices):
            f.write(f"{idx},{int(pred)}\n")
    # Please remember to upload your output.csv file to Kaggle for scoring
