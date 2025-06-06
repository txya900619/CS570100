import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def log(content, filename="log.txt"):
    with open(filename, "a") as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print("-" * 70)
    for name, param in model.named_parameters():
        print(f"{name:20} {str(param.shape):30} {str(param.dtype):15}")
        if with_values:
            print(param)


def print_nonzeros(model, filename="log.txt"):
    print("-" * 70)
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        log_text = (
            f"{name:20} | Nonzeros weight = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | "
            f"Total_pruned = {total_params - nz_count:7} | Shape = {tensor.shape}"
        )
        log(log_text, filename)
        print(log_text)
    log_text = "-" * 70 + "\n"
    log_text += (
        f"Alive: {nonzero}, Pruned weight: {total - nonzero}, Total: {total}, Compression rate : {total / nonzero:10.2f}x"
        f"  ({100 * (total - nonzero) / total:6.2f}% pruned)"
    )
    log(log_text, filename)
    print(log_text)


def test(model, use_cuda=True):
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=1000,
        shuffle=False,
        **kwargs,
    )
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # Sum up batch loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # Get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            f"Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
        )
    return accuracy
