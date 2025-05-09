# --------------------------------------------------------
# Imports and Global Setup
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import os, time, random
from tqdm import tqdm

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    print(f"Model size: {size_mb:.2f} MB")
    return size_mb

def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cpu', repeats=50):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    model.to(device)

    for _ in range(10):
        _ = model(dummy_input)

    start = time.time()
    for _ in range(repeats):
        _ = model(dummy_input)
    end = time.time()

    avg_time = (end - start) / repeats
    print(f"Inference time (avg over {repeats}): {avg_time*1000:.2f} ms")
    return avg_time

def evaluate(model, input_tensor, device_str='cpu'):
    input_batch = input_tensor.to(device_str)
    model.to(device_str)

    with torch.no_grad():
        output = model(input_batch)
        if output.is_quantized:
            output = output.dequantize()
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("/Users/jjvyas1/Downloads/mla_project/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")

def evaluate_topk_accuracy(model, dataloader, device="cpu", max_batches=10, topk=(1, 5)):
    model.to(device)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            if i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if outputs.is_quantized:
                outputs = outputs.dequantize()
            _, topk_preds = outputs.topk(max(topk), dim=1, largest=True, sorted=True)
            correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct[:, :5].sum().item()
            total += labels.size(0)

    return {"top1": correct_top1 / total, "top5": correct_top5 / total}

def get_imagenet_subset_loader(imagenet_val_dir, batch_size=32, num_samples=1000, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = ImageFolder(root=imagenet_val_dir, transform=preprocess)
    indices = random.sample(range(len(full_dataset)), num_samples)
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --------------------------------------------------------
# Main Pipeline - Unstructured Pruning Only
# --------------------------------------------------------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open("/Users/jjvyas1/Downloads/mla_project/n01443537_goldfish.JPEG").convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    model = models.resnet18(pretrained=True)

    # Apply unstructured pruning
    prune_amount = 0.3
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    print("\nPruned Model (Unstructured)")
    evaluate(model, input_tensor, 'cpu')
    get_model_size(model)
    measure_inference_time(model)

    imagenet_val_path = "/Users/jjvyas1/Downloads/mla_project/imagenet-val"
    test_loader = get_imagenet_subset_loader(imagenet_val_path, batch_size=32, num_samples=500)

    results = evaluate_topk_accuracy(model, test_loader, device="cpu", max_batches=20)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")