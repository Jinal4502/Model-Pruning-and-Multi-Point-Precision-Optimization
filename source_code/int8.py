# Imports
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
import multiprocessing

# Multiprocessing for DataLoader (required on macOS/Windows)
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)

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

def evaluate(model, input_tensor, device='cpu'):
    input_batch = input_tensor.unsqueeze(0).to(device)
    model.to(device)

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
            if next(model.parameters()).dtype == torch.float16:
                images = images.half()
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

def custom_abs(flattened_weights, zero_point, scale):
  # Dequantize the weights
  dequantized_weights = scale * (flattened_weights.dequantize() - zero_point)

  # Calculate the absolute values
  absolute_flattened_weights = torch.abs(dequantized_weights)
  quantized_absolute_weights = torch.quantize_per_tensor(
    absolute_flattened_weights, scale, zero_point, torch.qint8)

  return quantized_absolute_weights

def plot_weight_histogram(module, name, bins=15):
    weights = module.weight().dequantize().flatten().numpy()
    plt.hist(weights, bins=bins)
    plt.title(f"Histogram of Weights in {name}")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.show()

# --------------------------------------------------------
# Main Pipeline
# --------------------------------------------------------
if __name__ == '__main__':
    model = models.resnet18(pretrained=True).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open("/Users/jjvyas1/Downloads/mla_project/n01443537_goldfish.JPEG").convert("RGB")
    input_tensor = transform(img)

    print("Original Model")
    evaluate(model, input_tensor, 'cpu')
    get_model_size(model)
    measure_inference_time(model)

    imagenet_val_path = "/Users/jjvyas1/Downloads/mla_project/imagenet-val"
    test_loader = get_imagenet_subset_loader(imagenet_val_path, batch_size=32, num_samples=500)

    results = evaluate_topk_accuracy(model, test_loader, device="cpu", max_batches=10)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")

    # Quantization (QAT)
    from resnet import resnet18
    model = resnet18(pretrained=True)
    model.eval()
    fused_model = torch.ao.quantization.fuse_modules(model, model.modules_to_fuse())

    from torch.ao.quantization.fake_quantize import FakeQuantize
    activation_qconfig = FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.HistogramObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine))
    weight_qconfig = FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    fused_model.qconfig = torch.quantization.QConfig(activation=activation_qconfig, weight=weight_qconfig)
    fused_model.train()
    fake_quant_model = torch.ao.quantization.prepare_qat(fused_model)
    print("\nFake quant - PTQ")
    evaluate(fake_quant_model, input_tensor, 'cpu')

    fake_quant_model.apply(torch.ao.quantization.fake_quantize.disable_observer)
    print("\nFake quant - post-PTQ")
    evaluate(fake_quant_model, input_tensor, 'cpu')

    torch.backends.quantized.engine = 'qnnpack'
    converted_model = torch.ao.quantization.convert(fake_quant_model)

    print("\nQuantized model - INT8")
    evaluate(converted_model, input_tensor, 'cpu')
    get_model_size(converted_model)
    measure_inference_time(converted_model)

    results = evaluate_topk_accuracy(converted_model, test_loader, device="cpu", max_batches=20)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")

    for name, module in converted_model.named_modules():
        if name == "fc":
            plot_weight_histogram(module, name, bins=2)

    # Pruning Quantized Layers
    for name, module in converted_model.named_modules():
        if "conv1" in name or "conv2" in name:
            print(module)
            weight = module.weight()
            scale = module.scale
            zero_point = module.zero_point

            # Perform pruning: Setting a fraction of weights to zero (e.g., 30%)
            pruning_amount = 0.3
            num_weights = weight.numel()
            num_pruned_weights = int(pruning_amount * num_weights)

            # Flatten the weights, sort by absolute value, and zero out the smallest
            zero_point = max(min(zero_point, 127), -128)
            weight = custom_abs(weight, zero_point, scale)
            flattened_weights = weight.reshape(-1)
            sorted_indices = torch.argsort(flattened_weights)

            # # Prune the smallest weights
            pruned_weights = flattened_weights.clone()
            pruned_weights[sorted_indices[:num_pruned_weights]] = 0

            # # Reshape pruned weights back to the original shape
            pruned_weights = pruned_weights.view_as(weight)

            # print(module.weight())
            # print(module.bias())
            bias = module.bias()
            module.set_weight_bias(pruned_weights, bias)

        elif "fc" in name:
            weight, bias = module._weight_bias()
            try:
                scale = weight.q_scale()
                zero_point = weight.q_zero_point()
            except:
                scale = weight.q_per_channel_scales()
                zero_point = weight.q_per_channel_zero_points()
                scale = float(scale.mean())
                zero_point = int(zero_point.float().mean().round())

            # Perform pruning: Setting a fraction of weights to zero (e.g., 30%)
            pruning_amount = 0.1
            num_weights = weight.numel()
            num_pruned_weights = int(pruning_amount * num_weights)

            # Flatten the weights, sort by absolute value, and zero out the smallest
            zero_point = max(min(zero_point, 127), -128)
            weight = custom_abs(weight, zero_point, scale)
            flattened_weights = weight.reshape(-1)
            sorted_indices = torch.argsort(flattened_weights)

            # # Prune the smallest weights
            pruned_weights = flattened_weights.clone()
            pruned_weights[sorted_indices[:num_pruned_weights]] = 0

            # # Reshape pruned weights back to the original shape
            pruned_weights = pruned_weights.view_as(weight)

            # print(module.weight())
            # print(module.bias())
            module.set_weight_bias(pruned_weights, bias)

    print("\nPruned Quantized Model - INT8")
    get_model_size(converted_model)
    measure_inference_time(converted_model)
    evaluate(converted_model, input_tensor, 'cpu')

    results = evaluate_topk_accuracy(converted_model, test_loader, device="cpu", max_batches=20)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")

    for name, module in converted_model.named_modules():
        if name == "fc":
            plot_weight_histogram(module, name, bins=2)
