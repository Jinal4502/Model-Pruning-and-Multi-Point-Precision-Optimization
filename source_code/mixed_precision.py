# --------------------------------------------------------
# Mixed Precision Quantization + Pruning Pipeline (INT8 + FP32 for FC)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, time, random
from tqdm import tqdm
from torch.ao.quantization.fake_quantize import FakeQuantize
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def evaluate(model, device_str='cpu'):
    input_batch = input_tensor.unsqueeze(0).to(device_str)
    model = model.to(device_str)

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

def get_imagenet_subset_loader(imagenet_val_dir, batch_size=32, num_samples=500, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(root=imagenet_val_dir, transform=preprocess)
    subset = Subset(dataset, random.sample(range(len(dataset)), num_samples))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def evaluate_topk_accuracy(model, dataloader, device="cpu", max_batches=10, topk=(1, 5)):
    model.to(device).eval()
    correct_top1 = correct_top5 = total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= max_batches:
                break
            outputs = model(images.to(device))
            if outputs.is_quantized:
                outputs = outputs.dequantize()

            _, topk_preds = outputs.topk(max(topk), dim=1)
            correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds).to(device))
            correct_top1 += correct[:, :1].sum().item()
            correct_top5 += correct[:, :5].sum().item()
            total += labels.size(0)

    return {"top1": correct_top1 / total, "top5": correct_top5 / total}

def custom_abs(flattened_weights, zero_point, scale):
    dequantized = scale * (flattened_weights.dequantize() - zero_point)
    abs_values = torch.abs(dequantized)
    return torch.quantize_per_tensor(abs_values, scale, zero_point, torch.qint8)

def plot_fc_weights(fc_layer, title="FC Weights", bins=15):
    weights = fc_layer.weight.detach().flatten().numpy()
    plt.hist(weights, bins=bins)
    plt.title(title)
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.show()

# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == '__main__':
    model = models.resnet18(pretrained=True).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = Image.open("/Users/jjvyas1/Downloads/mla_project/n01443537_goldfish.JPEG").convert("RGB")
    input_tensor = transform(img)

    print("Original Model")
    evaluate(model, 'cpu')
    get_model_size(model)
    measure_inference_time(model)

    test_loader = get_imagenet_subset_loader("/Users/jjvyas1/Downloads/mla_project/imagenet-val")
    results = evaluate_topk_accuracy(model, test_loader)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")

    from resnet import resnet18
    model = resnet18(pretrained=True).eval()
    fused_model = torch.ao.quantization.fuse_modules(model, model.modules_to_fuse())

    activation_qconfig = FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.HistogramObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine))
    weight_qconfig = FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

    fused_model.qconfig = torch.quantization.QConfig(activation=activation_qconfig, weight=weight_qconfig)
    fake_quant_model = torch.ao.quantization.prepare_qat(fused_model.train())

    print("\nFake quant - PTQ")
    evaluate(fake_quant_model, 'cpu')
    fake_quant_model.apply(torch.ao.quantization.fake_quantize.disable_observer)
    print("\nFake quant - post-PTQ")
    evaluate(fake_quant_model, 'cpu')

    torch.backends.quantized.engine = 'qnnpack'
    converted_model = torch.ao.quantization.convert(fake_quant_model)

    # Replace FC layer with dequantized FP32
    class CustomDeQuantize(nn.Module):
        def forward(self, x):
            return x.dequantize() if x.is_quantized else x

    fc_q = converted_model.fc
    fc_fp32 = nn.Linear(fc_q.in_features, fc_q.out_features)
    fc_fp32.weight.data = fc_q._weight_bias()[0].dequantize()
    fc_fp32.bias.data = fc_q._weight_bias()[1]
    converted_model.fc = nn.Sequential(CustomDeQuantize(), fc_fp32)

    print("\nQuantized Model: Mixed Precision")
    evaluate(converted_model)
    get_model_size(converted_model)
    measure_inference_time(converted_model)

    results = evaluate_topk_accuracy(converted_model, test_loader)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")
    plot_fc_weights(converted_model.fc[1], bins=15)

    # Apply pruning
    for name, module in converted_model.named_modules():
        if "conv1" in name or "conv2" in name:
            weight = module.weight()
            scale = module.scale
            zero_point = max(min(module.zero_point, 127), -128)
            weight = custom_abs(weight, zero_point, scale)
            flattened = weight.reshape(-1)
            num_to_prune = int(0.3 * flattened.numel())
            flattened[torch.argsort(flattened)[:num_to_prune]] = 0
            pruned_weight = flattened.view_as(weight)
            module.set_weight_bias(pruned_weight, module.bias())

        elif "fc" in name and isinstance(module, nn.Sequential) and isinstance(module[1], nn.Linear):
            prune.l1_unstructured(module[1], name="weight", amount=0.1)
            prune.remove(module[1], "weight")

    print("\nPruned Quantized Model: Mixed Precision")
    get_model_size(converted_model)
    measure_inference_time(converted_model)
    evaluate(converted_model, 'cpu')
    results = evaluate_topk_accuracy(converted_model, test_loader)
    print(f"Top-1 Accuracy: {results['top1'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5'] * 100:.2f}%")
    plot_fc_weights(converted_model.fc[1], bins=2)
