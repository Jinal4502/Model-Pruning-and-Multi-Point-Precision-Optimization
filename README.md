# Model-Pruning-and-Multi-Point-Precision-Optimization
I explored FP16, INT8, mixed-precision quantization, and unstructured pruning on ResNet-18 to study their impact on inference time, size, and accuracy. Since PyTorch pruning isn’t compatible with quantized models, I implemented custom routines to enable effective INT8 and mixed-precision pruning.

# ResNet-18 Quantization and Pruning Analysis

This project investigates the impact of quantization and pruning on the ResNet-18 architecture. We compare model size, inference speed, and top-5 accuracy across four setups:
- INT8 quantization with pruning
- FP16 quantization with pruning
- Mixed-precision (INT8 + FP32 FC layer) with pruning
- Direct unstructured pruning (FP32 baseline)

## 📁 Project Structure

```
mla_project/
├── results/11.png – INT8 FC histogram (pre-pruning)
├── results/12.png – INT8 FC histogram (post-pruning)
├── results/21.png – FP16 FC histogram (pre-pruning)
├── results/22.png – FP16 FC histogram (post-pruning)
├── results/31.png – Mixed FC histogram (pre-pruning)
├── results/32.png – Mixed FC histogram (post-pruning)
├── source_files/mixed_precision.py – Mixed precision quant+prune code
├── source_files/fp16.py – FP16 quant+prune code
├── source_files/direct_pruning_comparison.py – Plain unstructured pruning
├── source_files/int8.py – INT8 quant+prune code
├── resnet.py – Custom ResNet wrapper with fusion support
├── imagenet_classes.txt – Class label mapping
├── n01443537_goldfish.JPEG – Sample image for quick evaluation
├── demo_files/ProjectMilestone3_*_demo.ipynb – Jupyter notebooks for reproducibility
├── Model Pruning and Multi-Point Precision Optimization_UpdatedPresentation - Updated PPT
```

## 📌 Requirements

Install dependencies using:

```bash
pip install torch torchvision matplotlib tqdm
```

## 🧪 Usage

Each experiment is run via the respective script:

```bash
python int8.py              # INT8 quantization + pruning
python fp16.py              # FP16 quantization + pruning
python mixed_precision.py   # Mixed precision (INT8 + FP32) + pruning
python direct_pruning_comparison.py   # Plain unstructured pruning
```

Each script will output:
- Inference latency (on CPU and GPU)
- Top-5 accuracy on 500 ImageNet val samples
- Model size in MB
- Histogram plots of FC layer weights (before and after pruning)

## 📊 Evaluation Summary

| Mode               | Inference Speed (CPU) | Model Size | Top-5 Accuracy |
|--------------------|------------------------|-------------|----------------|
| INT8 + Pruning     | 782 ms                | 11.83 MB    | 76.45%         |
| FP16 + Pruning     | 16.55 ms              | 23.43 MB    | 81.98%         |
| Mixed (INT8+FP32)  | 80.82 ms              | 14.29 MB    | 79.23%         |
| Pruned FP32        | 12.11 ms              | 44.96 MB    | 89.54%         |

## 📷 Visualizations

Each mode includes two plots of the FC layer’s weight distribution:
- **Pre-Pruning**: Weight distribution post-quantization
- **Post-Pruning**: Effect of L1 magnitude pruning

See files: `11.png`–`32.png`.

## 📝 Report

The full report (`.pdf`) contains:
- Background on quantization and pruning
- Why PTQ and unstructured pruning were used
- Evaluation metrics and trade-off discussion
- Model architecture outputs and pruning pseudocode

## 📄 License

This repository is for academic and research use only.
