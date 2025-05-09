# Model-Pruning-and-Multi-Point-Precision-Optimization
I explored FP16, INT8, mixed-precision quantization, and unstructured pruning on ResNet-18 to study their impact on inference time, size, and accuracy. Since PyTorch pruning isn’t compatible with quantized models, I implemented custom routines to enable effective INT8 and mixed-precision pruning.
