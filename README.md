# ViT-SAM-PyTorch-Implementations

## q1: ViT on CIFAR10
### How to run in Colab:
1. Download q1.ipynb
2. Enable GPU in Runtime -> Change runtime type
3. Run all cells sequentially

### Config for best model
Hyperparameter|Value
Image Size|32X32
Patch Size|4X4
Embedding dim|256
Transformer heads|8
Attention heads|8
MLP hidden dim|512
Dropout|0.1
Optimizer|AdamW
Learning Rate|0.0003
Weight Decay|0.05
Epochs|200
Batch size|256
Scheduler|CosineAnnealingLR with warmup
