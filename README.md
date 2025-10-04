# ViT-SAM-PyTorch-Implementations

## q1: ViT on CIFAR10
### How to run in Colab:
1. Download q1.ipynb
2. Enable GPU in Runtime -> Change runtime type
3. Run all cells sequentially

### Config for best model
|Hyperparameter|Value|
|:---:|:---:|
|Image Size|32X32|
|Patch Size|4X4|
|Embedding dim|256|
|Transformer heads|8|
|Attention heads|8|
|MLP hidden dim|512|
|Dropout|0.1|
|Optimizer|AdamW|
|Learning Rate|0.0003|
|Weight Decay|0.05|
|Epochs|200|
|Batch size|256|
|Scheduler|CosineAnnealingLR with warmup|


## q2: SAM 2

### How to run in colab:
1. Run the first cell to install dependencies and download checkpoints.
2. Enable GPU in Runtime -> Change runtime type
3. Upload an image (sample.jpg) or use a Colab file path.
4. Provide a text prompt and view the segmentation mask overlay.

### Pipeline
1. Text Prompt → Region Seed: Use Grounding DINO (or CLIPSeg/GLIP) to find bounding boxes for the text prompt.
2. Seed → Segmentation Mask: Feed bounding boxes to SAM 2 to produce a fine-grained mask.
3. Visualization: Overlay the predicted mask on the input image.

### Checkpoints used
|Model|File|Download Command|
|:---:|:---:|:---:|
|SAM ViT-B|sam_vit_b_01ec64.pth|wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth|
|SAM ViT-L (optional)|sam_vit_l_0b3195.pth|HuggingFace repo|
|Grounding DINO|downloaded via bash download_ckpts.sh|from IDEA-Research GitHub

### Limitations
- Bounding box detection accuracy depends heavily on text model quality.
- Generic or ambiguous prompts (e.g. “object”) yield weak results.
- Requires GPU for real-time inference.
- Checkpoint URLs may change; check official repos if a link fails.
