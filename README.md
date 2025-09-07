# PyTorch Fundamentals 🚀

A quick beginner-friendly guide to PyTorch basics: tensors, math operations, reshaping, NumPy interoperability, random seeds, and GPU usage.

## Key Concepts

- **Scalars / Vectors / Matrices / Tensors** → `torch.tensor()`  
- **Create tensors** → `torch.rand()`, `torch.zeros()`, `torch.ones()`, `torch.arange()`  
- **Data types & devices** → `.type()`, `requires_grad=True`, `.to(device)`  
- **Math operations** → `+`, `-`, `*`, `/`, `torch.matmul()`  
- **Matrix operations** → `.mm()`, `.T`, `.reshape()`, `.view()`, `torch.stack()`  
- **Dimension operations** → `.squeeze()`, `.unsqueeze()`, `.permute()`  
- **NumPy ↔ PyTorch** → `torch.from_numpy()`, `.numpy()`  
- **Random seeds** → `torch.manual_seed()`  
- **GPU / CUDA** → `.cuda()`, `torch.cuda.is_available()`, `.cpu()`  

## Quick Start

```bash
pip install torch torchvision torchaudio
```
## Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
