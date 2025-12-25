# Featurization Optimization Checklist

A practical checklist for optimizing memory usage and training efficiency in ProteinWorkshop.

## 🎯 Quick Assessment

**What's your constraint?**
- [ ] **Limited GPU Memory (<16GB)** → Follow [Memory-First Optimizations](#memory-first-optimizations)
- [ ] **Slow Training Speed** → Follow [Speed-First Optimizations](#speed-first-optimizations)
- [ ] **Both** → Follow [Balanced Approach](#balanced-approach)
- [ ] **Neither** → Follow [Performance-First](#performance-first)

---

## 🔴 Memory-First Optimizations

Use this checklist if you're hitting OOM errors or have limited GPU memory.

### Phase 1: Immediate Relief (Expected: 50-70% memory reduction)

- [ ] **Switch to CA representation**
  ```yaml
  representation: CA  # Not BB or FA
  ```
  **Impact**: 4-30× memory reduction

- [ ] **Reduce k in KNN edges**
  ```yaml
  edge_types: [knn_10]  # Not knn_30 or knn_16
  ```
  **Impact**: 3× memory reduction

- [ ] **Use minimal features**
  ```yaml
  scalar_node_features: [amino_acid_one_hot]
  vector_node_features: []
  ```
  **Impact**: 2-3× memory reduction

- [ ] **Enable mixed precision**
  ```bash
  workshop train ... trainer.precision=16
  ```
  **Impact**: 30-40% memory reduction

- [ ] **Reduce batch size**
  ```yaml
  datamodule.batch_size: 8  # Or smaller
  ```
  **Impact**: Linear memory reduction

### Phase 2: Advanced Memory Optimization (Expected: Additional 20-30%)

- [ ] **Use gradient accumulation instead of large batch**
  ```python
  accumulation_steps = 4  # Effective batch = 8 × 4 = 32
  ```

- [ ] **Enable gradient checkpointing**
  ```python
  model = model.gradient_checkpointing_enable()
  ```
  **Impact**: 30-50% memory reduction, 10-20% slower

- [ ] **Use learned embeddings instead of one-hot**
  ```python
  # Replace 23D one-hot with 8D learned embedding
  self.aa_embedding = nn.Embedding(23, 8)
  ```
  **Impact**: ~65% reduction for this feature

- [ ] **Pre-allocate tensors and use in-place operations**
  ```python
  # In featurization, use in-place ops where possible
  tensor.mul_(factor)  # Instead of tensor = tensor * factor
  ```

- [ ] **Clean up GPU memory after each epoch**
  ```python
  from proteinworkshop.utils.memory_utils import clean_up_torch_gpu_memory
  clean_up_torch_gpu_memory()
  ```

### Phase 3: Extreme Measures (If still OOM)

- [ ] **Use CPU offloading**
  ```python
  model = model.cpu()  # Offload model to CPU between batches
  ```

- [ ] **Reduce sequence length**
  ```python
  # Filter dataset to shorter proteins
  dataset = dataset.filter(lambda x: x.coords.shape[0] < 200)
  ```

- [ ] **Use sequential edges only**
  ```yaml
  edge_types: [seq_forward]  # Minimal edges
  ```

---

## ⚡ Speed-First Optimizations

Use this checklist if training is too slow but memory is not an issue.

### Phase 1: Quick Wins (Expected: 2-5× speedup)

- [ ] **Use multiple workers for data loading**
  ```python
  train_loader = DataLoader(..., num_workers=4, pin_memory=True)
  ```
  **Impact**: 2-4× faster data loading

- [ ] **Enable mixed precision training**
  ```bash
  workshop train ... trainer.precision=16
  ```
  **Impact**: 1.5-2× faster training

- [ ] **Use persistent workers**
  ```python
  train_loader = DataLoader(..., persistent_workers=True)
  ```
  **Impact**: Eliminates worker restart overhead

- [ ] **Pin memory for faster GPU transfer**
  ```python
  train_loader = DataLoader(..., pin_memory=True)
  ```
  **Impact**: 10-20% faster

### Phase 2: Feature Caching (Expected: 3-10× speedup for data loading)

- [ ] **Pre-compute and cache features**
  ```python
  # Run once
  precomputed_data = precompute_features(datamodule, featuriser)
  torch.save(precomputed_data, 'cached_features.pt')
  
  # Then use cached data
  cached_data = torch.load('cached_features.pt')
  ```
  **Impact**: 3-10× faster data loading

- [ ] **Use memory-mapped datasets for large corpora**
  ```python
  import numpy as np
  data_mmap = np.memmap('features.dat', dtype='float32', mode='r', shape=...)
  ```

### Phase 3: Computation Optimization (Expected: 20-50% speedup)

- [ ] **Optimize edge construction**
  ```python
  # Avoid computing multiple KNN graphs
  # Instead of: edge_types = [knn_10, knn_16, knn_30]
  # Use: edge_types = [knn_30]  # Subsample if needed
  ```

- [ ] **Compile model with PyTorch 2.0+**
  ```python
  compiled_model = torch.compile(model, mode="reduce-overhead")
  ```
  **Impact**: 10-30% speedup

- [ ] **Use efficient attention mechanisms**
  ```python
  # For transformer-based models
  from torch.nn.functional import scaled_dot_product_attention
  # Uses flash attention when available
  ```

- [ ] **Profile and optimize bottlenecks**
  ```python
  from torch.profiler import profile, ProfilerActivity
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      model(batch)
  print(prof.key_averages().table())
  ```

---

## ⚖️ Balanced Approach

Use this when you need both reasonable memory and speed.

### Recommended Configuration

```yaml
# features.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features:
  - amino_acid_one_hot
  - dihedrals
  - kappa
vector_node_features: []
edge_types:
  - knn_16
scalar_edge_features:
  - edge_distance
vector_edge_features: []
```

```bash
# Training command
workshop train \
    dataset=cath \
    encoder=egnn \
    task=inverse_folding \
    features=balanced \
    datamodule.batch_size=32 \
    datamodule.num_workers=4 \
    trainer.precision=16 \
    trainer=gpu
```

### Checklist

- [ ] Use CA representation
- [ ] Use knn_16 (balanced connectivity)
- [ ] Include essential features (AA type, dihedrals, kappa)
- [ ] Enable mixed precision (16-bit)
- [ ] Use 4 workers for data loading
- [ ] Batch size 16-32 (adjust based on GPU)
- [ ] Pin memory for faster transfer
- [ ] Cache features if training multiple epochs

**Expected Results**:
- Memory usage: 4-8 GB for typical batch
- Training speed: 50-100 batches/min (V100)
- Performance: Near-optimal for most tasks

---

## 🏆 Performance-First

Use this when you have a large GPU (32GB+) and want maximum performance.

### Recommended Configuration

```yaml
# features.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: BB  # or CA if BB is too much
scalar_node_features:
  - amino_acid_one_hot
  - sequence_positional_encoding
  - dihedrals
  - alpha
  - kappa
  - sidechain_torsions
vector_node_features:
  - orientation
edge_types:
  - knn_30
  - seq_forward
scalar_edge_features:
  - edge_distance
  - edge_type
  - pos_emb
vector_edge_features:
  - edge_vectors
```

```bash
# Training command
workshop train \
    dataset=cath \
    encoder=gcpnet \
    task=inverse_folding \
    features=full \
    datamodule.batch_size=64 \
    datamodule.num_workers=8 \
    trainer.precision=16 \
    trainer=gpu
```

### Checklist

- [ ] Use BB or CA representation (based on task)
- [ ] Include all relevant features
- [ ] Use knn_30 for rich connectivity
- [ ] Add sequential edges for sequence information
- [ ] Large batch size (64+)
- [ ] Many workers (8+)
- [ ] Mixed precision still recommended
- [ ] Consider using multiple GPUs if available

---

## 📊 Optimization Impact Reference

| Optimization | Memory Reduction | Speed Improvement | Quality Impact |
|--------------|------------------|-------------------|----------------|
| CA → BB | -75% | +2× | Minimal |
| knn_30 → knn_16 | -47% | +1.5× | Small |
| knn_30 → knn_10 | -67% | +2× | Moderate |
| Mixed precision | -30% | +1.5× | Minimal |
| Gradient checkpointing | -40% | -15% | None |
| Feature caching | 0% | +5× | None |
| Gradient accumulation | Enables larger effective batch | Neutral | May improve |
| Learned embeddings | -5% | 0% | May improve |
| Multi-worker loading | 0% | +3× | None |
| Compiled model | 0% | +20% | None |

---

## 🔍 Profiling & Monitoring

### Memory Profiling

```python
import torch
from proteinworkshop.utils.memory_utils import gpu_memory_usage_all

def profile_memory(model, batch, num_iterations=10):
    """Profile peak memory usage."""
    torch.cuda.reset_peak_memory_stats()
    
    for _ in range(num_iterations):
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    usage, cache = gpu_memory_usage_all()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Current usage: {usage:.2f} GB")
    print(f"Cache: {cache:.2f} GB")
    print(f"Peak usage: {peak:.2f} GB")
    
    return peak

# Usage
peak_memory = profile_memory(model, batch)
```

### Speed Profiling

```python
import time
import torch

def profile_speed(dataloader, model, num_batches=50):
    """Profile training speed."""
    model.train()
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    # Profile
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    batches_per_sec = num_batches / elapsed
    time_per_batch = elapsed / num_batches
    
    print(f"Batches per second: {batches_per_sec:.2f}")
    print(f"Time per batch: {time_per_batch*1000:.2f} ms")
    
    return batches_per_sec

# Usage
throughput = profile_speed(train_loader, model)
```

### Combined Profiling

```python
def profile_training(model, dataloader, num_iterations=10):
    """Profile both memory and speed."""
    print("="*60)
    print("Training Profile")
    print("="*60)
    
    # Memory
    print("\n--- Memory ---")
    torch.cuda.reset_peak_memory_stats()
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    usage, cache = gpu_memory_usage_all()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Memory usage: {usage:.2f} GB")
    print(f"Peak memory: {peak:.2f} GB")
    
    # Speed
    print("\n--- Speed ---")
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    elapsed = time.time() - start_time
    
    print(f"Time per iteration: {elapsed/num_iterations*1000:.2f} ms")
    print(f"Throughput: {num_iterations/elapsed:.2f} batches/sec")
    
    print("="*60)
```

---

## ✅ Pre-Training Checklist

Before starting a long training run, verify:

- [ ] **Features are appropriate for your task**
  - Inverse folding: dihedrals, alpha, kappa
  - Fold classification: amino_acid_one_hot, dihedrals
  - Binding site: amino_acid_one_hot, sidechain_torsions

- [ ] **Memory usage is sustainable**
  - Run `profile_memory()` on a few batches
  - Ensure peak memory < 80% of GPU capacity

- [ ] **Training speed is acceptable**
  - Run `profile_speed()` on a few batches
  - Calculate estimated time to completion

- [ ] **Data loading is not a bottleneck**
  - Monitor GPU utilization (should be >80%)
  - If low, increase `num_workers`

- [ ] **Gradients are flowing**
  - Check gradient norms after first batch
  - Verify no NaN/Inf values

- [ ] **Checkpointing is configured**
  - Model checkpoints will be saved
  - Can resume from checkpoint if interrupted

- [ ] **Logging is set up**
  - WandB or TensorBoard configured
  - Tracking relevant metrics

---

## 🐛 Troubleshooting

### OOM Despite Following Checklist

1. **Check actual memory usage**:
   ```python
   nvidia-smi  # Check total GPU memory
   ```

2. **Find memory hogs**:
   ```python
   from proteinworkshop.utils.memory_utils import get_tensors
   for t in get_tensors():
       if t.numel() > 1_000_000:
           print(f"Large tensor: {t.shape}, {t.numel()*t.element_size()/1024**2:.2f} MB")
   ```

3. **Reduce further**:
   - Smaller batch size
   - Filter out large proteins
   - Use CPU for some operations

### Slow Training Despite Optimizations

1. **Check data loading**:
   ```python
   # Time data loading
   start = time.time()
   batch = next(iter(dataloader))
   print(f"Data loading time: {time.time()-start:.2f}s")
   ```
   If >0.5s, data loading is bottleneck → increase `num_workers`

2. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon -s u
   ```
   If <80%, GPU is underutilized → optimize data loading

3. **Profile model**:
   ```python
   from torch.profiler import profile
   with profile() as prof:
       model(batch)
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```
   Find and optimize slow operations

### Poor Model Performance

1. **Check if features are meaningful**:
   - Visualize feature distributions
   - Ensure no NaN/Inf values
   - Check if angles are in correct range

2. **Try richer featurization**:
   - Add more features incrementally
   - Increase k in KNN
   - Try different representation

3. **Verify task-feature alignment**:
   - Inverse folding needs structural features
   - Classification may need sequence features
   - Binding site needs detailed local features

---

## 📈 Progressive Optimization Strategy

**Phase 1: Baseline** (Week 1)
- [ ] Use standard configuration
- [ ] Train to convergence
- [ ] Establish baseline metrics

**Phase 2: Memory Optimization** (Week 2)
- [ ] Apply memory optimizations from checklist
- [ ] Maintain performance within 5% of baseline
- [ ] Document memory savings

**Phase 3: Speed Optimization** (Week 3)
- [ ] Apply speed optimizations from checklist
- [ ] Ensure no performance degradation
- [ ] Document speedup

**Phase 4: Refinement** (Week 4)
- [ ] Experiment with advanced techniques
- [ ] Fine-tune hyperparameters
- [ ] Achieve optimal efficiency/performance trade-off

---

## 🎯 Success Metrics

Track these metrics to measure optimization success:

| Metric | Baseline | Target | Achieved |
|--------|----------|--------|----------|
| Peak GPU memory (GB) | ___ | <50% of baseline | ___ |
| Training time (hrs) | ___ | <50% of baseline | ___ |
| Batches per second | ___ | >2× baseline | ___ |
| Final performance | ___ | ≥95% of baseline | ___ |
| GPU utilization (%) | ___ | >80% | ___ |

---

**Last Updated**: 2025-12-25  
**Version**: 1.0
