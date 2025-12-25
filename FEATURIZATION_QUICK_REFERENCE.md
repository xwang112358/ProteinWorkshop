# Featurization Module - Quick Reference Card

## 🚀 Quick Start

```python
from proteinworkshop.features.factory import ProteinFeaturiser

# Create featuriser
featuriser = ProteinFeaturiser(
    representation="CA",                           # CA, BB, or FA
    scalar_node_features=["amino_acid_one_hot"],  # Invariant features
    vector_node_features=[],                       # Equivariant features
    edge_types=["knn_16"],                        # Edge construction
    scalar_edge_features=["edge_distance"],       # Edge properties
    vector_edge_features=[],                       # Equivariant edge features
)

# Apply to batch
featurized_batch = featuriser(batch)
```

## 📊 Feature Dimensions Reference

### Scalar Node Features

| Feature | Dimension | Description | Typical Use |
|---------|-----------|-------------|-------------|
| `amino_acid_one_hot` | 23 | One-hot AA encoding | ✅ Always use |
| `sequence_positional_encoding` | 16 | Transformer-style encoding | Sequence models |
| `dihedrals` | 6 | Backbone φ, ψ, ω angles | ✅ Structural models |
| `alpha` | 2 | Virtual torsion angle | Structural detail |
| `kappa` | 2 | Virtual bond angle | Structural detail |
| `sidechain_torsions` | 8 | χ₁-χ₄ angles | High detail |

### Vector Node Features

| Feature | Dimension | Description | Typical Use |
|---------|-----------|-------------|-------------|
| `orientation` | 2 × 3D | Forward/backward vectors | Equivariant models |

### Edge Types

| Type | Syntax | Edges | Description | Memory |
|------|--------|-------|-------------|--------|
| KNN | `knn_10` | ~10N | 10 nearest neighbors | Low |
| KNN | `knn_16` | ~16N | 16 nearest neighbors | Medium |
| KNN | `knn_30` | ~30N | 30 nearest neighbors | High |
| Epsilon | `eps_8` | Variable | Within 8Å | Medium-High |
| Sequential | `seq_forward` | ~N | i→i+1 | Minimal |
| Sequential | `seq_backward` | ~N | i+1→i | Minimal |

### Scalar Edge Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `edge_distance` | 1 | Euclidean distance |
| `node_features` | 2×F | Concatenated node features |
| `edge_type` | 1 | Edge type index |
| `sequence_distance` | 1 | Sequence separation |
| `pos_emb` | 16 | Positional embedding |

### Vector Edge Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `edge_vectors` | 1 × 3D | Unit directional vector |

## 🎯 Common Configurations

### Minimal (Fast & Low Memory)
```yaml
representation: CA
scalar_node_features: [amino_acid_one_hot]
vector_node_features: []
edge_types: [knn_10]
scalar_edge_features: [edge_distance]
vector_edge_features: []
```
**Use**: Prototyping, limited GPU

### Standard Invariant
```yaml
representation: CA
scalar_node_features: [amino_acid_one_hot, dihedrals, kappa]
vector_node_features: []
edge_types: [knn_16]
scalar_edge_features: [edge_distance, edge_type]
vector_edge_features: []
```
**Use**: GearNet, DimeNet++, SchNet

### Equivariant
```yaml
representation: CA
scalar_node_features: [dihedrals]
vector_node_features: [orientation]
edge_types: [knn_30]
scalar_edge_features: [edge_distance]
vector_edge_features: [edge_vectors]
```
**Use**: EGNN, GVP-GNN, TFN

### Full Detail
```yaml
representation: BB
scalar_node_features: [amino_acid_one_hot, dihedrals, alpha, kappa, sidechain_torsions]
vector_node_features: [orientation]
edge_types: [knn_30]
scalar_edge_features: [edge_distance, edge_type, pos_emb]
vector_edge_features: [edge_vectors]
```
**Use**: Maximum performance, large GPU

## 💾 Memory Hierarchy

| Configuration | Relative Memory | Speed | Quality |
|---------------|-----------------|-------|---------|
| CA + knn_10 + minimal features | 1× | ⚡⚡⚡ | ⭐⭐ |
| CA + knn_16 + standard features | 2× | ⚡⚡ | ⭐⭐⭐ |
| CA + knn_30 + full features | 4× | ⚡ | ⭐⭐⭐⭐ |
| BB + knn_16 + standard features | 8× | 💨 | ⭐⭐⭐⭐ |
| BB + knn_30 + full features | 16× | 🐌 | ⭐⭐⭐⭐⭐ |
| FA + knn_10 + minimal features | 30× | 🐌🐌 | ⭐⭐⭐⭐⭐ |

## 🎨 Representation Comparison

```
┌─────────────────────────────────────────────────────┐
│ CA (C-alpha only)          Memory: 1×    Speed: ⚡⚡⚡ │
│ ○────○────○────○           Nodes: N                │
│                                                     │
│ BB (Backbone)              Memory: 4×    Speed: ⚡⚡  │
│ ●─●─●─●──●─●─●─●           Nodes: 4N               │
│ N Cα C O  N Cα C O                                  │
│                                                     │
│ FA (Full Atom)             Memory: 30×   Speed: ⚡   │
│ ●●●●●●●●●●●●●●●●●          Nodes: ~20N              │
│ (all atoms)                                         │
└─────────────────────────────────────────────────────┘
```

## ⚡ Performance Tips

### Memory Optimization
1. ✅ Use `CA` instead of `BB`/`FA`
2. ✅ Start with `knn_10`, increase if needed
3. ✅ Enable mixed precision: `trainer.precision=16`
4. ✅ Reduce batch size, use gradient accumulation
5. ✅ Use learned embeddings (8D) instead of one-hot (23D)

### Speed Optimization
1. ✅ Pre-compute features and cache
2. ✅ Use multiple workers: `num_workers=4`
3. ✅ Enable mixed precision training
4. ✅ Use `pin_memory=True` in DataLoader
5. ✅ Compile with PyTorch 2.0: `torch.compile(model)`

## 🔧 CLI Commands

### Train with custom features
```bash
workshop train \
    dataset=cath \
    encoder=egnn \
    task=inverse_folding \
    features=ca_bb \
    trainer=gpu
```

### Override specific settings
```bash
workshop train \
    dataset=cath \
    encoder=egnn \
    task=inverse_folding \
    features.representation=CA \
    features.edge_types=[knn_16,seq_forward] \
    trainer.precision=16
```

### Create custom config
```bash
# Create: proteinworkshop/config/features/my_features.yaml
_target_: proteinworkshop.features.factory.ProteinFeaturiser
representation: CA
scalar_node_features: [amino_acid_one_hot, dihedrals]
vector_node_features: []
edge_types: [knn_16]
scalar_edge_features: [edge_distance]
vector_edge_features: []

# Use it
workshop train dataset=cath encoder=egnn task=inverse_folding features=my_features
```

## 🐛 Common Issues & Solutions

### Issue: Out of Memory (OOM)
**Solutions**:
- Reduce batch size: `datamodule.batch_size=8`
- Use smaller k: `knn_10` instead of `knn_30`
- Switch to CA: `representation=CA`
- Enable mixed precision: `trainer.precision=16`
- Gradient accumulation: `accumulation_steps=4`

### Issue: Slow training
**Solutions**:
- Increase num_workers: `datamodule.num_workers=4`
- Pre-compute features and cache
- Use mixed precision
- Reduce feature computation

### Issue: Poor performance
**Solutions**:
- Add more features: `dihedrals`, `alpha`, `kappa`
- Increase k: `knn_30` instead of `knn_10`
- Try different representation: `BB` instead of `CA`
- Add sequential edges: `[knn_16, seq_forward]`

## 📈 Feature Selection Guide

### Task-Specific Recommendations

**Inverse Folding** (sequence from structure):
```yaml
scalar_node_features: [dihedrals, alpha, kappa]
edge_types: [knn_30]
```

**Fold Classification**:
```yaml
scalar_node_features: [amino_acid_one_hot, dihedrals]
edge_types: [knn_16]
```

**Protein-Protein Interaction**:
```yaml
scalar_node_features: [amino_acid_one_hot, sequence_positional_encoding]
edge_types: [knn_16, seq_forward]
```

**Binding Site Prediction**:
```yaml
representation: BB  # More detail needed
scalar_node_features: [amino_acid_one_hot, dihedrals, sidechain_torsions]
edge_types: [knn_30]
```

## 🔬 Debugging

### Check feature shapes
```python
print(f"Node features: {batch.x.shape}")  # Should be (N, F)
print(f"Edge index: {batch.edge_index.shape}")  # Should be (2, E)
print(f"Edge features: {batch.edge_attr.shape}")  # Should be (E, F_e)
print(f"Positions: {batch.pos.shape}")  # Should be (N, 3)
```

### Visualize features
```python
import matplotlib.pyplot as plt

# Visualize edge distribution
plt.hist(batch.edge_attr[:, 0].cpu(), bins=50)
plt.xlabel("Edge Distance")
plt.title("Edge Distance Distribution")
plt.show()

# Check degree distribution
from torch_geometric.utils import degree
deg = degree(batch.edge_index[0], batch.num_nodes)
plt.hist(deg.cpu(), bins=30)
plt.xlabel("Node Degree")
plt.title("Degree Distribution")
plt.show()
```

## 📚 Resources

- **Documentation**: https://www.proteins.sh
- **Repository**: https://github.com/a-r-j/ProteinWorkshop
- **Paper**: [Evaluating Representation Learning on the Protein Structure Universe](https://openreview.net/forum?id=sTYuRVrdK3) (ICLR 2024)
- **Feature Guide**: `/workspace/FEATURIZATION_WORKFLOW_GUIDE.md`
- **Examples**: `/workspace/examples/featurization_examples.py`

## 🎓 Learning Path

1. **Start**: Minimal config → Train small model → Understand basics
2. **Explore**: Try different features → Compare performance
3. **Optimize**: Profile memory/speed → Apply optimizations
4. **Advanced**: Custom featurizers → Multi-scale representations

---

**Last Updated**: 2025-12-25  
**Version**: ProteinWorkshop v1.0+
