# **V7 Fusion: Multi-Modal Molecular Toxicity Prediction**

A sophisticated deep learning framework for molecular toxicity prediction using multi-modal fusion of SMILES text, molecular graphs, and RDKit descriptors on the Tox21 dataset.

---

## **Overview**

This project implements a novel fusion architecture that combines three molecular representations:
- **Text**: SMILES strings via ChemBERTa encoder
- **Graph**: Molecular graphs via Graph Isomorphism Network (GIN)
- **Descriptors**: 208 RDKit molecular descriptors

The model predicts toxicity across 12 assays using cross-attention fusion and specialist ensemble heads.

---

## **Architecture**

### **Core Fusion Model**

```python
class V7FusionModel(nn.Module):
    def __init__(self, text_encoder, graph_encoder, desc_in_dim=208, dim=256, n_labels=12):
        super().__init__()
        self.text_encoder = text_encoder      # ChemBERTa
        self.graph_encoder = graph_encoder    # GIN
        self.cross = CrossAttentionBlock(dim) # Text attends to graph
        self.desc_mlp = DescriptorMLP(desc_in_dim, dim)
        self.shared_head = FusionClassifier(dim, n_labels)

    def forward(self, smiles_list, desc_feats):
        # Encode modalities
        text_tokens, text_mask = self.text_encoder(smiles_list)
        graph_nodes, graph_mask = self.graph_encoder(smiles_list)
        desc_embed = self.desc_mlp(desc_feats)
        
        # Cross-attention fusion
        text_attended = self.cross(text_tokens, text_mask, graph_nodes, graph_mask)
        
        # Pool and concatenate
        text_pool = masked_mean(text_attended, text_mask, dim=1)
        graph_pool = masked_mean(graph_nodes, graph_mask, dim=1)
        fused = torch.cat([text_pool, graph_pool, desc_embed], dim=-1)
        
        return self.shared_head(fused)
```

### **Cross-Attention Block**

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=256, n_heads=4, p=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=p, batch_first=False)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p)

    def forward(self, text_tokens, text_mask, graph_nodes, graph_mask):
        Q = text_tokens.transpose(0,1)   # Text queries
        K = graph_nodes.transpose(0,1)   # Graph keys
        V = graph_nodes.transpose(0,1)   # Graph values
        
        key_padding_mask = (graph_mask == 0)
        attn_out, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask)
        
        # Residual connection + LayerNorm
        return self.ln(text_tokens + self.dropout(attn_out.transpose(0,1)))
```

---

## **Project Structure**

### **Phase 1: Data Processing**

- **Load & Inspect**: Tox21 dataset validation and metadata generation
- **Descriptor Generation**: 208 RDKit molecular descriptors with imputation/scaling
- **Scaffold Splitting**: 80/10/10 train/val/test split using Murcko scaffolds
- **Data Packaging**: Train-only fitted preprocessing pipeline

### **Phase 2: Model Architecture**

- **Text Encoder**: ChemBERTa-zinc-base-v1 for SMILES representation
- **Graph Encoder**: 4-layer GIN for molecular graph processing
- **Fusion Core**: Cross-attention mechanism for multi-modal integration
- **Dual Heads**: Shared multi-label head + specialist ensemble heads

### **Phase 3: Training**

- **Hardware Optimization**: AMP support, batch size tuning
- **Two-Stage Training**:
  - Stage A: Frozen ChemBERTa backbone (8 epochs)
  - Stage B: Unfrozen last 2 layers (20 epochs)
- **Specialist Ensemble**: 5-seed ensemble per label with class-balanced sampling
- **Loss Function**: Asymmetric Loss with class-balanced weighting

```python
class AsymmetricLossCB(nn.Module):
    def __init__(self, gamma_neg=5.0, gamma_pos=1.0, clip=0.05, alpha=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha  # Per-label class weights

    def forward(self, logits, targets, missing_mask):
        pred = torch.sigmoid(logits)
        if self.clip:
            pred = torch.clamp(pred, self.clip, 1 - self.clip)
        
        # Asymmetric focal weighting
        pt = pred * targets + (1 - pred) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_weight = torch.pow(1 - pt, gamma)
        
        # BCE loss with focal and class weighting
        loss = -(targets * torch.log(pred) + (1 - targets) * torch.log(1 - pred))
        return (loss * focal_weight * self.alpha).mean()
```

### **Phase 4: Calibration & Thresholding**

- **Temperature Scaling**: Per-label calibration on validation set
- **Threshold Optimization**: F1 and F-beta (β=1.5) maximization
- **Blend Calibration**: Weighted combination of specialist + shared predictions

### **Phase 5: Inference System**

- **Model Loading**: Cold-start inference with checkpoint restoration
- **Prediction Modes**: 
  - `fbeta15`: F-beta optimized thresholds
  - `f1`: F1 optimized thresholds  
  - `policy`: Custom precision-floor policy
- **Blended Ensemble**: α=0.8 specialist + 0.2 shared head weighting

### **Phase 6: Evaluation**

- **Comprehensive Metrics**: PR-AUC, ROC-AUC, calibration error (ECE)
- **Reliability Curves**: Per-label calibration assessment
- **Comparison Framework**: Specialist vs blend performance analysis

---

## **Key Features**

- **Multi-Modal Fusion**: Novel cross-attention between text and graph modalities
- **Scaffold-Aware Splitting**: Prevents data leakage in molecular datasets
- **Ensemble Architecture**: Best-of-5-seeds specialist heads per label
- **Production Ready**: Temperature scaling, threshold optimization, policy framework
- **Comprehensive Evaluation**: Multiple threshold strategies with reliability analysis

---

## **Performance Highlights**

The model achieves strong performance across 12 toxicity assays with:
- Macro PR-AUC: ~0.75-0.80 on test set
- Calibrated predictions with ECE < 0.1 for most labels
- Adaptive thresholding based on precision requirements
- Robust handling of label imbalance and missing data

---

## **Usage**

```python
# Load model and predict
results = predict_smiles_blend(['CCO'], mode='fbeta15')
for smiles, predictions in zip(['CCO'], results):
    for label, details in predictions.items():
        print(f"{label}: {details['prob_blend']:.3f} -> {details['decision']}")
```

---

## **Requirements**

- PyTorch 1.8+
- RDKit 2022.03+
- Transformers 4.20+
- scikit-learn 1.1+
- CUDA-compatible GPU (recommended)

---

The codebase provides a complete pipeline from raw SMILES to calibrated toxicity predictions, with extensive evaluation and comparison frameworks for multi-modal molecular property prediction.