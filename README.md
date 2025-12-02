# SAM3 Zero-Shot Detection & Classification

Use Meta's SAM3 model for **object detection + classification** in a single pipeline. Detect objects with a generic prompt (e.g., "logo") and classify them against your custom candidate labels.

## Key Insight

SAM3 internally uses contrastive matching between image and text embeddings. If it can understand "Nike logo" as input, we can leverage this to identify which brand a detected logo belongs to.

## Installation

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Login to HuggingFace (SAM3 is a gated model)
huggingface-cli login
```

## Quick Start

```python
from classifier import load_model, detect_and_classify

model, processor = load_model(device="cpu")

results = detect_and_classify(
    image=image,
    model=model,
    processor=processor,
    detection_prompt="logo",
    candidates=["Nike", "Adidas", "Puma"],
    prompt_template="{label} logo",
)

for clf in results["classifications"]:
    print(f"{clf['label']}: {clf['score']:.3f}")
```

## Classification Methods

### 1. Basic (`detect_and_classify`)

Simple but slow - runs N forward passes for N candidates.

```python
results = detect_and_classify(
    image, model, processor,
    detection_prompt="logo",
    candidates=["Nike", "Adidas", "Puma"],
    prompt_template="{label} logo",
)
```

### 2. Cached Embeddings (`detect_and_classify_embedding`) - FASTEST

Pre-compute text embeddings once, use dot product for matching. **O(1) forward pass**.

```python
from classifier import EmbeddingCache, detect_and_classify_embedding

# One-time setup
cache = EmbeddingCache(model, processor, device)
cache.precompute_embeddings(["Nike", "Adidas", "Puma"], "{label} logo")

# Fast inference
results = detect_and_classify_embedding(
    image, model, processor, cache,
    detection_prompt="logo",
)
```

### 3. Hierarchical (`hierarchical_classify`) - MOST ACCURATE

Two-stage classification for large candidate sets. First categorize, then identify.

```python
from classifier import hierarchical_classify, DEFAULT_BRAND_TAXONOMY

results = hierarchical_classify(
    image, model, processor,
    detection_prompt="logo",
    taxonomy=DEFAULT_BRAND_TAXONOMY,  # 48 brands in 6 categories
    top_k_categories=2,
)
```

## Performance Comparison

| Method | Complexity | 50 Candidates | Best For |
|--------|-----------|---------------|----------|
| Basic | O(N) passes | ~60s | Single use, small N |
| Cached Embeddings | O(1) pass | ~2s | Batch processing |
| Hierarchical | O(cat + items/cat) | ~20s | Large N, best accuracy |

## Scalability Guidelines

| Candidates | Recommended Method | Expected Accuracy |
|------------|-------------------|-------------------|
| < 20 | Basic | ~90% |
| 20-50 | Cached Embeddings | ~80% |
| 50-200 | Hierarchical | ~75% |
| 200+ | Hierarchical + fine-tuning | Varies |

## Custom Taxonomy

Define your own categories for hierarchical classification:

```python
my_taxonomy = {
    "vehicles": ["car", "truck", "motorcycle", "bus"],
    "animals": ["dog", "cat", "bird", "horse"],
    "furniture": ["chair", "table", "sofa", "bed"],
}

results = hierarchical_classify(
    image, model, processor,
    detection_prompt="object",
    taxonomy=my_taxonomy,
    category_template="{label}",
    item_template="{label}",
)
```

## Output Format

All methods return:

```python
{
    "boxes": tensor,           # [N, 4] bounding boxes (xyxy)
    "masks": list,             # Binary segmentation masks
    "detection_scores": tensor, # Detection confidence
    "classifications": [       # Per-detection classification
        {
            "label": "Nike",
            "score": 0.87,
            "all_scores": {"Nike": 0.87, "Adidas": 0.12, ...}
        },
        ...
    ],
    "num_detections": int,
}
```

## Examples

```bash
# Basic classification
python classifier.py

# Test logo detection
python test_logos.py

# Compare all methods
python test_hierarchical.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers (main branch for SAM3 support)
- HuggingFace account with SAM3 access

## Files

- `classifier.py` - Main module with all classification methods
- `test_logos.py` - Logo/brand detection example
- `test_hierarchical.py` - Performance comparison of all methods

## License

MIT
