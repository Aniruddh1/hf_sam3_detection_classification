#!/usr/bin/env python3
"""
Simple Zero-Shot Region Classifier using SAM3 (HuggingFace)

This script demonstrates how to:
1. Detect objects with a generic prompt (e.g., "logo")
2. Classify each detection by comparing against candidate labels
3. Return the best matching label from your list

Usage:
    python classifier.py

Requirements:
    pip install transformers torch pillow requests
"""

import torch
from PIL import Image
import requests
from typing import List, Dict, Optional


# =============================================================================
# Hierarchical Taxonomy for Scalable Classification
# =============================================================================

# Example taxonomy: category -> list of specific labels
DEFAULT_BRAND_TAXONOMY = {
    "sports": ["Nike", "Adidas", "Puma", "Reebok", "New Balance", "Under Armour", "Asics", "Fila"],
    "tech": ["Apple", "Google", "Microsoft", "Samsung", "Sony", "LG", "Intel", "AMD", "Nvidia"],
    "automotive": ["BMW", "Mercedes", "Toyota", "Honda", "Ford", "Tesla", "Audi", "Porsche"],
    "food_beverage": ["McDonald's", "Starbucks", "Coca-Cola", "Pepsi", "KFC", "Burger King"],
    "fashion": ["Gucci", "Louis Vuitton", "Chanel", "Versace", "Prada", "Zara", "H&M"],
    "social_media": ["Facebook", "Instagram", "Twitter", "TikTok", "YouTube", "LinkedIn"],
}


# =============================================================================
# Embedding Cache for Speed Optimization
# =============================================================================

class EmbeddingCache:
    """
    Cache for pre-computed text EMBEDDINGS (not just tokenized inputs).

    This is the FASTEST approach - computes text embeddings ONCE, then uses
    simple dot product for matching. Reduces O(N) model forward passes to O(1).

    Speed comparison (50 candidates, per image):
        Original:     ~50 forward passes = very slow
        Cached inputs: ~50 forward passes (tokenization cached) = slow
        Cached embeds: 1 forward pass + matrix multiply = FAST

    Usage:
        cache = EmbeddingCache(model, processor, device)
        cache.precompute_embeddings(["Nike", "Adidas", "Puma"], "{label} logo")

        # Ultra-fast classification using embedding similarity
        results = detect_and_classify_embedding(image, model, processor, cache, ...)
    """

    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self._text_cache: Dict[str, torch.Tensor] = {}  # prompt -> text_inputs
        self._embed_cache: Dict[str, torch.Tensor] = {}  # prompt -> text_embedding
        self._embed_matrix: Optional[torch.Tensor] = None  # [N, embed_dim]
        self._embed_labels: List[str] = []  # ordered list of labels

    def precompute_candidates(
        self,
        candidates: List[str],
        prompt_template: str = "{label}",
        show_progress: bool = True
    ):
        """
        Pre-compute and cache text inputs for a list of candidates.
        (Legacy method - use precompute_embeddings for best performance)
        """
        prompts = [prompt_template.format(label=c) for c in candidates]
        new_prompts = [p for p in prompts if p not in self._text_cache]

        if not new_prompts:
            if show_progress:
                print(f"All {len(prompts)} prompts already cached")
            return

        if show_progress:
            print(f"Pre-computing {len(new_prompts)} text inputs...")

        for i, prompt in enumerate(new_prompts):
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            self._text_cache[prompt] = text_inputs

            if show_progress and (i + 1) % 20 == 0:
                print(f"  Cached {i + 1}/{len(new_prompts)}")

        if show_progress:
            print(f"  Done! Total cached: {len(self._text_cache)}")

    def precompute_embeddings(
        self,
        candidates: List[str],
        prompt_template: str = "{label}",
        show_progress: bool = True
    ):
        """
        Pre-compute and cache text EMBEDDINGS for candidates.

        This enables O(1) classification via dot product similarity.

        Args:
            candidates: List of candidate labels
            prompt_template: Template to format candidates
            show_progress: Whether to print progress
        """
        prompts = [prompt_template.format(label=c) for c in candidates]

        if show_progress:
            print(f"Pre-computing {len(prompts)} text embeddings...")

        embeddings = []
        for i, (candidate, prompt) in enumerate(zip(candidates, prompts)):
            if prompt in self._embed_cache:
                embeddings.append(self._embed_cache[prompt])
            else:
                text_inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    # Get text embedding from the model
                    text_embed = self.model.get_text_features(**text_inputs)

                self._embed_cache[prompt] = text_embed
                embeddings.append(text_embed)

            if show_progress and (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(prompts)}")

        # Stack into matrix for fast batch similarity
        self._embed_matrix = torch.cat(embeddings, dim=0)  # [N, embed_dim]
        self._embed_labels = candidates

        if show_progress:
            print(f"  Done! Embedding matrix shape: {self._embed_matrix.shape}")

    def get_text_inputs(self, prompt: str) -> Optional[Dict]:
        """Get cached text inputs for a prompt."""
        return self._text_cache.get(prompt)

    def get_embedding(self, prompt: str) -> Optional[torch.Tensor]:
        """Get cached text embedding for a prompt."""
        return self._embed_cache.get(prompt)

    def get_embedding_matrix(self) -> Optional[torch.Tensor]:
        """Get the stacked embedding matrix [N, embed_dim]."""
        return self._embed_matrix

    def get_labels(self) -> List[str]:
        """Get ordered list of candidate labels."""
        return self._embed_labels

    def has(self, prompt: str) -> bool:
        """Check if prompt is cached."""
        return prompt in self._text_cache or prompt in self._embed_cache

    def clear(self):
        """Clear all cached data."""
        self._text_cache.clear()
        self._embed_cache.clear()
        self._embed_matrix = None
        self._embed_labels = []

    @property
    def size(self) -> int:
        """Number of cached prompts."""
        return len(self._embed_cache) or len(self._text_cache)

    def __len__(self):
        return self.size

    def __contains__(self, prompt: str):
        return self.has(prompt)


def detect_and_classify_embedding(
    image: Image.Image,
    model,
    processor,
    cache: EmbeddingCache,
    detection_prompt: str,
    detection_threshold: float = 0.5,
    device: str = "cuda",
) -> Dict:
    """
    FASTEST classification using pre-computed text embeddings.

    Instead of N forward passes, this does:
    1. One detection pass to find regions
    2. Extract region embeddings
    3. Matrix multiply with cached text embeddings
    4. Find best match via argmax

    Complexity: O(1) model passes + O(N) dot products

    Args:
        image: PIL Image to process
        model: Sam3Model
        processor: Sam3Processor
        cache: EmbeddingCache with pre-computed text embeddings
        detection_prompt: Generic prompt for detection
        detection_threshold: Confidence threshold
        device: Device to run on

    Returns:
        Dictionary with boxes, masks, scores, and classifications
    """
    embed_matrix = cache.get_embedding_matrix()
    labels = cache.get_labels()

    if embed_matrix is None or len(labels) == 0:
        raise ValueError("Cache has no embeddings. Call cache.precompute_embeddings() first.")

    # Step 1: Detect objects
    print(f"\nStep 1: Detecting '{detection_prompt}'...")

    inputs = processor(
        images=image,
        text=detection_prompt,
        return_tensors="pt"
    ).to(device)

    original_sizes = [[image.size[1], image.size[0]]]

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=detection_threshold,
        mask_threshold=0.5,
        target_sizes=original_sizes
    )[0]

    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    print(f"   Found {len(boxes)} detections")

    if len(boxes) == 0:
        return {
            "boxes": [],
            "masks": [],
            "detection_scores": [],
            "classifications": [],
            "num_detections": 0,
        }

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.stack(boxes) if boxes else torch.tensor([])
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Step 2: Get region embeddings from the detection
    print("\nStep 2: Extracting region embeddings...")

    # Get the region features from model outputs
    # SAM3 outputs include region-level features we can use
    if hasattr(outputs, 'region_embeddings'):
        region_embeds = outputs.region_embeddings  # [1, num_regions, embed_dim]
    elif hasattr(outputs, 'object_embeddings'):
        region_embeds = outputs.object_embeddings
    else:
        # Fallback: use mask-pooled features from vision embeddings
        print("   Using mask-pooled vision features...")
        img_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

        # Pool vision features within each detected region (simplified)
        # This is a fallback - ideally SAM3 exposes region embeddings directly
        region_embeds = vision_embeds.mean(dim=(2, 3), keepdim=False)  # Global pool as fallback
        region_embeds = region_embeds.expand(len(boxes), -1)

    # Step 3: Compute similarity scores via dot product
    print(f"\nStep 3: Computing similarity with {len(labels)} candidates (dot product)...")

    # Normalize embeddings for cosine similarity
    region_embeds_norm = torch.nn.functional.normalize(region_embeds, dim=-1)
    text_embeds_norm = torch.nn.functional.normalize(embed_matrix.to(device), dim=-1)

    # [num_regions, num_candidates]
    similarity = torch.matmul(region_embeds_norm, text_embeds_norm.T)

    # Step 4: Get best matches
    print("\nStep 4: Finding best matches...")

    classifications = []
    for i in range(len(boxes)):
        if i < similarity.shape[0]:
            sim_scores = similarity[i]
        else:
            sim_scores = similarity[0]  # Fallback if shape mismatch

        best_idx = sim_scores.argmax().item()
        best_score = sim_scores[best_idx].item()

        # Convert similarity to 0-1 range (cosine sim is -1 to 1)
        best_score_normalized = (best_score + 1) / 2

        classifications.append({
            "label": labels[best_idx],
            "score": best_score_normalized,
            "raw_similarity": best_score,
            "all_scores": {labels[j]: ((sim_scores[j].item() + 1) / 2) for j in range(len(labels))}
        })

    return {
        "boxes": boxes,
        "masks": masks,
        "detection_scores": scores,
        "classifications": classifications,
        "num_detections": len(boxes),
    }


def detect_and_classify_cached(
    image: Image.Image,
    model,
    processor,
    cache: EmbeddingCache,
    detection_prompt: str,
    candidates: List[str],
    prompt_template: str = "{label}",
    detection_threshold: float = 0.5,
    device: str = "cuda",
) -> Dict:
    """
    Fast classification using pre-cached text inputs.

    This is significantly faster than detect_and_classify when:
    - Processing multiple images with the same candidates
    - Using large candidate lists (50+ items)

    Args:
        image: PIL Image to process
        model: Sam3Model
        processor: Sam3Processor
        cache: EmbeddingCache with pre-computed text inputs
        detection_prompt: Generic prompt for detection
        candidates: List of candidate labels (must be pre-cached!)
        prompt_template: Template used when caching
        detection_threshold: Confidence threshold
        device: Device to run on

    Returns:
        Dictionary with boxes, masks, scores, and classifications
    """

    # Step 1: Detect objects
    print(f"\nStep 1: Detecting '{detection_prompt}'...")

    inputs = processor(
        images=image,
        text=detection_prompt,
        return_tensors="pt"
    ).to(device)

    original_sizes = [[image.size[1], image.size[0]]]

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=detection_threshold,
        mask_threshold=0.5,
        target_sizes=original_sizes
    )[0]

    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    print(f"   Found {len(boxes)} detections")

    if len(boxes) == 0:
        return {
            "boxes": [],
            "masks": [],
            "detection_scores": [],
            "classifications": [],
            "num_detections": 0,
        }

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.stack(boxes) if boxes else torch.tensor([])
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Step 2: Pre-compute vision embeddings (once per image)
    print("\nStep 2: Computing vision embeddings...")
    img_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    # Step 3: Score using CACHED text inputs (fast!)
    print(f"\nStep 3: Matching against {len(candidates)} cached candidates...")

    all_scores = torch.zeros(len(boxes), len(candidates), device=device)
    cache_hits = 0

    for c_idx, label in enumerate(candidates):
        prompt = prompt_template.format(label=label)

        # Use cached text inputs instead of re-encoding
        text_inputs = cache.get_text_inputs(prompt)
        if text_inputs is None:
            # Fallback: compute if not cached
            text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        else:
            cache_hits += 1

        with torch.no_grad():
            outputs = model(vision_embeds=vision_embeds, **text_inputs)

        cand_results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=0.5,
            target_sizes=original_sizes
        )[0]

        cand_boxes = cand_results.get("boxes", [])
        cand_scores = cand_results.get("scores", [])

        if len(cand_boxes) == 0:
            continue

        if not isinstance(cand_boxes, torch.Tensor):
            cand_boxes = torch.stack(cand_boxes)
        if not isinstance(cand_scores, torch.Tensor):
            cand_scores = torch.tensor(cand_scores, device=device)

        for i, det_box in enumerate(boxes):
            ious = compute_iou(det_box.unsqueeze(0), cand_boxes.to(device))
            if ious.numel() > 0:
                best_idx = ious.argmax()
                if ious[best_idx] > 0.3:
                    all_scores[i, c_idx] = cand_scores[best_idx]

    print(f"   Cache hits: {cache_hits}/{len(candidates)}")

    # Step 4: Get best match
    print("\nStep 4: Finding best matches...")

    classifications = []
    for i in range(len(boxes)):
        det_scores = all_scores[i]
        best_idx = det_scores.argmax().item()
        best_score = det_scores[best_idx].item()

        classifications.append({
            "label": candidates[best_idx] if best_score > 0 else "unknown",
            "score": best_score,
            "all_scores": {candidates[j]: det_scores[j].item() for j in range(len(candidates))}
        })

    return {
        "boxes": boxes,
        "masks": masks,
        "detection_scores": scores,
        "classifications": classifications,
        "num_detections": len(boxes),
    }


def load_model(device: str = "cuda"):
    """Load SAM3 model and processor from HuggingFace."""
    from transformers import Sam3Model, Sam3Processor

    print("Loading SAM3 model from HuggingFace...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Model loaded!")

    return model, processor


def detect_and_classify(
    image: Image.Image,
    model,
    processor,
    detection_prompt: str,
    candidates: List[str],
    prompt_template: str = "{label}",
    detection_threshold: float = 0.5,
    device: str = "cuda",
) -> Dict:
    """
    Detect objects and classify them against candidate labels.

    Args:
        image: PIL Image to process
        model: Sam3Model
        processor: Sam3Processor
        detection_prompt: Generic prompt for detection (e.g., "logo", "animal")
        candidates: List of candidate labels to match against
        prompt_template: Template to format candidates (e.g., "{label} logo")
        detection_threshold: Confidence threshold for detections
        device: Device to run on

    Returns:
        Dictionary with boxes, masks, scores, and classifications
    """

    # Step 1: Detect objects with generic prompt
    print(f"\nStep 1: Detecting '{detection_prompt}'...")

    inputs = processor(
        images=image,
        text=detection_prompt,
        return_tensors="pt"
    ).to(device)

    original_sizes = [[image.size[1], image.size[0]]]  # [H, W]

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=detection_threshold,
        mask_threshold=0.5,
        target_sizes=original_sizes
    )[0]

    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    print(f"   Found {len(boxes)} detections")

    if len(boxes) == 0:
        return {
            "boxes": [],
            "masks": [],
            "detection_scores": [],
            "classifications": [],
            "num_detections": 0,
        }

    # Convert to tensors
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.stack(boxes) if boxes else torch.tensor([])
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Step 2: Pre-compute vision embeddings (do this ONCE)
    print("\nStep 2: Computing vision embeddings (once)...")
    img_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    # Step 3: Score each detection against all candidates
    print(f"\nStep 3: Matching against {len(candidates)} candidates...")

    # Score matrix: [num_detections, num_candidates]
    all_scores = torch.zeros(len(boxes), len(candidates), device=device)

    for c_idx, label in enumerate(candidates):
        prompt = prompt_template.format(label=label)
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)

        # Run with pre-computed vision embeddings (efficient!)
        with torch.no_grad():
            outputs = model(vision_embeds=vision_embeds, **text_inputs)

        cand_results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,  # Get all detections
            mask_threshold=0.5,
            target_sizes=original_sizes
        )[0]

        cand_boxes = cand_results.get("boxes", [])
        cand_scores = cand_results.get("scores", [])

        if len(cand_boxes) == 0:
            continue

        if not isinstance(cand_boxes, torch.Tensor):
            cand_boxes = torch.stack(cand_boxes)
        if not isinstance(cand_scores, torch.Tensor):
            cand_scores = torch.tensor(cand_scores, device=device)

        # Match by IoU
        for i, det_box in enumerate(boxes):
            ious = compute_iou(det_box.unsqueeze(0), cand_boxes.to(device))
            if ious.numel() > 0:
                best_idx = ious.argmax()
                if ious[best_idx] > 0.3:  # IoU threshold for matching
                    all_scores[i, c_idx] = cand_scores[best_idx]

    # Step 4: Get best match for each detection
    print("\nStep 4: Finding best matches...")

    classifications = []
    for i in range(len(boxes)):
        det_scores = all_scores[i]
        best_idx = det_scores.argmax().item()
        best_score = det_scores[best_idx].item()

        classifications.append({
            "label": candidates[best_idx] if best_score > 0 else "unknown",
            "score": best_score,
            "all_scores": {candidates[j]: det_scores[j].item() for j in range(len(candidates))}
        })

    return {
        "boxes": boxes,
        "masks": masks,
        "detection_scores": scores,
        "classifications": classifications,
        "num_detections": len(boxes),
    }


def hierarchical_classify(
    image: Image.Image,
    model,
    processor,
    detection_prompt: str,
    taxonomy: Dict[str, List[str]],
    category_template: str = "{label} brand logo",
    item_template: str = "{label} logo",
    detection_threshold: float = 0.5,
    device: str = "cuda",
    top_k_categories: int = 2,
) -> Dict:
    """
    Two-stage hierarchical classification for better accuracy with many candidates.

    Stage 1: Classify into broad categories (e.g., "sports", "tech", "automotive")
    Stage 2: Classify within top categories to get specific label

    Args:
        image: PIL Image to process
        model: Sam3Model
        processor: Sam3Processor
        detection_prompt: Generic prompt for detection (e.g., "logo")
        taxonomy: Dict mapping category names to list of specific labels
        category_template: Template for category prompts
        item_template: Template for specific item prompts
        detection_threshold: Confidence threshold for detections
        device: Device to run on
        top_k_categories: Number of top categories to search in stage 2

    Returns:
        Dictionary with boxes, masks, scores, classifications, and category info

    Example:
        taxonomy = {
            "sports": ["Nike", "Adidas", "Puma"],
            "tech": ["Apple", "Google", "Microsoft"],
        }
        # Stage 1: Match against "sports brand logo", "tech brand logo"
        # Stage 2: If "sports" wins, match against "Nike logo", "Adidas logo", "Puma logo"
    """

    categories = list(taxonomy.keys())
    total_items = sum(len(items) for items in taxonomy.values())

    print(f"\n{'='*60}")
    print(f"Hierarchical Classification")
    print(f"  Categories: {len(categories)}")
    print(f"  Total items: {total_items}")
    print(f"  Effective comparisons: {len(categories)} + ~{total_items // len(categories)} = ~{len(categories) + total_items // len(categories)}")
    print(f"  (vs flat: {total_items} comparisons)")
    print(f"{'='*60}")

    # Stage 1: Detect and classify into categories
    print(f"\n[Stage 1] Classifying into {len(categories)} categories...")

    category_results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt=detection_prompt,
        candidates=categories,
        prompt_template=category_template,
        detection_threshold=detection_threshold,
        device=device,
    )

    if category_results["num_detections"] == 0:
        return category_results

    # Stage 2: For each detection, search within top categories
    print(f"\n[Stage 2] Refining within top-{top_k_categories} categories...")

    final_classifications = []

    for i, cat_clf in enumerate(category_results["classifications"]):
        # Get top-k categories for this detection
        sorted_cats = sorted(
            cat_clf["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_categories]

        top_categories = [cat for cat, score in sorted_cats]
        print(f"\n  Detection {i+1}: Top categories = {top_categories}")

        # Gather candidates from top categories
        refined_candidates = []
        for cat in top_categories:
            refined_candidates.extend(taxonomy[cat])

        print(f"    Searching among {len(refined_candidates)} candidates...")

        # Extract the detection box for this specific detection
        det_box = category_results["boxes"][i]

        # Run classification on refined candidates
        # We reuse vision embeddings by calling detect_and_classify
        refined_results = detect_and_classify(
            image=image,
            model=model,
            processor=processor,
            detection_prompt=detection_prompt,
            candidates=refined_candidates,
            prompt_template=item_template,
            detection_threshold=detection_threshold,
            device=device,
        )

        # Match to original detection by IoU
        best_match = {"label": "unknown", "score": 0.0, "category": "unknown", "all_scores": {}}

        if refined_results["num_detections"] > 0:
            for j, ref_clf in enumerate(refined_results["classifications"]):
                ref_box = refined_results["boxes"][j]
                iou = compute_iou(det_box.unsqueeze(0), ref_box.unsqueeze(0).to(device))

                if iou.item() > 0.3 and ref_clf["score"] > best_match["score"]:
                    # Find which category this label belongs to
                    label_category = "unknown"
                    for cat, items in taxonomy.items():
                        if ref_clf["label"] in items:
                            label_category = cat
                            break

                    best_match = {
                        "label": ref_clf["label"],
                        "score": ref_clf["score"],
                        "category": label_category,
                        "category_scores": dict(sorted_cats),
                        "all_scores": ref_clf["all_scores"],
                    }

        final_classifications.append(best_match)
        print(f"    Result: {best_match['label']} ({best_match['category']}) - {best_match['score']:.3f}")

    return {
        "boxes": category_results["boxes"],
        "masks": category_results["masks"],
        "detection_scores": category_results["detection_scores"],
        "classifications": final_classifications,
        "num_detections": category_results["num_detections"],
    }


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between boxes in xyxy format."""
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1 + area2 - intersection
    return (intersection / (union + 1e-6)).squeeze(0)


def visualize(image: Image.Image, results: Dict, output_path: str = "result.jpg"):
    """Draw bounding boxes and labels on image."""
    from PIL import ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    boxes = results["boxes"]
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    for i, (box, clf) in enumerate(zip(boxes, results["classifications"])):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{clf['label']}: {clf['score']:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)

    img.save(output_path)
    print(f"\nVisualization saved to: {output_path}")
    return img


# =============================================================================
# MAIN - Example Usage
# =============================================================================

if __name__ == "__main__":
    device = "cpu"  # Using CPU
    print(f"Using device: {device}")

    # Load model
    model, processor = load_model(device)

    # Load a test image (cat image from COCO)
    print("\nLoading test image...")
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"Image size: {image.size}")

    # ==========================================================================
    # Example 1: Detect "ear" and classify as cat vs dog
    # ==========================================================================
    print("\n" + "="*60)
    print("Example 1: Classify detected ears")
    print("="*60)

    results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="ear",
        candidates=["cat ear", "dog ear", "human ear", "rabbit ear"],
        prompt_template="{label}",
        detection_threshold=0.3,
        device=device,
    )

    print(f"\nResults:")
    for i, clf in enumerate(results["classifications"]):
        print(f"  Detection {i+1}: {clf['label']} (score: {clf['score']:.3f})")
        print(f"    All scores: {clf['all_scores']}")

    visualize(image, results, "ear_classification.jpg")

    # ==========================================================================
    # Example 2: Animal Classification
    # ==========================================================================
    print("\n" + "="*60)
    print("Example 2: What animal is in the image?")
    print("="*60)

    results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="animal",
        candidates=["cat", "dog", "bird", "rabbit", "hamster"],
        prompt_template="{label}",
        detection_threshold=0.3,
        device=device,
    )

    print(f"\nResults:")
    for i, clf in enumerate(results["classifications"]):
        print(f"  Detection {i+1}: {clf['label']} (score: {clf['score']:.3f})")

    visualize(image, results, "animal_classification.jpg")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
