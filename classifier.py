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
from typing import List, Dict


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
