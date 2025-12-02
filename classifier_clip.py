#!/usr/bin/env python3
"""
Hybrid SAM3 + CLIP Classifier

SAM3 = Detection (WHERE are the objects?)
CLIP = Classification (WHAT are they?)

This is more accurate than using SAM3 alone because:
- SAM3 is trained for segmentation, not classification
- CLIP is specifically trained for image-text similarity
"""

import torch
from PIL import Image
import requests
from typing import List, Dict, Optional


def load_sam3(device: str = "cuda"):
    """Load SAM3 model for detection."""
    from transformers import Sam3Model, Sam3Processor

    print("Loading SAM3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM3 loaded!")
    return model, processor


def load_clip(device: str = "cuda", model_name: str = "openai/clip-vit-large-patch14"):
    """
    Load CLIP model for classification.

    Options:
    - openai/clip-vit-base-patch32 (fastest)
    - openai/clip-vit-large-patch14 (balanced)
    - openai/clip-vit-large-patch14-336 (best quality)
    - google/siglip-base-patch16-224 (better than CLIP for many tasks)
    """
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("CLIP loaded!")
    return model, processor


def detect_regions(
    image: Image.Image,
    sam3_model,
    sam3_processor,
    detection_prompt: str,
    threshold: float = 0.3,
    device: str = "cuda",
) -> Dict:
    """Use SAM3 to detect regions."""

    inputs = sam3_processor(
        images=image,
        text=detection_prompt,
        return_tensors="pt"
    ).to(device)

    original_sizes = [[image.size[1], image.size[0]]]

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=original_sizes
    )[0]

    boxes = results.get("boxes", [])
    masks = results.get("masks", [])
    scores = results.get("scores", [])

    if not isinstance(boxes, torch.Tensor) and len(boxes) > 0:
        boxes = torch.stack(boxes)
    if not isinstance(scores, torch.Tensor) and len(scores) > 0:
        scores = torch.tensor(scores)

    return {
        "boxes": boxes,
        "masks": masks,
        "scores": scores,
        "num_detections": len(boxes) if hasattr(boxes, '__len__') else 0,
    }


def crop_regions(image: Image.Image, boxes: torch.Tensor, padding: int = 10) -> List[Image.Image]:
    """Crop detected regions from image."""
    crops = []
    w, h = image.size

    if not isinstance(boxes, torch.Tensor) or len(boxes) == 0:
        return crops

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()

        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops


def classify_with_clip(
    crops: List[Image.Image],
    candidates: List[str],
    clip_model,
    clip_processor,
    prompt_template: str = "a photo of {} logo",
    device: str = "cuda",
) -> List[Dict]:
    """Classify cropped regions using CLIP."""

    if not crops:
        return []

    # Prepare text prompts
    text_prompts = [prompt_template.format(c) for c in candidates]

    classifications = []

    for crop in crops:
        # Process image and text
        inputs = clip_processor(
            images=crop,
            text=text_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        # Get similarity scores
        logits_per_image = outputs.logits_per_image  # [1, num_candidates]
        probs = logits_per_image.softmax(dim=1)[0]  # Convert to probabilities

        best_idx = probs.argmax().item()
        best_score = probs[best_idx].item()

        classifications.append({
            "label": candidates[best_idx],
            "score": best_score,
            "all_scores": {candidates[i]: probs[i].item() for i in range(len(candidates))},
        })

    return classifications


def detect_and_classify_hybrid(
    image: Image.Image,
    sam3_model,
    sam3_processor,
    clip_model,
    clip_processor,
    detection_prompt: str,
    candidates: List[str],
    prompt_template: str = "a photo of {} logo",
    detection_threshold: float = 0.3,
    device: str = "cuda",
) -> Dict:
    """
    Hybrid detection + classification pipeline.

    1. SAM3 detects regions (WHERE)
    2. Crop each region
    3. CLIP classifies each crop (WHAT)

    Args:
        image: Input PIL image
        sam3_model: SAM3 model for detection
        sam3_processor: SAM3 processor
        clip_model: CLIP model for classification
        clip_processor: CLIP processor
        detection_prompt: What to detect (e.g., "logo", "shoe")
        candidates: List of candidate labels
        prompt_template: Template for CLIP (use {} for label)
        detection_threshold: SAM3 confidence threshold
        device: Device to run on

    Returns:
        Dictionary with boxes, masks, and classifications
    """

    print(f"\n[Step 1] SAM3: Detecting '{detection_prompt}'...")
    detections = detect_regions(
        image, sam3_model, sam3_processor,
        detection_prompt, detection_threshold, device
    )

    print(f"   Found {detections['num_detections']} detections")

    if detections['num_detections'] == 0:
        return {
            "boxes": [],
            "masks": [],
            "detection_scores": [],
            "classifications": [],
            "num_detections": 0,
        }

    print(f"\n[Step 2] Cropping {detections['num_detections']} regions...")
    crops = crop_regions(image, detections['boxes'])

    print(f"\n[Step 3] CLIP: Classifying against {len(candidates)} candidates...")
    classifications = classify_with_clip(
        crops, candidates, clip_model, clip_processor,
        prompt_template, device
    )

    return {
        "boxes": detections['boxes'],
        "masks": detections['masks'],
        "detection_scores": detections['scores'],
        "classifications": classifications,
        "num_detections": detections['num_detections'],
        "crops": crops,  # Include crops for visualization
    }


def visualize_hybrid(
    image: Image.Image,
    results: Dict,
    output_path: str = "hybrid_result.jpg",
    show_crops: bool = True,
):
    """Visualize detection + classification results."""
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

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"{clf['label']}: {clf['score']:.1%}"
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)

    img.save(output_path)
    print(f"\nSaved: {output_path}")

    # Save individual crops
    if show_crops and "crops" in results:
        for i, (crop, clf) in enumerate(zip(results["crops"], results["classifications"])):
            crop_path = output_path.replace(".jpg", f"_crop{i}_{clf['label']}.jpg")
            crop.save(crop_path)
            print(f"Saved crop: {crop_path}")

    return img


# =============================================================================
# Convenience class
# =============================================================================

class HybridClassifier:
    """
    SAM3 + CLIP hybrid classifier.

    Usage:
        classifier = HybridClassifier(device="cpu")
        classifier.set_candidates(["Nike", "Adidas", "Puma"])

        results = classifier.classify(image, detection_prompt="logo")
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ):
        self.device = device

        # Load models
        self.sam3_model, self.sam3_processor = load_sam3(device)
        self.clip_model, self.clip_processor = load_clip(device, clip_model_name)

        self.candidates = []
        self.prompt_template = "a photo of {} logo"

    def set_candidates(self, candidates: List[str], prompt_template: str = None):
        """Set candidate labels for classification."""
        self.candidates = candidates
        if prompt_template:
            self.prompt_template = prompt_template

    def classify(
        self,
        image: Image.Image,
        detection_prompt: str = "logo",
        detection_threshold: float = 0.3,
    ) -> Dict:
        """Run detection + classification."""

        if not self.candidates:
            raise ValueError("No candidates set. Call set_candidates() first.")

        return detect_and_classify_hybrid(
            image=image,
            sam3_model=self.sam3_model,
            sam3_processor=self.sam3_processor,
            clip_model=self.clip_model,
            clip_processor=self.clip_processor,
            detection_prompt=detection_prompt,
            candidates=self.candidates,
            prompt_template=self.prompt_template,
            detection_threshold=detection_threshold,
            device=self.device,
        )

    def visualize(self, image: Image.Image, results: Dict, output_path: str = "result.jpg"):
        """Visualize results."""
        return visualize_hybrid(image, results, output_path)


# =============================================================================
# Main example
# =============================================================================

if __name__ == "__main__":
    import os
    from io import BytesIO

    device = "cpu"
    os.makedirs("test_images", exist_ok=True)

    print("="*60)
    print("SAM3 + CLIP Hybrid Classification")
    print("="*60)

    # Load models
    sam3_model, sam3_processor = load_sam3(device)
    clip_model, clip_processor = load_clip(device)

    # Test image
    print("\nFetching test image...")
    url = "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=640"
    response = requests.get(url, timeout=15)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Image size: {image.size}")

    # Candidates
    candidates = ["Louis Vuitton Logo", "Gucci Logo", "Chanel Logo", "Prada Logo", "Burberry Logo", "Hermes Logo", "Coach Logo"]
    print(f"\nCandidates: {candidates}")

    # Run hybrid classification
    results = detect_and_classify_hybrid(
        image=image,
        sam3_model=sam3_model,
        sam3_processor=sam3_processor,
        clip_model=clip_model,
        clip_processor=clip_processor,
        detection_prompt="Logo",
        candidates=candidates,
        prompt_template="a photo of a {} brand logo",
        detection_threshold=0.3,
        device=device,
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Detections: {results['num_detections']}")

    for i, clf in enumerate(results["classifications"]):
        print(f"\n  Detection {i+1}: {clf['label']}")
        print(f"    Confidence: {clf['score']:.1%}")

        # Show top 3
        sorted_scores = sorted(clf['all_scores'].items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 3:")
        for label, score in sorted_scores[:3]:
            print(f"      {label}: {score:.1%}")

    # Visualize
    if results['num_detections'] > 0:
        visualize_hybrid(image, results, "test_images/hybrid_result.jpg")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
