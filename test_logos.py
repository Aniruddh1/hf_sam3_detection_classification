#!/usr/bin/env python3
"""Test logo detection and brand classification."""

import torch
from PIL import Image
import requests
from classifier import load_model, detect_and_classify, visualize

device = "cpu"
print(f"Using device: {device}")

# Load model
model, processor = load_model(device)

# Test image with Nike logo (sports shoe image)
print("\n" + "="*60)
print("Test: Logo Detection")
print("="*60)

# Using an image that likely has a sports brand logo
# Let's try a few different images

# Image 1: Sports/athletic image
image_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640"  # Red Nike shoe
print(f"\nFetching image: {image_url}")

try:
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    from io import BytesIO
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Image size: {image.size}")

    # Define brand candidates
    brand_candidates = [
        "Nike", "Adidas", "Puma", "Reebok", "New Balance",
        "Under Armour", "Converse", "Vans", "Jordan", "Asics"
    ]

    print(f"\nCandidates: {brand_candidates}")

    # Detect and classify
    results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="logo",
        candidates=brand_candidates,
        prompt_template="{label} logo",
        detection_threshold=0.3,
        device=device,
    )

    print(f"\nResults:")
    print(f"  Found {results['num_detections']} logos")

    for i, clf in enumerate(results["classifications"]):
        print(f"\n  Logo {i+1}: {clf['label']} (score: {clf['score']:.3f})")
        print(f"    All scores:")
        sorted_scores = sorted(clf['all_scores'].items(), key=lambda x: x[1], reverse=True)
        for brand, score in sorted_scores[:5]:
            print(f"      {brand}: {score:.3f}")

    if results['num_detections'] > 0:
        visualize(image, results, "logo_classification.jpg")

except Exception as e:
    print(f"Error with image 1: {e}")

# Try detecting "shoe" or "sneaker" if logo detection doesn't work well
print("\n" + "="*60)
print("Test: Shoe Brand Classification")
print("="*60)

try:
    # Reuse the same image
    results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="shoe",
        candidates=brand_candidates,
        prompt_template="{label} shoe",
        detection_threshold=0.3,
        device=device,
    )

    print(f"\nResults:")
    print(f"  Found {results['num_detections']} shoes")

    for i, clf in enumerate(results["classifications"]):
        print(f"\n  Shoe {i+1}: {clf['label']} (score: {clf['score']:.3f})")
        sorted_scores = sorted(clf['all_scores'].items(), key=lambda x: x[1], reverse=True)
        for brand, score in sorted_scores[:5]:
            print(f"      {brand}: {score:.3f}")

    if results['num_detections'] > 0:
        visualize(image, results, "shoe_brand_classification.jpg")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("Done!")
print("="*60)
