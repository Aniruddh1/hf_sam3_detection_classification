#!/usr/bin/env python3
"""Test logo detection and brand classification on multiple images."""

import os
import torch
from PIL import Image
import requests
from io import BytesIO
from classifier import load_model, detect_and_classify, visualize, DEFAULT_BRAND_TAXONOMY

device = "cpu"
print(f"Using device: {device}")

# Create output directory
os.makedirs("test_images", exist_ok=True)

# Load model
model, processor = load_model(device)

# =============================================================================
# Test Images - Various Brand Categories
# =============================================================================

TEST_IMAGES = [
    # Sports brands
    {
        "name": "nike_shoe",
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640",
        "description": "Red Nike shoe",
        "detection_prompt": "shoe",
        "candidates": ["Nike", "Adidas", "Puma", "Reebok", "New Balance", "Under Armour"],
        "prompt_template": "{label} shoe",
    },
    {
        "name": "adidas_shoe",
        "url": "https://images.unsplash.com/photo-1518002171953-a080ee817e1f?w=640",
        "description": "Adidas sneaker",
        "detection_prompt": "shoe",
        "candidates": ["Nike", "Adidas", "Puma", "Reebok", "New Balance", "Converse"],
        "prompt_template": "{label} shoe",
    },
    # Tech brands
    {
        "name": "apple_macbook",
        "url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=640",
        "description": "MacBook laptop",
        "detection_prompt": "laptop",
        "candidates": ["Apple", "Dell", "HP", "Lenovo", "Microsoft", "Asus"],
        "prompt_template": "{label} laptop",
    },
    {
        "name": "iphone",
        "url": "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=640",
        "description": "iPhone",
        "detection_prompt": "phone",
        "candidates": ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Huawei"],
        "prompt_template": "{label} phone",
    },
    # Automotive
    {
        "name": "car_front",
        "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=640",
        "description": "Car front view",
        "detection_prompt": "car",
        "candidates": ["BMW", "Mercedes", "Audi", "Tesla", "Porsche", "Toyota"],
        "prompt_template": "{label} car",
    },
    # Food & Beverage
    {
        "name": "coffee_cup",
        "url": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=640",
        "description": "Coffee cup",
        "detection_prompt": "coffee cup",
        "candidates": ["Starbucks", "Dunkin", "Costa", "Tim Hortons", "Peet's"],
        "prompt_template": "{label} coffee",
    },
    # Fashion
    {
        "name": "handbag",
        "url": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=640",
        "description": "Luxury handbag",
        "detection_prompt": "bag",
        "candidates": ["Louis Vuitton", "Gucci", "Chanel", "Prada", "Hermes", "Coach"],
        "prompt_template": "{label} bag",
    },
]


def fetch_image(url: str, timeout: int = 15) -> Image.Image:
    """Fetch image from URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def test_single_image(image_config: dict, model, processor, device: str):
    """Test classification on a single image."""
    name = image_config["name"]
    print(f"\n{'='*60}")
    print(f"Test: {name} - {image_config['description']}")
    print(f"{'='*60}")

    try:
        # Fetch image
        print(f"Fetching: {image_config['url']}")
        image = fetch_image(image_config["url"])
        print(f"Image size: {image.size}")

        # Save original
        original_path = f"test_images/{name}_original.jpg"
        image.save(original_path)
        print(f"Saved original: {original_path}")

        # Classify
        print(f"\nDetecting '{image_config['detection_prompt']}'...")
        print(f"Candidates: {image_config['candidates']}")

        results = detect_and_classify(
            image=image,
            model=model,
            processor=processor,
            detection_prompt=image_config["detection_prompt"],
            candidates=image_config["candidates"],
            prompt_template=image_config["prompt_template"],
            detection_threshold=0.3,
            device=device,
        )

        print(f"\nResults: {results['num_detections']} detections")

        for i, clf in enumerate(results["classifications"]):
            print(f"\n  Detection {i+1}: {clf['label']} (score: {clf['score']:.3f})")
            sorted_scores = sorted(clf['all_scores'].items(), key=lambda x: x[1], reverse=True)
            print(f"    Top 3:")
            for brand, score in sorted_scores[:3]:
                print(f"      {brand}: {score:.3f}")

        # Save visualization
        if results['num_detections'] > 0:
            output_path = f"test_images/{name}_result.jpg"
            visualize(image, results, output_path)

        return {"name": name, "success": True, "results": results}

    except Exception as e:
        print(f"Error: {e}")
        return {"name": name, "success": False, "error": str(e)}


def test_with_taxonomy(image: Image.Image, model, processor, device: str):
    """Test using the full brand taxonomy."""
    print(f"\n{'='*60}")
    print("Test: Full Taxonomy Classification (48 brands)")
    print(f"{'='*60}")

    # Flatten taxonomy
    all_brands = []
    for items in DEFAULT_BRAND_TAXONOMY.values():
        all_brands.extend(items)

    print(f"Total candidates: {len(all_brands)}")
    print(f"Categories: {list(DEFAULT_BRAND_TAXONOMY.keys())}")

    results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="logo",
        candidates=all_brands,
        prompt_template="{label} logo",
        detection_threshold=0.3,
        device=device,
    )

    print(f"\nResults: {results['num_detections']} logos found")

    for i, clf in enumerate(results["classifications"]):
        # Find category
        category = "unknown"
        for cat, items in DEFAULT_BRAND_TAXONOMY.items():
            if clf['label'] in items:
                category = cat
                break

        print(f"\n  Logo {i+1}: {clf['label']} ({category})")
        print(f"    Score: {clf['score']:.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Logo Detection & Brand Classification Test Suite")
    print("="*60)

    # Run all tests
    results_summary = []

    for config in TEST_IMAGES:
        result = test_single_image(config, model, processor, device)
        results_summary.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results_summary if r["success"]]
    failed = [r for r in results_summary if not r["success"]]

    print(f"\nTotal tests: {len(results_summary)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSuccessful tests:")
        for r in successful:
            num_det = r["results"]["num_detections"]
            if num_det > 0:
                top_label = r["results"]["classifications"][0]["label"]
                top_score = r["results"]["classifications"][0]["score"]
                print(f"  - {r['name']}: {top_label} ({top_score:.3f})")
            else:
                print(f"  - {r['name']}: No detections")

    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")

    print("\n" + "="*60)
    print(f"Output images saved to: test_images/")
    print("="*60)
