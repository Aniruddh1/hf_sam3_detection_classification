#!/usr/bin/env python3
"""Test all classification methods: Flat, Cached, Embedding, Hierarchical."""

import time
import torch
from PIL import Image
import requests
from classifier import (
    load_model,
    detect_and_classify,
    detect_and_classify_cached,
    detect_and_classify_embedding,
    hierarchical_classify,
    visualize,
    EmbeddingCache,
    DEFAULT_BRAND_TAXONOMY,
)

device = "cpu"
print(f"Using device: {device}")

# Load model
model, processor = load_model(device)

# Test image
print("\n" + "="*60)
print("Scalability Test: All Classification Methods")
print("="*60)

image_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=640"
print(f"\nFetching image: {image_url}")

try:
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    from io import BytesIO
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Image size: {image.size}")

    # Gather all brands
    all_brands = []
    for items in DEFAULT_BRAND_TAXONOMY.values():
        all_brands.extend(items)

    print(f"\nTotal brands: {len(all_brands)} across {len(DEFAULT_BRAND_TAXONOMY)} categories")

    # =========================================================================
    # Method 1: Flat (baseline - O(N) forward passes)
    # =========================================================================
    print("\n" + "="*60)
    print("Method 1: FLAT (O(N) forward passes)")
    print("="*60)

    start = time.time()
    flat_results = detect_and_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="logo",
        candidates=all_brands,
        prompt_template="{label} logo",
        detection_threshold=0.3,
        device=device,
    )
    flat_time = time.time() - start

    print(f"\nFlat Results: {flat_results['num_detections']} detections")
    for clf in flat_results["classifications"]:
        print(f"  -> {clf['label']} (score: {clf['score']:.3f})")
    print(f"Time: {flat_time:.2f}s")

    # =========================================================================
    # Method 2: Cached Text Inputs (still O(N) forward passes)
    # =========================================================================
    print("\n" + "="*60)
    print("Method 2: CACHED TEXT INPUTS (O(N) passes, no re-tokenization)")
    print("="*60)

    cache = EmbeddingCache(model, processor, device)

    cache_start = time.time()
    cache.precompute_candidates(all_brands, "{label} logo")
    cache_setup = time.time() - cache_start

    start = time.time()
    cached_results = detect_and_classify_cached(
        image=image,
        model=model,
        processor=processor,
        cache=cache,
        detection_prompt="logo",
        candidates=all_brands,
        prompt_template="{label} logo",
        detection_threshold=0.3,
        device=device,
    )
    cached_time = time.time() - start

    print(f"\nCached Results: {cached_results['num_detections']} detections")
    for clf in cached_results["classifications"]:
        print(f"  -> {clf['label']} (score: {clf['score']:.3f})")
    print(f"Time: {cached_time:.2f}s (setup: {cache_setup:.2f}s)")

    # =========================================================================
    # Method 3: Cached EMBEDDINGS (O(1) forward pass + matrix multiply)
    # =========================================================================
    print("\n" + "="*60)
    print("Method 3: CACHED EMBEDDINGS (O(1) pass + dot product)")
    print("="*60)

    embed_cache = EmbeddingCache(model, processor, device)

    embed_start = time.time()
    embed_cache.precompute_embeddings(all_brands, "{label} logo")
    embed_setup = time.time() - embed_start

    start = time.time()
    embed_results = detect_and_classify_embedding(
        image=image,
        model=model,
        processor=processor,
        cache=embed_cache,
        detection_prompt="logo",
        detection_threshold=0.3,
        device=device,
    )
    embed_time = time.time() - start

    print(f"\nEmbedding Results: {embed_results['num_detections']} detections")
    for clf in embed_results["classifications"]:
        print(f"  -> {clf['label']} (score: {clf['score']:.3f})")
    print(f"Time: {embed_time:.2f}s (setup: {embed_setup:.2f}s)")

    # =========================================================================
    # Method 4: Hierarchical (2-stage for accuracy)
    # =========================================================================
    print("\n" + "="*60)
    print("Method 4: HIERARCHICAL (2-stage for accuracy)")
    print("="*60)

    start = time.time()
    hier_results = hierarchical_classify(
        image=image,
        model=model,
        processor=processor,
        detection_prompt="logo",
        taxonomy=DEFAULT_BRAND_TAXONOMY,
        category_template="{label} brand logo",
        item_template="{label} logo",
        detection_threshold=0.3,
        device=device,
        top_k_categories=2,
    )
    hier_time = time.time() - start

    print(f"\nHierarchical Results: {hier_results['num_detections']} detections")
    for clf in hier_results["classifications"]:
        print(f"  -> {clf['label']} ({clf.get('category', '?')}) (score: {clf['score']:.3f})")
    print(f"Time: {hier_time:.2f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY - Time Comparison")
    print("="*60)
    print(f"\nWith {len(all_brands)} candidates:")
    print(f"")
    print(f"  Method              | Single Image | Batch (N images)")
    print(f"  --------------------|--------------|------------------")
    print(f"  Flat                | {flat_time:>10.2f}s | {flat_time:.2f}N")
    print(f"  Cached Inputs       | {cached_time:>10.2f}s | {cache_setup:.2f} + {cached_time:.2f}N")
    print(f"  Cached Embeddings   | {embed_time:>10.2f}s | {embed_setup:.2f} + {embed_time:.2f}N  <-- FASTEST")
    print(f"  Hierarchical        | {hier_time:>10.2f}s | {hier_time:.2f}N  <-- MOST ACCURATE")

    print(f"\n  Speedup (Embedding vs Flat): {flat_time/embed_time:.1f}x faster")

    if hier_results['num_detections'] > 0:
        visualize(image, hier_results, "result.jpg")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("Done!")
print("="*60)
