# Mini-CIR: CIRCO validation benchmark

This is a lightweight, reproducible implementation of the experiment plan in `PROJECT_PLAN_PROMPT.md`. It uses OpenCLIP ViT-B/32 and the official CIRCO validation annotations. It builds a deterministic COCO gallery that includes every validation reference and ground-truth image, then fills the remaining slots with seeded distractors.

## Important interpretation

The plan is guidance, not a source of measured results. The illustrative numbers in its result table are not reproduced measurements and are never written as results by this project. A full COCO download and a GPU are needed to produce actual metrics. The author-proxy cache mentioned in the plan is not included in this workspace; supply edited images yourself in `--proxy-dir` to run proxy-based methods. The implemented reranker is a CLIP cross-modal reranker, not an MLLM. No paid image-generation API is called automatically.

The original `Imagine-and-Seek/` checkout describes its official proxy setup and expects large external weights/data. This implementation is isolated from it and does not modify upstream code.

## Data layout

Download CIRCO validation annotations and COCO 2017 Unlabeled images plus `image_info_unlabeled2017.json`, then arrange:

```text
CIRCO/
  annotations/val.json
  COCO2017_unlabeled/
    annotations/image_info_unlabeled2017.json
    unlabeled2017/000000243611.jpg ...
```

Each query annotation must contain `id`, `reference_img_id`, `relative_caption`, and `gt_img_ids`. Images are resolved by their 12-digit COCO ID. The official validation file may contain a different number of queries than the plan's approximate 220; the script reports the actual count.

## Run

```bash
python -m pip install -r Mini-CIR/requirements.txt
python Mini-CIR/cir.py --data /path/to/CIRCO --gallery-size 40000 --output Mini-CIR/output
```

Proxy images, when available, must be JPEG files named `<query-id>.jpg`:

```bash
python Mini-CIR/cir.py --data /path/to/CIRCO --proxy-dir /path/to/proxies --output Mini-CIR/output
```

The first run caches normalized gallery embeddings. `--rebuild-cache` forces re-encoding. Reduce `--batch-size` if GPU memory is limited. The result JSON reports mAP@5/10/25, Recall@5/10, and reference-image Top-1 rate for every method that could run. Proxy methods are omitted unless every query has a proxy image, preventing mixed or misleading comparisons.

## Implemented methods

* `baseline_no_proxy`: normalized CLIP feature for `0.5 * reference + 0.5 * relative text`.
* `fixed_proxy`: fixed three-way normalized feature fusion when all proxies are present.
* `dynamic_fusion`: adjusts the proxy weight using proxy/reference cosine similarity.
* `dynamic_rerank`: reorders the top 50 with CLIP image-to-proxy and image-to-text scores; penalizes the source reference when it is the leading candidate.

These are transparent lightweight baselines inspired by the plan. They are not an exact reproduction of the paper's full LDRE/IP-CIR pipeline or its reported configuration. In particular, results on a 40k sampled gallery are not directly comparable with official full-gallery benchmark scores.

## Notebooks

`01_baseline.ipynb` and `02_improved.ipynb` provide Kaggle/Colab run guides and evaluation entry points. They expect the dataset to be mounted or uploaded by the user; the notebooks do not trigger multi-GB downloads.
