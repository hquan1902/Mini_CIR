"""Practical CIRCO validation benchmark: CLIP baselines and lightweight fusion."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CocoImages(Dataset):
    def __init__(self, ids: list[int], image_dir: Path, preprocess):
        self.ids, self.image_dir, self.preprocess = ids, image_dir, preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        path = self.image_dir / f"{image_id:012d}.jpg"
        try:
            with Image.open(path) as im:
                return self.preprocess(im.convert("RGB")), image_id
        except Exception:
            return torch.zeros(3, 224, 224), image_id


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def prepare_gallery(annotations: list[dict], coco_info: dict, image_dir: Path,
                    size: int, seed: int) -> list[int]:
    """Keep every validation target, then sample distractors deterministically."""
    info = coco_info.get("images", coco_info)
    available = {int(row["id"]) for row in info if (image_dir / f"{int(row['id']):012d}.jpg").is_file()}
    required = set()
    for q in annotations:
        required.update(int(x) for x in q.get("gt_img_ids", []))
        if "reference_img_id" in q:
            required.add(int(q["reference_img_id"]))
    missing = required - available
    if missing:
        raise FileNotFoundError(f"{len(missing)} required reference/GT images missing; download COCO unlabeled images first")
    if len(required) > size:
        raise ValueError(f"Required images ({len(required)}) exceed gallery size ({size})")
    pool = sorted(available - required)
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(size - len(required), len(pool)))
    return sorted(required | set(chosen))


def encode_gallery(model, preprocess, ids, image_dir, device, batch_size):
    loader = DataLoader(CocoImages(ids, image_dir, preprocess), batch_size=batch_size,
                        num_workers=2, pin_memory=device.startswith("cuda"))
    feats, good_ids = [], []
    model.eval()
    with torch.inference_mode():
        for pixels, batch_ids in loader:
            pixels = pixels.to(device)
            emb = model.encode_image(pixels).float()
            emb = F.normalize(emb, dim=-1)
            valid = pixels.abs().sum(dim=(1, 2, 3)) > 0
            feats.append(emb[valid].cpu())
            good_ids.extend(int(x) for x, ok in zip(batch_ids, valid.tolist()) if ok)
    return torch.cat(feats), good_ids


def encode_text(model, tokenizer, text, device):
    with torch.inference_mode():
        return F.normalize(model.encode_text(tokenizer([text]).to(device)).float(), dim=-1)[0]


def ap_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, image_id in enumerate(ranked_ids[:k], 1):
        if image_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def evaluate_rankings(rows: list[dict], rankings: list[list[int]], ks=(5, 10, 25)):
    out = {}
    for k in ks:
        out[f"mAP@{k}"] = float(np.mean([ap_at_k(r, set(map(int, q.get("gt_img_ids", []))), k)
                                          for q, r in zip(rows, rankings)]))
    for k in (5, 10):
        out[f"Recall@{k}"] = float(np.mean([
            len(set(r[:k]) & set(map(int, q.get("gt_img_ids", [])))) /
            max(1, len(set(map(int, q.get("gt_img_ids", []))))) for q, r in zip(rows, rankings)]))
    refs = [int(q["reference_img_id"]) for q in rows if "reference_img_id" in q]
    out["reference_top1_rate"] = float(np.mean([r[0] == ref for q, r in zip(rows, rankings)
                                                 for ref in [int(q.get("reference_img_id", -1))]])) if refs else None
    return out


def rank_methods(args):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Install requirements: pip install -r Mini-CIR/requirements.txt") from exc
    data = Path(args.data)
    ann_path = data / "annotations" / "val.json"
    coco_info_path = data / "COCO2017_unlabeled" / "annotations" / "image_info_unlabeled2017.json"
    image_dir = data / "COCO2017_unlabeled" / "unlabeled2017"
    rows = load_json(ann_path)
    if not isinstance(rows, list):
        raise ValueError("CIRCO val annotations must be a JSON list")
    info = load_json(coco_info_path)
    ids = prepare_gallery(rows, info, image_dir, args.gallery_size, args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    cache = Path(args.output) / f"gallery_{len(ids)}_{args.model.replace('/', '-')}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not args.rebuild_cache:
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("ids") == ids:
            gallery = saved["features"]
        else:
            gallery, ids = encode_gallery(model, preprocess, ids, image_dir, device, args.batch_size)
    else:
        gallery, ids = encode_gallery(model, preprocess, ids, image_dir, device, args.batch_size)
    torch.save({"ids": ids, "features": gallery}, cache)
    index = {image_id: i for i, image_id in enumerate(ids)}
    ref_vectors, text_vectors, proxy_vectors = [], [], []
    proxy_dir = Path(args.proxy_dir) if args.proxy_dir else None
    start = time.perf_counter()
    for q in rows:
        rid = int(q["reference_img_id"])
        ref_vectors.append(gallery[index[rid]])
        text_vectors.append(encode_text(model, tokenizer, q["relative_caption"], device).cpu())
        proxy = proxy_dir / f"{q['id']}.jpg" if proxy_dir else None
        if proxy and proxy.is_file():
            with Image.open(proxy) as im, torch.inference_mode():
                px = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
                pv = F.normalize(model.encode_image(px).float(), dim=-1)[0].cpu()
        else:
            pv = None
        proxy_vectors.append(pv)
    refs = F.normalize(torch.stack(ref_vectors), dim=-1)
    texts = F.normalize(torch.stack(text_vectors), dim=-1)
    proxy_present = all(x is not None for x in proxy_vectors)
    proxies = F.normalize(torch.stack(proxy_vectors), dim=-1) if proxy_present else None
    gallery_dev = gallery.to(device)
    # Baseline follows the plan's 0.5/0.5 query. Fixed proxy fusion uses normalized components.
    baseline_q = F.normalize(0.5 * refs + 0.5 * texts, dim=-1)
    baseline_scores = baseline_q @ gallery_dev.T
    methods = {"baseline_no_proxy": baseline_scores}
    if proxies is not None:
        proxy_dev = proxies.to(device)
        fixed = F.normalize(0.2 * refs + 0.3 * texts + 0.5 * proxy_dev, dim=-1)
        methods["fixed_proxy"] = fixed @ gallery_dev.T
        # Adaptive proxy contribution: close proxy/reference => preserve identity; otherwise trust edit/text more.
        sim = (refs * proxy_dev).sum(-1).clamp(-1, 1)
        lam_proxy = (0.45 + 0.35 * ((sim + 1) / 2)).clamp(0.45, 0.80)
        lam_text = (1 - lam_proxy) * 0.55
        lam_ref = 1 - lam_proxy - lam_text
        dynamic = F.normalize(lam_ref[:, None] * refs + lam_text[:, None] * texts + lam_proxy[:, None] * proxy_dev, dim=-1)
        methods["dynamic_fusion"] = dynamic @ gallery_dev.T
        # Cheap cross-modal reranker over top-50: score candidate-image agreement with proxy and text.
        candidates = torch.topk(methods["dynamic_fusion"], k=min(50, len(ids)), dim=-1).indices
        reranked = methods["dynamic_fusion"].clone()
        for qi in range(len(rows)):
            cand_feat = gallery_dev[candidates[qi]]
            cross = 0.65 * (cand_feat @ proxy_dev[qi]) + 0.35 * (cand_feat @ texts[qi].to(device))
            # Blend normalized scores to retain the first-stage signal; remove source image if it is an unedited match.
            reranked[qi, candidates[qi]] = 0.55 * methods["dynamic_fusion"][qi, candidates[qi]] + 0.45 * cross
            if ids[int(candidates[qi, 0])] == int(rows[qi]["reference_img_id"]):
                reranked[qi, int(rows[qi]["reference_img_id"])] -= 0.15
        methods["dynamic_rerank"] = reranked
    outputs = {}
    for name, scores in methods.items():
        ordered = torch.argsort(scores, dim=-1, descending=True).cpu()
        rankings = [[ids[j] for j in row[:50].tolist()] for row in ordered]
        outputs[name] = {"metrics": evaluate_rankings(rows, rankings),
                         "rankings": {str(q["id"]): r for q, r in zip(rows, rankings)}}
    result = {"config": {"model": args.model, "pretrained": args.pretrained, "gallery_size": len(ids),
                          "query_count": len(rows), "proxy_dir": str(proxy_dir) if proxy_dir else None,
                          "elapsed_seconds": round(time.perf_counter() - start, 3), "seed": args.seed},
              "methods": outputs}
    out = Path(args.output)
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v["metrics"] for k, v in outputs.items()}, indent=2))
    print(f"Saved results to {out / 'results.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="CIRCO root containing annotations/val.json and COCO2017_unlabeled/")
    p.add_argument("--output", default="Mini-CIR/output")
    p.add_argument("--proxy-dir", help="Optional folder with one edited proxy per query, named <query-id>.jpg")
    p.add_argument("--gallery-size", type=int, default=40000)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--rebuild-cache", action="store_true")
    rank_methods(p.parse_args())


if __name__ == "__main__":
    main()
