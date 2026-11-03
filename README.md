# 🌆 UrbanMoE: A Sparse Multi-Modal Mixture-of-Experts Framework for Multi-Task Urban Region Profiling
 A multimodal multi-task framework for urban region representation and regression, integrating images, text, and POI data with a Mixture-of-Experts architecture.


> **Note:** the region embedding is a per-tile lookup with ids assigned separately in train/val/test, so val and test tiles get embeddings trained for other tiles (or never trained), and in training it can memorize tiles. TODO: test `--no-use_region_embedding` (added, not yet tested) and check performance without it.

> **Note:** in this code the gate weights are applied to expert outputs, and every eligible expert always runs on the full input, even when its weight is zero. The paper claims sparse routing reduces computation by activating only selected experts (Sections 2.4, 2.4.2, 2.6), but that does not happen here, so it is unclear where such a benefit would come from.

> **Note:** the expert layout (shared and task-specific experts, with each task's gate over its own plus the shared ones) more closely resembles PLE (Tang et al., RecSys 2020, one-level CGC block), which the paper does not cite, than the MMoE it cites. The pair-shared experts and the gate threshold are not in PLE.

## ⚙️ Environment

`srai` is installed from a fork (https://github.com/zachkaupp/srai-fork). `uv.lock` pins a specific commit of it, so plain `uv sync` won't pick up new commits. To pull the latest commit from the fork:

```
uv sync --upgrade-package srai
```


## 📊 Dataset

The dataset used in this project is publicly available:

👉 **Download link:** *https://drive.google.com/drive/folders/1ITILOIvuD-PVa6F9W8DeGhsQ_3yEr36-?usp=sharing*

- Also need to download `RemoteCLIP-ViT-L-14.pt` from https://huggingface.co/chendelong/RemoteCLIP/tree/main, and place in repo root.

### The dataset includes:
- 🛰️ Satellite images  
- 📝 Textual region descriptions   
- 📈 Ground-truth labels (e.g., population, carbon emissions, light)

### Note: model selection depends on the random split
Best-checkpoint selection uses validation carbon error. The NewYork carbon labels are heavily right-skewed, with a handful of extreme values (one far above the rest), so validation carbon error depends heavily on whether those tiles land in the validation split, and checkpoint choice follows.

The highest-carbon NewYork tiles line up with large point sources, which the tile's POIs and text do not reveal (tile locations come from the GeoTIFF georeference, UTM zone 18N):
- `tile_436`: E. F. Barrett Power Station (Island Park, NY)
- `tile_1682`: Port Jefferson Power Station (Port Jefferson, NY)
- `tile_105`: JFK Airport (its description also says airport)

This fits how the paper's carbon source (ODIAC) places emissions, using power plant locations and nighttime lights.

## 🗺️ OSM embeddings

`scripts/compute_embeddings.py` writes OSM-based tile embeddings (SRAI) to `data/embeddings/`. `main.py` uses them as extra branches via `--modalities` (e.g. `--modalities text alphaearth s2vec`).

```
uv run python scripts/compute_embeddings.py --dataset NewYork --embedder s2vec --train_region northeast
```

- **S2Vec grid:** S2Vec uses modified grid cells: the tiles' own projected grid (`srai-fork`) in place of S2 cells, so each tile is one S2Vec cell.
- **S2Vec training area:** with `--train_region northeast`, S2Vec trains on the whole Northeast, then embeds the tiles.
- **Computation changes:** OSM features are counted a few images at a time and freed (`scripts/s2vec_counts.py`), since S2Vec trains only on the per-cell counts and the features were what ran out of memory. The counting is identical to doing it all at once; the loader can return slightly different features (about 0.1-0.3%) for a different query area, so counts differ in under 0.1% of entries. The model is fitted on the training area and then used only to embed the tiles.

- **AlphaEarth:** `scripts/compute_alphaearth.py` writes `data/embeddings/<dataset>_alphaearth.csv` (64 dims, unit length). It downloads the 80 m overviews of the public AlphaEarth Foundations COGs (latest year, no Earth Engine account) and averages the pixels inside each tile. The 10 m pixels are not read directly because the files are laid out so that each tile window downloads about 25x more data than it uses; the overview average is within 0.9998 cosine similarity of the 10 m average on the tiles checked.

- **Val/test tiles in S2Vec training:** S2Vec and the feature vocabulary are fit on unlabeled OSM data that includes the val and test tiles. No labels are used.

---

 ## 📄 Paper

This work has been **accepted to The Web Conference 2026 (WWW2026)** 🎉

**Title:** *UrbanMoE: A Sparse Multi-Modal Mixture-of-Experts Framework for Multi-Task Urban Region Profiling*  
**Conference:** WWW2026 (The Web Conference 2026)  
**Paper Link:** *https://arxiv.org/pdf/2601.22746*  
