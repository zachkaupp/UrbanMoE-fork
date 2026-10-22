# 🌆 UrbanMoE: A Sparse Multi-Modal Mixture-of-Experts Framework for Multi-Task Urban Region Profiling
 A multimodal multi-task framework for urban region representation and regression, integrating images, text, and POI data with a Mixture-of-Experts architecture.


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

---

 ## 📄 Paper

This work has been **accepted to The Web Conference 2026 (WWW2026)** 🎉

**Title:** *UrbanMoE: A Sparse Multi-Modal Mixture-of-Experts Framework for Multi-Task Urban Region Profiling*  
**Conference:** WWW2026 (The Web Conference 2026)  
**Paper Link:** *https://arxiv.org/pdf/2601.22746*  
