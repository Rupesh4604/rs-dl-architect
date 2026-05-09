# rs-dl-architect

**Expert guidance for designing, analyzing, and reviewing deep learning architectures in remote sensing and satellite imagery.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Skill](https://img.shields.io/badge/Claude-Skill-orange.svg)](https://github.com/anthropics/skills)
[![Documentation](https://img.shields.io/badge/docs-live-green.svg)](https://your-username.github.io/rs-dl-architect)

A comprehensive Claude AI skill covering segmentation, change detection, multi-modal fusion, temporal analysis, foundation models (SatMAE, Prithvi, GeoSAM), object detection, and uncertainty quantification. Supports both **PyTorch** and **TensorFlow** with production-ready code templates.

---

## 🎯 What This Skill Does

- **Designs** architectures tailored to your remote sensing task (task-shape diagnosis first, not reflexive model recommendations)
- **Analyzes** existing models, papers, and checkpoints (explains what's actually happening, not just listing layers)
- **Reviews** your code/architecture with a 10-section rigorous checklist (blocking issues + concrete fixes, not vague suggestions)

**Key differentiator:** Not generic computer vision adapted to RS — built from the ground up for satellite/aerial imagery physics, GSD considerations, geographic generalization, and multi-spectral/multi-temporal/multi-modal challenges.

---

## 🚀 Quick Start

### Installation

**Option 1: Via Claude Code Plugin** _(Recommended)_

```bash
claude plugin install rs-dl-architect@Rupesh4604
```

**Option 2: Manual Installation**

```bash
# Clone the repository
git clone https://github.com/Rupesh4604/rs-dl-architect.git

# Copy to Claude skills directory
cp -r rs-dl-architect/rs-dl-architect ~/.claude/skills/

# For project-specific installation
cp -r rs-dl-architect/rs-dl-architect ./.claude/skills/
```

**Option 3: Direct Download**
Download the [latest .skill file](https://github.com/Rupesh4604/rs-dl-architect/releases) and upload via Claude.ai or Claude Code.

### Usage

Just ask Claude naturally — the skill triggers automatically on relevant queries:

```
"I need to detect crop field boundaries from 10m Sentinel-2 with 500 labels"

"How should I fuse SAR and optical for flood mapping?"

"Review my U-Net — using ImageNet weights with 13 Sentinel-2 bands"

"Should I use SatMAE or DINOv2 for EuroSAT land cover classification?"
```

---

## 📖 Coverage

### Architectures & Tasks

| Domain                    | Coverage                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| **Segmentation**          | U-Net family, SegFormer, UNetFormer, Mask2Former, DC-Swin, RS-Mamba, boundary-aware patterns     |
| **Change Detection**      | Siamese (FC-Siam-diff), BIT, ChangeFormer, ChangeMamba, semantic CD, continuous monitoring       |
| **Data Fusion**           | SAR+optical, RGB+DEM, hyperspectral+LiDAR, early/mid/late fusion, modality dropout, CMX, DOFA    |
| **Temporal Analysis**     | ConvLSTM, U-TAE, TSViT, Presto, TempCNN, irregular time series, cloud masking                    |
| **Foundation Models**     | SatMAE, Prithvi-EO, Clay, DOFA, GeoSAM, RemoteCLIP, LoRA/PEFT adaptation, multispectral adapters |
| **Object Detection**      | RetinaNet, FCOS, DETR/DINO, oriented bbox (DOTA), small-object patterns, YOLOv8-OBB              |
| **Interpretability & UQ** | Grad-CAM, MC dropout, deep ensembles, evidential learning, conformal prediction, calibration     |

### Code Templates (PyTorch + TensorFlow)

- Multispectral input adapters for ImageNet weights
- Minimal U-Net with shape annotations
- Siamese change-detection (FC-Siam-diff)
- Combined Dice + CE loss
- Training loop with mixed precision + gradient accumulation
- Tile-based inference with cosine-weighted blending
- Deep ensemble patterns

---

## 💡 Example Use Cases

<details>
<summary><strong>Architecture Selection</strong></summary>

**Query:**

```
I need to detect crop field boundaries from 10m Sentinel-2 data
with only 500 labeled samples. What architecture should I use?
```

**Response:**

- Recommends foundation-model fine-tuning (Prithvi-EO with LoRA)
- Explains why label efficiency matters
- Provides PyTorch implementation with proper multispectral input handling
- Lists failure modes (GSD mismatch, small fields <2 pixels)
- Suggests alternatives (SatMAE++, from-scratch U-Net as baseline)
</details>

<details>
<summary><strong>Multi-Modal Fusion Design</strong></summary>

**Query:**

```
How do I combine SAR and optical data for flood mapping?
Should I fuse early or late?
```

**Response:**

- Analyzes modality differences (speckle vs additive noise, cloud penetration)
- Recommends mid-level cross-attention fusion (not early concat)
- Flags modality dropout requirement for robustness
- Provides dual-encoder architecture with shape annotations
- Cites CMX, DeCUR as reference implementations
</details>

<details>
<summary><strong>Code Review</strong></summary>

**Query:**

```
Review my U-Net for building extraction — using ImageNet weights
with 13 Sentinel-2 bands via channel concat.
```

**Response:**

- **Blocking issue:** Silently dropping spectral information (red-edge, SWIR)
- **Fix:** Use multispectral adapter pattern from code templates
- Checks receptive field vs GSD (10m → need larger context)
- Suggests boundary-aware loss for cadastral precision
- Verifies train/val split is region-based, not random tiles
</details>

<details>
<summary><strong>Foundation Model Selection</strong></summary>

**Query:**

```
Should I use SatMAE or DINOv2 for land cover classification on EuroSAT?
```

**Response:**

- Compares both: SatMAE native multispectral vs DINOv2 stronger features
- Recommends SatMAE++ with linear probe baseline first
- Provides LoRA fine-tuning template for parameter efficiency
- Explains when each wins (SatMAE: <10k labels; DINOv2: RGB-only high-res)
- Notes EuroSAT is small (27k tiles) — foundation models mandatory
</details>

---

## 🧠 How It Works

### Three-Mode Workflow

The skill automatically identifies your intent:

1. **Build Mode** — You want a new architecture or implementation
   - Task-shape diagnosis (what's being predicted, input modality, GSD, label budget)
   - Routes to relevant subdomain reference (segmentation, fusion, temporal, etc.)
   - Produces 5-part recommendation: What / Why / Cost / Failure Modes / Alternatives

2. **Analyze Mode** — You want to understand an existing architecture/paper
   - Structured walkthrough: data flow → tensor shapes → key decisions → novelty
   - Comparisons to baselines and related work
   - Diagrams when branches/attention is non-trivial

3. **Review Mode** — You want critique or debugging help
   - 10-section checklist (task-architecture match, data flow, loss, evaluation, etc.)
   - Blocking issues with concrete fixes (not "consider regularization")
   - Acknowledges strengths + flags weaknesses with evidence

### Cross-Cutting Principles

Built into every recommendation:

- **Channels ≠ RGB** — multispectral input needs proper adapters, never silent band dropping
- **GSD matters** — receptive field, augmentation, class definitions all depend on ground sampling distance
- **Geographic generalization** — held-out region tests, not random tile splits
- **Foundation models changed the calculus** — for <50k labels, fine-tune beats from-scratch
- **Temporal context is often free signal** — multi-date stacks almost always beat single-date
- **Class imbalance is the norm** — weighted loss / focal loss / sampling strategies required

---

## 📂 Repository Structure

```
rs-dl-architect/
├── README.md                  # This file
├── docs/                      # GitHub Pages documentation site
│   └── index.html
├── rs-dl-architect/          # The actual skill
│   ├── SKILL.md              # Main entry point (~120 lines)
│   └── references/           # Loaded on-demand
│       ├── segmentation.md
│       ├── change-detection.md
│       ├── data-fusion.md
│       ├── temporal-analysis.md
│       ├── foundation-models.md
│       ├── object-detection.md
│       ├── interpretability-uncertainty.md
│       ├── review-checklist.md
│       ├── code-templates.md
│       └── design-principles.md
├── LICENSE
└── .gitignore
```

**Size:** ~1,460 lines of expert guidance across 10 files  
**Main file:** 120 lines (routes to references)  
**References:** 100–300 lines each (loaded only when relevant)

---

## 🎓 Target Users

- Remote sensing researchers
- GIS engineers and geospatial AI practitioners
- Graduate students in earth observation / geospatial ML
- Satellite imagery ML engineers
- Anyone working with Sentinel-1/2, Landsat, PlanetScope, aerial imagery, hyperspectral, SAR, or multi-modal EO data

**Assumes:** MS/PhD-level technical depth, familiarity with PyTorch or TensorFlow, working knowledge of remote sensing fundamentals.

---

## 🤝 Contributing

Contributions welcome! Areas of particular interest:

- Additional code templates (e.g., TensorFlow equivalents, JAX patterns)
- Newer foundation models (as they're released)
- Domain-specific extensions (agriculture, disaster response, urban planning)
- More review checklist items
- Dataset recommendations and benchmark updates

**To contribute:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-reference`)
3. Make your changes (follow the existing reference structure)
4. Test with realistic queries
5. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This skill is released under the [MIT License](LICENSE).

You are free to:

- Use commercially
- Modify and redistribute
- Use privately
- No warranty or liability

---

## 🔗 Links

- **Documentation:** [https://Rupesh4604.github.io/rs-dl-architect](https://Rupesh4604.github.io/rs-dl-architect)
- **Issues & Discussions:** [GitHub Issues](https://github.com/your-username/rs-dl-architect/issues)
- **Claude Skills Specification:** [Anthropic Skills Repo](https://github.com/anthropics/skills)
- **Related Work:** [Awesome Remote Sensing Change Detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)

---

## 📊 Stats

![Lines of Code](https://img.shields.io/badge/lines-1460-brightgreen)
![References](https://img.shields.io/badge/references-10-blue)
![Frameworks](https://img.shields.io/badge/frameworks-2-orange)

**Coverage:** Segmentation • Change Detection • Data Fusion • Temporal • Foundation Models • Detection • Interpretability • Review

---

## 🙏 Acknowledgments

Built for the remote sensing and geospatial AI community. Special thanks to:

- The teams behind SatMAE, Prithvi-EO, Clay, GeoSAM, and other RS foundation models
- Authors of BIT, ChangeFormer, U-TAE, and other open-source RS architectures
- The broader earth observation ML research community

---

## 📮 Contact

**Maintainer:** [M Rupesh Kumar Yadav](<[@Rupesh4604](https://github.com/Rupesh4604)>)  
**Email:** rupesh32003@gmai.com  
**Institution:** IIT Bombay, M.Tech

For questions, feedback, or collaboration opportunities, open an issue or reach out directly.

---

<p align="center">
  <strong>Built with ❤️ for the remote sensing community</strong>
  <br>
  <sub>Making deep learning architecture decisions clearer, one satellite image at a time</sub>
</p>
