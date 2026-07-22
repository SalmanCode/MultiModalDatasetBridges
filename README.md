# Synthetic Bridge Dataset with Multi-Modal Data

A complete pipeline for generating synthetic bridge datasets with both 3D models and realistic laser-scanned point clouds. This repository creates parametric bridge models and simulates terrestrial laser scanning (TLS) from multiple viewpoints, providing ground-truth data for point cloud analysis, semantic segmentation, and AI model training.

## About

This work is part of ongoing research for the **I3CE Conference 2026**. If you use this dataset or methodology in your research, please consider citing this work.

## What This Pipeline Does

The complete workflow goes through three main stages:

1. **Parameter Generation** → Creates randomized but realistic bridge configurations (dimensions, spans, pier types, etc.)
2. **3D Model Generation** → Uses CadQuery to build parametric 3D bridge models with separate components (deck, piers, railings, etc.)
3. **Point Cloud Simulation** → Uses HELIOS++ to simulate laser scanning from 8 positions around each bridge, generating realistic point clouds

You end up with:
- Parametric 3D bridge models (OBJ format)
- Bridge metadata (dimensions, configurations)
- Simulated point clouds from multiple scanner positions
- Semantically segmented point clouds (each component labeled)

## Quick Start

### Complete Pipeline (Recommended)

Run everything with one command:

```bash
# Full pipeline: Generate models → Simulate scanning → Segment → Convert to NPY
python main.py --num-bridges 5 --include-components --run-simulation --semantic-segmentation --npy-conversion
```

This automatically:
1. Generates 3D bridge models with components
2. Simulates laser scanning with HELIOS++
3. Performs semantic segmentation
4. Converts to ML-ready NPY format


## Full Pipeline Overview

```
Parameter Generation (param_gen.py)
    ↓
Bridge Configuration (bridge_summary.json)
    ↓
3D Model Generation (CadQuery + bridge_model.py)
    ↓
3D Models (OBJ files: deck, piers, railings, etc.)
    ↓
Survey Planning (calculate scanner positions)
    ↓
HELIOS++ Simulation (laser scanning from 8 positions)
    ↓
Point Clouds (XYZ files from each scanner position)
    ↓
Semantic Segmentation (label each point by component)
    ↓
Final Dataset (3D models + point clouds + labels)
```

## Where to Find Everything

You can find the dataset with 1000 bridges at  https://huggingface.co/datasets/SyedSalmanAhmed/MultiModalBridgeDataset

### Dataset Structure
```
Dataset/
├─ BridgeModels/
│  └─ bridge_X/          # 3D models for each bridge
│     ├─ approach_slabs.obj
│     ├─ back_walls.obj
│     ├─ deck.obj
│     ├─ piers.obj
│     ├─ railings.obj
│     └─ wing_walls.obj
│
├─ PointCloudScans/
│  ├─ raw/               # Raw laser scan data
│  │  └─ bridge_X/
│  │     ├─ leg000_points.xyz → leg007_points.xyz
│  │     └─ bridge_X_complete.xyz
│  │
│  ├─ segmented/         # Semantically segmented scans
│  │  └─ bridge_X/
│  │     ├─ approach_slab.xyz
│  │     ├─ back_wall.xyz
│  │     ├─ deck.xyz
│  │     ├─ piers.xyz
│  │     ├─ railings.xyz
│  │     └─ wing_walls.xyz
│  │
│  ├─ npy/               # ML-ready format
│  │  └─ bridge_X.npy (8192 points × 5 features)
│  │
│  └─ scanner_positions.json  # Scanner coordinates for each bridge
│
└─ bridge_summary.json   # All bridge parameters and metadata
```


## Scanner Setup

Each bridge is scanned from **8 strategically placed positions** for complete coverage:


- **Legs 0-1**: Left and right sides of the bridge
- **Legs 2-3**: Front and back ends along the length
- **Legs 4-5**: Below the bridge (underside coverage)
- **Legs 6-7**: Above the bridge (top-down view)

Scanner positions are automatically calculated based on bridge dimensions. Adjust offsets in `helios/config.py`:

```python
WIDTH_OFFSET = 7   # Distance from bridge sides (meters)
LENGTH_OFFSET = 25 # Distance from bridge ends (meters)
```

## Bridge Types

The pipeline generates two types of bridges:

1. **Box Girder Bridges** - Hollow box cross-section, typically 2 cells across width
2. **Beam-Slab Bridges** - Multiple beam columns with slab deck

Both types include realistic variations in:
- Number of spans (1-5)
- Total length (35-160 meters)
- Width (20.5-27.5 meters)
- Pier configurations (single-column, multi-column, hammer-head)
- Pier cross-sections (circular, rectangular)

## Command Options

### Master Pipeline (All-in-One)
```bash
python main.py --num-bridges N [OPTIONS]

# Examples:
python main.py --num-bridges 10 --include-components --run-simulation --semantic-segmentation
python main.py --num-bridges 5 --bridge-type box_girder --run-simulation
python main.py --num-bridges 3 --skip-simulation  # Models only

Options:
  --num-bridges N              Number of bridges to generate (required)
  --bridge-type TYPE           box_girder or beam_slab (default: mixed)
  --include-components         Generate separate component OBJ files
  --run-simulation             Run HELIOS++ laser scanning
  --semantic-segmentation      Segment point clouds by component
  --skip-simulation            Generate models only, no scanning
  --skip-npy-conversion        Skip NPY conversion step
```


## Acknowledgments

This work builds upon open-source projects:

- **[CadQuery](https://github.com/CadQuery/cadquery)** - Parametric 3D CAD modeling in Python
- **[HELIOS++](https://github.com/3dgeo-heidelberg/helios)** - LiDAR simulation framework developed by 3DGeo Research Group, Heidelberg University


## Contact

For questions or collaboration:
- salman.ahmed@tum.de
- florian.noichl@tum.de

## License

MIT License

