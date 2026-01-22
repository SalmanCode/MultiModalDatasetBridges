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

### Step 1: Generate Bridge Models

```bash
# Generate 10 bridges with all components
python main.py 10 --include_components
```

This creates:
- 3D models in `Generated_Bridges/BridgeObjects/`
- Bridge parameters in `Generated_Bridges/bridge_summary.json`

### Step 2: Simulate Laser Scanning

```bash
cd helios
python main.py --num-bridges 10 --run-simulation --semantic-segmentation
```

This creates:
- Survey and scene files for HELIOS++
- Simulated point clouds from 8 scanner positions
- Segmented point clouds by component type

### Test with Few Bridges First

```bash
# Generate just 2 bridges for testing
python main.py 2 --include_components

# Simulate scanning for those 2 bridges
cd helios
python main.py --num-bridges 2 --run-simulation --semantic-segmentation
```

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

### Bridge Models
- **3D Components**: `Generated_Bridges/BridgeObjects/bridge_X/`
  - `approach_slabs.obj` - Bridge approach slabs
  - `back_walls.obj` - Back abutment walls
  - `deck.obj` - Bridge deck
  - `piers.obj` - Support piers
  - `railings.obj` - Safety railings
  - `wing_walls.obj` - Wing walls at abutments

- **Full Models**: `Generated_Bridges/BridgeObjects/bridge_X.obj`
- **Metadata**: `Generated_Bridges/bridge_summary.json` - All bridge parameters

### Point Clouds
- **Raw Scans**: `helios/output/TLS_bridge_X/[timestamp]/`
  - `leg000_points.xyz` through `leg007_points.xyz` - Individual scanner positions
  - `leg000_trajectory.txt` through `leg007_trajectory.txt` - Scanner trajectories

- **Segmented Components**: Same directory after segmentation
  - `approach_slab.xyz` - Points belonging to approach slabs
  - `back_wall.xyz` - Points from back walls
  - `deck.xyz` - Deck points
  - `piers.xyz` - Pier points
  - `railings.xyz` - Railing points
  - `wing_walls.xyz` - Wing wall points

### Scanner Configuration
- **Survey Files**: `helios/data/surveys/TLS_bridge_X_survey.xml`
- **Scene Files**: `helios/data/scenes/TLS_bridge_X_scene.xml`
- **Scanner Positions**: `helios/scanner_positions.json` - Coordinates of all 8 scanner positions per bridge

### Visualizations
- **Analysis Plots**: `Generated_Bridges/analysis_plots/` - Statistical analysis of bridge parameters

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
- Number of spans (2-5)
- Total length (35-160 meters)
- Width (20.5-27.5 meters)
- Pier configurations (single-column, multi-column, hammer-head)
- Pier cross-sections (circular, rectangular)

## Command Options

### Bridge Generation
```bash
python main.py <num_bridges> [--bridge_type TYPE] [--include_components]

# Examples:
python main.py 20                              # 20 bridges, mixed types
python main.py 10 --bridge_type box_girder    # 10 box girder bridges only
python main.py 5 --include_components          # 5 bridges with separate components
```

### Point Cloud Simulation
```bash
python helios/main.py [--num-bridges N] [--run-simulation] [--semantic-segmentation]

# Examples:
python main.py --num-bridges 5                                   # Generate survey files only
python main.py --num-bridges 3 --run-simulation                  # Generate + simulate
python main.py --run-simulation --semantic-segmentation          # Full pipeline, all bridges
```

## Research Context

This dataset addresses the need for ground-truth data in bridge inspection and monitoring using point cloud analysis. The synthetic approach allows:

- **Controlled experiments** with known ground truth
- **Large-scale dataset generation** for deep learning
- **Variation testing** across different bridge configurations
- **Sensor position optimization** for real-world deployment

Perfect for research in:
- Point cloud semantic segmentation
- Bridge component detection and classification
- Structural health monitoring
- AI/ML model training and validation

## Acknowledgments

This work builds upon excellent open-source projects:

- **[CadQuery](https://github.com/CadQuery/cadquery)** - Parametric 3D CAD modeling in Python
- **[HELIOS++](https://github.com/3dgeo-heidelberg/helios)** - LiDAR simulation framework developed by 3DGeo Research Group, Heidelberg University

Special thanks to the developers and maintainers of these tools for making high-quality geometric modeling and laser scanning simulation accessible to the research community.

This work used HELIOS++ for the simulation of TLS point clouds:

**Winiwarter, L., Esmorís Pena, A., Weiser, H., Anders, K., Martínez Sanchez, J., Searle, M., Höfle, B. (2022)**: Virtual laser scanning with HELIOS++: A novel take on ray tracing-based simulation of topographic full-waveform 3D laser scanning. *Remote Sensing of Environment*, 269. [doi:10.1016/j.rse.2021.112772](https://doi.org/10.1016/j.rse.2021.112772)

## Citation

If you use this dataset or methodology in your research, please cite:

```
[Citation will be added after I3CE Conference 2026 publication]
```

## Contact

For questions or collaboration:
- salman.ahmed@tum.de
- florian.noichl@tum.de

## License

MIT License - See LICENSE file for details