# Bridge Dataset Generator

This repo generates synthetic 3D bridge models and simulates laser scanning them. You get both the 3D models and realistic point cloud scans.

## What it does

1. **Generates bridge models** - Creates parametric 3D bridge models (box girder and beam-slab types) with different configurations
2. **Simulates laser scanning** - Uses HELIOS to simulate TLS (Terrestrial Laser Scanning) from multiple positions around each bridge
3. **Semantic segmentation** - Automatically segments point clouds into bridge components (deck, piers, railings, etc.)

## Quick Start

### Generate point clouds for 2 bridges:
```bash
cd helios
python main.py --num-bridges 2 --run-simulation --semantic-segmentation
```

That's it! The pipeline will create survey files, run simulations, and segment the point clouds.

## Where to find stuff

- **3D Bridge Models**: `Generated_Bridges/BridgeObjects/bridge_X/`
  - Each bridge has OBJ files for different components (deck, piers, railings, etc.)
  
- **Bridge Metadata**: `Generated_Bridges/bridge_summary.json`
  - All bridge parameters (dimensions, number of spans, pier types, etc.)

- **Point Clouds**: `helios/output/TLS_bridge_X/[timestamp]/`
  - Raw scans: `leg000_points.xyz`, `leg001_points.xyz`, etc.
  - Segmented components: `deck.xyz`, `piers.xyz`, `railings.xyz`, etc.

- **Scanner Positions**: `helios/scanner_positions.json`
  - Shows where scanners were placed for each bridge

## Configuration

Want to adjust scanner positions? Edit `helios/config.py`:

```python
WIDTH_OFFSET = 7   # How far from bridge sides
LENGTH_OFFSET = 25 # How far from bridge ends
```

## Options

```bash
# Just create survey files (no simulation)
python main.py --num-bridges 5

# Run simulations only
python main.py --num-bridges 3 --run-simulation

# Full pipeline with segmentation
python main.py --run-simulation --semantic-segmentation
```

## Scanner Setup

Each bridge is scanned from 8 positions:
- **Legs 1-2**: Left and right sides
- **Legs 3-4**: Front and back ends
- **Legs 5-6**: Below the bridge
- **Legs 7-8**: Above the bridge

This gives you complete coverage of the entire bridge structure.
