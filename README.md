# 3D Neuron Visualizer

Interactive 3D visualizations of neuron morphologies and synaptic connectivity from [neuPrint](https://neuprint.janelia.org/).

If you're new to Python, see [README_IF_YOURE_LOST.md](README_IF_YOURE_LOST.md) for setup help.

### Examples

Pre-built visualizations (self-contained HTML files — download and open in Chrome):

- **EPG** — 50 neurons, 2 types — [download](https://drive.google.com/file/d/1MpyMU-bE7-B6wzMu7Oewg9p3Y9eFSb87/view?usp=drive_link) | [notebook](Examples/EPG/)
- **FB** — 601 neurons, 166 types, custom color modes — [download](https://drive.google.com/file/d/1JEHjA27eSceIz5E_SPy_g1vUlZLB5H4A/view?usp=drive_link) | [notebook](Examples/FB/)

## Quick Start

1. **Clone and install**:
   ```bash
   git clone https://github.com/your-username/neuron-visualizer.git
   cd neuron-visualizer
   pip install -r requirements.txt
   ```
   Or with conda:
   ```bash
   conda create -n neuron_visualizer python=3.11
   conda activate neuron_visualizer
   conda install -c conda-forge gevent   # needed before pip on some systems
   pip install -r requirements.txt
   ```

2. **Get your neuPrint token** at https://neuprint-cns.janelia.org/ → Account → Auth Token.

3. **Open `generate_viz.ipynb`** and fill in your token, pattern, and optional color CSVs, then run.

   Or from the command line:
   ```bash
   export NEUPRINT_TOKEN='your_token_here'

   python generate_visualization.py FB
   python generate_visualization.py EPG --use-meshes
   python generate_visualization.py FB --use-meshes --mesh-faces auto --max-file-mb 500
   python generate_visualization.py EPG --continuous scores.csv modalities.csv
   python generate_visualization.py FB --categorical regions.csv --open
   python generate_visualization.py --all --server neuprint-cns.janelia.org --dataset cns
   ```

4. **Open the HTML** — each visualization is a fully self-contained file (no server needed).

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `patterns` | *(required)* | One or more neuron type patterns (e.g. `FB`, `EPG`, `'^LNO\|GLNO.*'`) |
| `--all` | | Generate all standard patterns |
| `--server` | `neuprint-cns.janelia.org` | neuPrint server URL |
| `--dataset` | `cns` | neuPrint dataset |
| `--token` | `$NEUPRINT_TOKEN` | neuPrint API token (or set env var) |
| `--continuous` | | Continuous color mode CSVs |
| `--categorical` | | Categorical color mode CSVs |
| `--use-meshes` | off | Fetch and embed 3D neuron meshes |
| `--mesh-faces` | `auto` | Faces per mesh: `auto`, a number, or `0` for no decimation |
| `--max-file-mb` | `500` | Target file size in MB when mesh-faces=auto |
| `--skip-synapses` | off | Skip synapse position fetching |
| `--synapse-limit` | unlimited | Max synapses to keep |
| `--output-dir` | `Examples/` | Output directory |
| `--open` | off | Open HTML in browser when done |

## Color Modes

Every visualization has four built-in color modes: **Cell Type**, **Instance**, **Predicted NT**, and **Custom**.

- **Cell Type**: each type gets a unique color
- **Instance**: each individual neuron gets a unique color
- **Predicted NT**: colored by predicted neurotransmitter (acetylcholine, GABA, glutamate, etc.)
- **Custom**: user-editable — click any color swatch in the sidebar to pick a new color

You can add more by providing CSVs:

### Continuous

CSVs with a `type` column and one or more numeric columns. Each column becomes its own color mode.

- Divergent data (has negative values) automatically gets a **RdBu** colormap
- Non-divergent data gets a sequential colormap (Oranges, Purples, Greens, etc.)
- Right-click any color mode button to swap colormaps

### Categorical

CSVs with a `type` column and one or more category columns. Types sharing a value share a color.

## Neuron Meshes

When `--use-meshes` is enabled, 3D neuron surface meshes are fetched from neuPrint and embedded in the HTML alongside skeletons. Toggle between Mesh and Skeleton views in the right panel.

- **`auto` decimation** (default): maximizes mesh quality while keeping the file under the target size
- **Mesh ROI clipping**: faces are pre-assigned to ROIs so clipping works in mesh mode
- **Mesh hover**: uses GPU pixel picking for instant identification even with millions of triangles

## Synapse Visualization

Each HTML can include individual synapse position data. Right-click any partner row in the Connections panel to show presynapses (upstream) or postsynapses (downstream) as 3D spheres.

- **Synapse Groups panel**: tracks all visible synapse groups with visibility toggles, color pickers (presets + RGB sliders), and a global size slider
- **Split by instance**: right-click a synapse group to break it into per-neuron sub-groups
- **Hover info**: the info box shows "Synapse: TypeA → TypeB" when hovering over synapse spheres
- **Magnifier**: toggle the magnifier (top bar) for precise hover using GPU pixel picking

## Session Persistence

Your session state is **automatically saved** to localStorage and restored when you reopen the file:

- Highlighted types/neurons, clip states, camera position
- Color mode, color filter settings, Custom mode colors
- Saved view and selection banks
- Synapse groups (with colors and visibility)

### Sharing Sessions

- **Export** (⬇ button): downloads a `_session.json` file
- **Import** (⬆ button): loads a session file — the recipient sees your exact view

## Video Export

Click the 🎬 button in the top bar to record rotation videos:

- **Motion**: full 360° rotation or pivot (±N°)
- **Axis**: world (D/V, L/R, A/P) or viewport (screen X/Y/Z)
- **Orbit center**: scene center or selection centroid
- **Formats**: AVI (universal — plays in VLC), WebM, GIF
- **Preview**: preview the animation before recording

## Utility Scripts

### `add_synapses.py` — Add Synapse Data

Add synapse position data to HTMLs that don't have it:

```bash
python add_synapses.py --all                  # All HTMLs missing synapse data
python add_synapses.py Examples/aMe_visualization.html  # Specific file
```

## Notebook Reference

The notebook (`generate_viz.ipynb`) exposes these inputs:

| Variable | Description |
|----------|-------------|
| `NEUPRINT_TOKEN` | Your neuPrint API token |
| `NEUPRINT_SERVER` | neuPrint server URL |
| `NEUPRINT_DATASET` | Dataset to query |
| `PATTERN` | neuPrint regex (e.g. `'^FB.*'`, `'^EPG.*'`) |
| `CONTINUOUS_CSVS` | List of paths to continuous score CSVs |
| `CATEGORICAL_CSVS` | List of paths to categorical CSVs |
| `USE_MESHES` | `True` to fetch 3D neuron meshes |
| `MESH_FACES` | `'auto'`, a number, or `None` |
| `MAX_FILE_MB` | Target file size for auto decimation |
| `LOAD_SYNAPSES` | `True` to fetch synapse positions |
| `RELOAD_SYNAPSES` | `True` to re-fetch even if .json exists |

### CSV Formats

**Continuous CSV** — columns: `type`, then one or more numeric columns:
```
type,valence_score
FB2M_a,0.0926
FB2F_b,0.0827
```

Multi-column continuous CSVs create one color mode per column:
```
type,olfactory,gustatory,visual
FB1A,3.43e-06,5.21e-06,1.85e-08
```

**Categorical CSV** — columns: `type`, then one or more category columns:
```
type,region
FB1A,dorsal
FB2C,ventral
```

## File Structure

```
generate_visualization.py   # Main script (Python data pipeline + JS application)
generate_viz.ipynb           # Notebook interface
add_synapses.py              # Add synapse data to existing HTMLs
Color_by_CSVs/               # Score and modality CSVs
Examples/                    # Generated HTML visualizations
Archive/                     # Old notebooks and scripts
three.min.js                 # Three.js r160 (inlined into HTMLs)
OrbitControls.js             # Orbit camera controls (inlined)
TrackballControls.js         # Trackball camera controls (inlined)
Chucky_gold_cropped.png      # Loading screen logo
```

## Keyboard & Mouse Controls

| Action | Control |
|--------|---------|
| Rotate | Left-click drag |
| Pan | Right-click drag |
| Zoom | Scroll wheel |
| Highlight type/neuron | Double-click sidebar row |
| Load connections | Single-click a highlighted row |
| Show synapses | Right-click a partner in Connections panel |
| Split synapse group | Right-click a group in Synapse Groups panel |
| Shuffle colors | Click the 🔀 button (top bar) |
| Screenshot | Click the 📷 button (top bar) |
| Record video | Click the 🎬 button (top bar) |
| Save camera view | Click "+ Add view" in the camera panel |
| Reset camera | Click the ↺ button (top bar) |
| Toggle magnifier | Click the 🔍 button (top bar) |
