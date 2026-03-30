# Getting Started — Complete Setup Guide

This guide walks you through everything from scratch. No prior Python experience assumed.

---

## What You're Setting Up

This tool generates interactive 3D visualizations of neurons from the [neuPrint](https://neuprint.janelia.org/) connectome database. You'll end up with self-contained HTML files that open in any web browser — no server or special software needed to view them.

To *generate* those files, you need Python and a few libraries installed. That's what this guide covers.

---

## Step 1: Install Python (via Anaconda)

Anaconda gives you Python plus a package manager that handles dependencies cleanly.

### Mac

1. Go to https://www.anaconda.com/download
2. Download the **macOS** installer (Apple Silicon if you have an M1/M2/M3/M4 Mac, Intel otherwise)
3. Open the downloaded `.pkg` file and follow the installer
4. When it's done, open **Terminal** (search for "Terminal" in Spotlight)
5. Type `conda --version` and press Enter. You should see something like `conda 23.x.x`

### Windows

1. Go to https://www.anaconda.com/download
2. Download the **Windows** installer
3. Run the `.exe` file. **Important**: check "Add Anaconda to my PATH" when the installer asks
4. When it's done, open **Anaconda Prompt** (search for it in the Start menu — NOT regular Command Prompt)
5. Type `conda --version` and press Enter. You should see a version number

---

## Step 2: Create a Python Environment

An "environment" is an isolated workspace so this project's libraries don't interfere with anything else on your computer.

Open Terminal (Mac) or Anaconda Prompt (Windows) and run these commands one at a time:

```bash
conda create -n neuron_visualizer python=3.9
```

It will ask you to confirm — type `y` and press Enter.

Then activate it:

```bash
conda activate neuron_visualizer
```

Your prompt should now show `(neuron_visualizer)` at the beginning. **Every time you want to use this tool, you need to activate the environment first** with that same `conda activate neuron_visualizer` command.

---

## Step 3: Install Dependencies

With your environment activated (you should see `(neuron_visualizer)` in your prompt), run:

```bash
conda install -c conda-forge gevent
```

Type `y` to confirm. Then:

```bash
pip install -r requirements.txt
```

(Make sure you're in the `neuron-visualizer` folder when you run this. If not, `cd` into it first — see Step 4.)

This installs everything the tool needs. It might take a few minutes.

---

## Step 4: Download This Repository

If you haven't already, download this code:

**Option A — If you have git:**
```bash
git clone https://github.com/sauvolac1/neuron-visualizer.git
cd neuron-visualizer
```

**Option B — No git:**
1. Download the ZIP from GitHub (green "Code" button → "Download ZIP")
2. Unzip it somewhere you'll remember (like your Desktop or Documents folder)
3. In Terminal/Anaconda Prompt, navigate to it:
   ```bash
   cd ~/Desktop/neuron-visualizer    # Mac
   cd C:\Users\YourName\Desktop\neuron-visualizer    # Windows
   ```

---

## Step 5: Get Your neuPrint Token

1. Go to https://neuprint-cns.janelia.org/
2. Sign in with your Google account
3. Click your profile icon (top right) → **Account**
4. Copy the **Auth Token** (it's a long string starting with `eyJ...`)

You'll paste this into the notebook in the next step.

---

## Step 6: Generate a Visualization

### Option A: Jupyter Notebook (Recommended)

1. With your environment activated, run:
   ```bash
   jupyter notebook
   ```
2. A browser window will open. Click on `generate_viz.ipynb`
3. In the notebook:
   - Paste your token into `NEUPRINT_TOKEN = '...'`
   - Set `PATTERN` to the neurons you want (e.g. `'^EPG.*'`)
   - Run all cells (Cell → Run All, or Shift+Enter through each cell)
4. When it's done, your HTML file will be in the `Examples/` folder

### Option B: Command Line

```bash
export NEUPRINT_TOKEN='paste_your_token_here'    # Mac
set NEUPRINT_TOKEN=paste_your_token_here          # Windows

python Core_Code/generate_visualization.py EPG --use-meshes
```

---

## Step 7: Open Your Visualization

Find the HTML file in the `Examples/` folder (e.g. `EPG_visualization.html`) and double-click it. It opens in Chrome, Firefox, Safari, or Edge — whatever your default browser is.

That's it. The HTML file is completely self-contained. You can email it, put it on a USB drive, or share it however you like. The recipient doesn't need Python or any special software — just a web browser.

---

## Troubleshooting

### "No module named 'navis'" (or any other module)

You forgot to activate the environment. Run:
```bash
conda activate neuron_visualizer
```

### "conda: command not found"

Anaconda isn't in your PATH. On Mac, try closing and reopening Terminal. On Windows, use **Anaconda Prompt** instead of regular Command Prompt.

### "gevent" or "cloud-volume" fails to install with pip

Install gevent via conda first:
```bash
conda install -c conda-forge gevent
pip install cloud-volume
```

### The notebook won't open

Make sure Jupyter is installed in your environment:
```bash
pip install jupyter
jupyter notebook
```

### "No default Client" error

Your token isn't set. Make sure you've pasted it into the notebook's `NEUPRINT_TOKEN` field, or set the environment variable before running the command line tool.

### The HTML file is blank or broken

Try opening it in Chrome. Some features may not work in older browsers. Safari and Firefox work too, but Chrome gives the best performance for large files.

### It's taking forever to generate

Large neuron sets (hundreds of neurons) take time. The progress bars tell you what's happening. The synapse fetch is the slowest part — it only happens once, then the results are saved as a `.json` file next to your HTML so future runs are fast.

---

## Updating

If you get a newer version of this code:

```bash
cd neuron-visualizer
git pull                    # if you used git
pip install -r requirements.txt  # in case dependencies changed
```

To regenerate an HTML with the latest features, just re-run the notebook or command line — the synapse `.json` file is reused so it's much faster the second time.
