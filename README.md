# Particle collision simulator

Interactive particle accelerator simulator built with Streamlit. Configure two beams (protons, electrons, positrons, antiproton), set collision energy and visualize the resulting decay products in 3D. Includes a statistical display and CSV export.

## Setup:

### Linux/macOS:
#### 1. Create and activate a .venv, install requirements:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows:
#### 1. Create and activate a .venv, install requirements:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the app:
```
streamlit run particle_sim.py |optional| --browser.gatherUsageStats false
```

## Usage
- Choose beam particle types and collision energy in the sidebar
- Click "Trigger collision" to simulate one event
- Rotate/zoom the 3D view to inspect tracks
- Use tabs for Distribution (histogram), Particle types (pie chart + legend), Event log, and Data export (CSV)

## How it works
- Particle definition: `ParticleType` dataclass (name, symbol, mass, charge, color)
- `RelativisticParticle` approximates energy/momentum/gamma/velocity
- Center-of-mass energy computed from 4-momentum; this drives particle multiplicity and types
- Decay products are genereted and displayed as radial tracks in the 3d Plotly scene
- Physical relations used include `E^2 = (pc)^2 +(mc^2)^2` and `gamma=E/mc^2`

## AI and tooling disclosure

- AI assistance: Google Gemini (cloud) and Gemma (local) used to help design the statisctical plots (`plot_energy_distribution` and `plot_particle_types`) and to help spot/fix typos during debugging and developement
- Pyright was used