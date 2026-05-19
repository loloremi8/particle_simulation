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
- **Beam configuration**: Select from matter and antimatter particles in the sidebar.
- **Luminosity scaling**: Adjust the instantaneous luminosity different run conditions. The number of expected events $N$ scales according to $N=\sigma \mathcal{L} \Delta t$.
- **Triger**: Click "Triger collision" to simulate the event batch.
- **Visualization**: Rotate/zoom the 3D view to inspect tracks.
- **Analysis**: Use tabs for Distribution (histogram), Particle types (pie chart + legend), Event log, and Data export (CSV)

## How it works

### 1. Relativistic Kinematics

The simulator treats every particle as a relativistic object where classical kinematics are insufficient. We use the energy-momentum relation:

$$E^2 = (pc)^2 + (m_0c^2)^2$$

For every particle, the code solves the **Lorentz factor** ($\gamma$) and **velocity** ($\beta = v / c$), where

- $\gamma = \frac{E}{m_0c^2} = \frac{E_{kin} + m_0c^2}{m_0c^2}$
- $\beta = \sqrt{1 - \frac{1}{\gamma^2}}$

### 2. Center-of-Mass (CoM) energy

In a collider, the useful energy for creating new particles is the invariant mass of the system. For a head-on collision of two particles with four-momentums $P_1$ and $P_2$:

$$s = (P_1 + P_2)^2 \Rightarrow E_{cm}= \sqrt{s}$$

The simulation operates in the CoM frame where $P_{total} = 0$, thus $E_{cm} = E_1 + E_2$. This energy budget determines the production thresholds for heavier particles.

### 3. Stochastic particle production

Particle generation is modelled as a multi-channel stochastic process. The types and multiplicity of decay products are energy-dependent:

- **Low energy** ($E < 200$ MeV): Dominated photon and electron pairs.

- **High energy** ($E > 1$ TeV): Enables heavy particle production channels including muons, antimuons, protons, and antiprotons.

- **Angular distribution**: Particles are distributed isotropically in 3D space to ensure a uniform distribution on the unit sphere.

## AI and tooling disclosure

- AI assistance: Google Gemini (cloud) and Gemma (local) used to help design the statisctical plots (`plot_energy_distribution` and `plot_particle_types`) and to help spot/fix typos during debugging and developement
- Pyright was used