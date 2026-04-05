import streamlit as st
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

#* Physics constants
C =  299792458          #* speed of light - m/s
ELECTRON_MASS = 0.511   #* MeV/c^2
PROTON_MASS = 938.3     #* MeV/c^2
MUON_MASS = 105.7       #* MeV/c^2

@dataclass
class ParticleType:
    """
    * Particle class with its properties
    """
    name: str
    symbol: str
    mass: float
    charge: int
    color: str

PARTICLES: Dict[str, ParticleType] = {
    "proton": ParticleType("Proton", "p-", PROTON_MASS, 1, "red"),
    "electron": ParticleType("Electron", "e-", ELECTRON_MASS, -1, "blue"),
    "positron": ParticleType("Positron", "p+", ELECTRON_MASS, 1, "green"),
    "antiproton": ParticleType("Antiproton", "p-", PROTON_MASS, -1, "pink"),
    "photon": ParticleType("Photon", "y", 0.0, 0, "gold"),
    "muon": ParticleType("Muon", "mi-", MUON_MASS, 1, "light_blue"),
    "antimuon": ParticleType("Antimuon", "mi+", MUON_MASS, 1, "light_blue")
}


def main():
    st.set_page_config(page_title="Particle collision simulator", layout="wide")

    st.title("Particle collision simulator")

    with st.sidebar:
        st.title("Config")
        energy = st.slider("Collision energy (TeV)", 0.1, 14.0, 7.0)
        n_tracks = st.number_input("Number of tracks", 2, 50, 10)
        show_pipe = st.checkbox("Show beam pipe", value=True)

        st.divider()
        run_sim = st.button("Triger collision", type="primary", use_container_width=True)

    def create_event_display(collision_active=False):
        """
        * 3D graph engine
        """
        fig = go.Figure()

        #* creating the beam pipe
        if show_pipe:
            length = 100
            radius = 10
            nb_points = 50
            theta = np.linspace(0, 2 * np.pi, nb_points)
            x_pipe = np.linspace(-length, length, 2)

            for t in theta:
                fig.add_trace(go.Scatter3d(
                    x=x_pipe,
                    y=[radius * np.cos(t)] * 2,
                    z=[radius * np.sin(t)] * 2,
                    mode="lines",
                    line=dict(color="rgba(100, 100, 255, 0.1)", width=1),
                    showlegend=False,
                    hoverinfo="skip"
                ))
        
        #* collision calc - vertexes and trails
        if collision_active:
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers",
                marker=dict(size=8, color="white", symbol="diamond"),
                name="Vertex"
            ))

            #* making trails
            for i in range(n_tracks):
                phi = np.random.uniform(0, 2 * np.pi)
                costheta = np.random.uniform(-1, 1)
                theta = np.arccos(costheta)

                #* cord transf
                r = np.random.uniform(20, 80)
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)

                #* draw track + trail
                fig.add_trace(go.Scatter3d(
                    x=[0, dx], y=[0, dy], z=[0, dz],
                    mode="lines+markers",
                    line=dict(color="cyan", width=3),
                    marker=dict(size=2, color="white"),
                    name=f"Track {i+1}",
                    showlegend=False
                ))

        #* camera view
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            height=700
        )
        
        return fig

    st.header("3D particle collision display")
    st.markdown("Mouse to rotate, zoom, and pan in the graph")

    plot_placeholder = st.empty()

    if run_sim:
        fig = create_event_display(collision_active=True)
        plot_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_event_display(collision_active=False)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Luminosity", "1.0e34 cm-2*s-1")
    with c2:
        st.metric("Center of mass energy", f"{energy} TeV")
    with c3:
        st.metric("Track count", n_tracks if run_sim else 0)

if __name__ == "__main__":
    main()