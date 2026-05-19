import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt

# * Physics constants
C = 299792458  # * speed of light - m/s
ELECTRON_MASS = 0.511  # * MeV/c^2
PROTON_MASS = 938.3  # * MeV/c^2
MUON_MASS = 105.7  # * MeV/c^2


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
    "proton": ParticleType("Proton", "p+", PROTON_MASS, 1, "red"),
    "electron": ParticleType("Electron", "e-", ELECTRON_MASS, -1, "blue"),
    "positron": ParticleType("Positron", "e+", ELECTRON_MASS, 1, "green"),
    "antiproton": ParticleType("Antiproton", "p-", PROTON_MASS, -1, "pink"),
    "photon": ParticleType("Photon", "y", 0.0, 0, "gold"),
    "muon": ParticleType("Muon", "mi-", MUON_MASS, -1, "lightblue"),
    "antimuon": ParticleType("Antimuon", "mi+", MUON_MASS, 1, "lightblue"),
}


class RelativisticParticle:
    """
    * Represents a particele with relativistic properties
    * Creates attributes for: Kinetic eneegy, relativistic momentum, Lorenz factoc, and velocity as a fraction of C
    """

    def __init__(self, particle_type: ParticleType, energy_tev: float):
        """
        * initializes a RelativisticParticle
        * args:
        *   particle_type: particle name, symbol, mass, charge, color
        *   energy_tev: kinetic energy in TeV
        """
        self.type = particle_type
        self.energy_mev = energy_tev * 10**6

        self.momentum = self._calculate_momentum()
        self.gamma = self._calculate_gamma()
        self.velocity = self._calculate_velocity()

    def _calculate_momentum(self) -> float:
        """
        * Calculating relativistic momentum in MeV/c
        * relativistic momentum: E^2 = (pc)^2 + (mc^2)^2
        * => p = 1/c * sqrt(E^2 - (mc^2)^2)
        """
        total_energy = self.energy_mev + self.type.mass
        if self.type.mass == 0:  # * photon
            return total_energy
        return np.sqrt(total_energy**2 - self.type.mass**2)

    def _calculate_gamma(self) -> float:
        """
        * Calculating Lorenz factor: y = E/(mc^2)
        """
        if self.type.mass == 0:
            return np.inf
        return (self.energy_mev + self.type.mass) / self.type.mass

    def _calculate_velocity(self) -> float:
        """
        * Calculating velocity as a fraction of c
        """
        if self.type.mass == 0:
            return 1.0
        return np.sqrt(1 - 1 / self.gamma**2)

    def get_4momentum(self) -> np.ndarray:
        """
        * returns a 4-momentum vecotr (E/c, px, py, pz)
        """
        E = self.energy_mev + self.type.mass
        return np.array(
            [E, 0, 0, 0]
        )  # * spacial components set to zero, since it's a collision on x-axis only


class CollisionEvent:
    """
    * Represents a single collision event with products
    """

    def __init__(self, p1: RelativisticParticle, p2: RelativisticParticle):
        """
        * Initialize collision between two particles
        """
        self.particle1 = p1
        self.particle2 = p2
        self.timestamp = datetime.now()
        self.products: List[Tuple[ParticleType, float, float, float]] = []
        self.com_energy = self._calculate_com_energy()

    def _calculate_com_energy(self) -> float:
        """
        * Calculates center-of-mass energy in TeV
        """
        p1_4mom = self.particle1.get_4momentum()
        p2_4mom = self.particle2.get_4momentum()

        total_4mom = p1_4mom + p2_4mom
        E_total = total_4mom[0]

        return E_total / 10**6

    def generate_decay_products(
        self,
    ) -> List[Tuple[ParticleType, float, float, float]]:
        """
        * Generates decay products in 3D
        * return: List of (partle_type, theta, phi, energy_fraction)
        """
        energy_tev = self.com_energy
        products = []

        # * Determines the number and types based on energy
        if energy_tev < 0.001:
            n_particles = np.random.randint(2, 4)
            particle_keys = ["photon"]
            probabilities = [1.0]
        elif energy_tev < 0.2:
            n_particles = np.random.randint(3, 6)
            particle_keys = ["photon", "electron", "positron"]
            probabilities = [0.5, 0.25, 0.25]
        elif energy_tev < 1.0:
            n_particles = np.random.randint(4, 8)
            particle_keys = ["photon", "electron", "positron", "muon", "antimuon"]
            probabilities = [0.3, 0.2, 0.2, 0.15, 0.15]
        else:
            n_particles = np.random.randint(6, 15)
            particle_keys = [
                "photon",
                "electron",
                "positron",
                "muon",
                "antimuon",
                "proton",
                "antiproton",
            ]
            probabilities = [0.25, 0.15, 0.15, 0.15, 0.15, 0.075, 0.075]

        for _ in range(n_particles):  # * 3D visualisation, using the spherical coordinates
            phi = np.random.uniform(0, 2 * np.pi)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)

            particle_key = np.random.choice(particle_keys, p=probabilities)
            particle_type = PARTICLES[particle_key]
            energy_frac = np.random.uniform(0.1, 1.0)

            products.append((particle_type, theta, phi, energy_frac))

        self.products = products
        return products


def initialize_session_state() -> None:
    """
    * initializes streamlit state variables
    """
    if "collision_data" not in st.session_state:
        st.session_state.collision_data = []

    if "total_collisions" not in st.session_state:
        st.session_state.total_collisions = 0

    if "total_particles_created" not in st.session_state:
        st.session_state.total_particles_created = 0
    
    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    
    if "current_products" not in st.session_state:
        st.session_state.current_products = []

def create_event_display(show_pipe: bool, collision_active: bool, decay_products: List[Tuple[ParticleType, float, float, float]]) -> go.Figure:
    """
    * Creates the beam projection with particles, and their respectable traces after collision
    """
    fig = go.Figure()

    # * creating the beam pipe
    if show_pipe:
        length = 100
        radius = 10
        nb_points = 50
        theta = np.linspace(0, 2 * np.pi, nb_points)
        x_pipe = np.linspace(-length, length, 2)

        for t in theta:
            fig.add_trace(
                go.Scatter3d(
                    x=x_pipe,
                    y=[radius * np.cos(t)] * 2,
                    z=[radius * np.sin(t)] * 2,
                    mode="lines",
                    line=dict(color="rgba(100, 100, 255, 0.1)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # * collision calc - vertexes and trails
    if collision_active and decay_products:
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=8, color="white", symbol="diamond"),
            name="Collision Vertex",
            hovertext="Collision Point"
        ))

        # * making trails
        for particle_type, theta, phi, energy_frac in decay_products:
            r = 20 + energy_frac * 60

            dx = r * np.sin(theta) * np.cos(phi)
            dy = r * np.sin(theta) * np.sin(phi)
            dz = r * np.cos(theta)

            # * draw track + trail for specific particle
            fig.add_trace(go.Scatter3d(
                x=[0, dx],
                y=[0, dy],
                z=[0, dz],
                mode="lines+markers",
                line=dict(color=particle_type.color, width=4),
                marker=dict(size=3, color=particle_type.color),
                name=f"{particle_type.symbol}",
                hovertext=f"{particle_type.name}<br>Energy: {energy_frac:.2f}",
                showlegend=True,
                ))

    # * camera view
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    return fig

def plot_energy_distribution(collision_data: pd.DataFrame) -> None:
    """
    * Rendering a historam of center-of-mass energies using Matplotlib
    * input - collision_data
    * Output - render of plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(collision_data["com_energy_tev"], bins=20, color="#00ffff", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Center-of-mass energy (TeV)")
    ax1.set_ylabel("Collisions")
    ax1.set_title("Collision energy distribution")
    ax1.set_facecolor(("#1a1a1a"))
    ax1.grid(True, alpha=0.3)

    ax2.hist(collision_data["n_products"], bins=range(2, int(collision_data["n_products"].max()) + 2), color="#ff4444", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Particles per collision")
    ax2.set_ylabel("Collisions")
    ax2.set_title("Particle multiplicity distribution")
    ax2.set_facecolor("#1a1a1a")
    ax2.grid(True, alpha=0.3)

    fig.patch.set_facecolor("#e0e0e0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_particle_types(collision_data: pd.DataFrame) -> None:
    """
    * Renders a pie chart of created particles using matplotlib
    * input - collision_data
    * Output - render of plot
    """
    all_particles: List[str] = []
    for products_str in collision_data["products"]:
        all_particles.extend(products_str.split(", "))

    counts = pd.Series(all_particles).value_counts()
    symbol_to_color = {p.symbol: p.color for p in PARTICLES.values()}
    colors = [symbol_to_color.get(s, "#888888") for s in counts.index]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 10, "color": "white"})
    ax.set_title("Particle type distribution", fontsize=12, fontweight="bold", color="white")
    fig.patch.set_facecolor("#0a0a0a")
    st.pyplot(fig)
    plt.close()

def main():
    st.set_page_config(page_title="Particle collision simulator", layout="wide")

    initialize_session_state()

    st.title("Particle collision simulator")

    with st.sidebar:
        st.title("Config")
        st.subheader("Beam setup")

        luminosity = st.slider(
            "Luminosity (\\*10^34 cm^-2\\*s^-1)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.01,
            help="Higher luminosity = more collisions per triger, scales number of events"
        )

        integration_time = st.slider(
            "Integration time (s)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Simulated duration, affects collision count via N = sigma * L * t"
        )

        collision_probability = st.slider(
            "Collision probability (%)",
            min_value=10,
            max_value=100,
            value=100,
            step=1,
            help="Probability each particle pair actually collides"
        )

        beam1_type = st.selectbox("Beam 1 Partcile",
                        options=["proton", "electron", "positron", "antiproton"],
                        format_func=lambda x: f"{PARTICLES[x].symbol} : {PARTICLES[x].name}")
        beam2_type = st.selectbox("Beam 2 Partcile",
                        options=["proton", "electron", "positron", "antiproton"],
                        format_func=lambda x: f"{PARTICLES[x].symbol} : {PARTICLES[x].name}")

        energy_presets = {
            "LHC (nominal)": 6.5,
            "LHC (maximum)": 6.8,
            "Tevatron": 0.98,
            "Custom": 7.0 
        }
        preset = st.selectbox("Accelerator preset", list(energy_presets))

        if preset == "Custom":
            energy = st.slider("Beam energy (TeV)", 0.1, 14.0, 7.0)
        else:
            energy = energy_presets[preset]
            st.write(f"Fixed energy: {energy} TeV")

        st.subheader("Display settings")
        show_pipe = st.checkbox("Show beam pipe", value=True)

        st.divider()

        run_sim = st.button(
            "Triger collision", type="primary", width="stretch"
        )
        reset_button = st.button(
            "Reset all", width="stretch"
        )

        st.divider()

        st.info("""
        **Physics notes:**
        - E^2 = (pc)^2 + (mc^2)^2
        - Higher energy = more particles
        - Track colors show particle type
        - Track length is proportional to energy
        """)

    if reset_button:
        st.session_state.collision_data = []
        st.session_state.total_collisions = 0
        st.session_state.total_particles_created = 0
        st.session_state.event_log = []
        st.session_state.current_products = []
        st.rerun()

    if run_sim:
        # * Calculates expected collisions: N = sigma * L * t
        expected_collisions = int(luminosity * integration_time)
        expected_collisions = max(1, expected_collisions)
        
        all_products = []
        collision_count = 0

        for _ in range(expected_collisions):
            if np.random.random() * 100 < collision_probability:
                p1 = RelativisticParticle(PARTICLES[beam1_type], energy)
                p2 = RelativisticParticle(PARTICLES[beam2_type], energy)

                #* collision event
                collision = CollisionEvent(p1, p2)
                products = collision.generate_decay_products()

                all_products.extend(products)
                collision_count += 1

                collision_data = {
                    "timestamp": collision.timestamp,
                    "particle1": p1.type.symbol,
                    "particle2": p2.type.symbol,
                    "com_energy_tev": collision.com_energy,
                    "n_products": len(products),
                    "products": ", ".join([p[0].symbol for p in products])
                }

                st.session_state.collision_data.append(collision_data)

                # * log event
                event_message = f"Collision: {p1.type.symbol} with {p2.type.symbol} created {len(products)} partcles at an energy level of {collision.com_energy:.2f} TeV"
                st.session_state.event_log.insert(0, (datetime.now(), event_message))

        if collision_count > 0:
            st.session_state.current_products = all_products[-30:]
            st.session_state.total_collisions += collision_count
            st.session_state.total_particles_created +=len(all_products)

    # * Main display
    st.header("3D Event display")
    st.markdown("*Use mouse to rotate, zoom, and pan*")

    fig = create_event_display(
        show_pipe=show_pipe,
        collision_active=len(st.session_state.current_products) > 0,
        decay_products=st.session_state.current_products
    )
    st.plotly_chart(fig, width="stretch")

    # * stats
    st.divider()
    column1, column2, column3, column4 = st.columns(4)

    with column1:
        st.metric("Total collisions", st.session_state.total_collisions)
    with column2:
        st.metric("Luminosity", f"{luminosity:.1f}\\*10^34 cm^-2\\*s^-1")
    with column3:
        st.metric("CoM energy", f"{energy * 2:.1f} TeV")
    with column4:
        st.metric("Particles created", st.session_state.total_particles_created)

    # * analysis of data
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Particle types", "Event log", "Data export"])

    with tab1:
        st.subheader("Statistical analysis")
        if st.session_state.collision_data:
            df = pd.DataFrame(st.session_state.collision_data)
            plot_energy_distribution(df)
        else:
            st.info("Run collsions to see analysis")
    with tab2:
        st.subheader("Particle production")
        if st.session_state.collision_data:
            df = pd.DataFrame(st.session_state.collision_data)
            column_chart, column_legend = st.columns([2, 1])
            
            with column_chart:
                plot_particle_types(df)
            with column_legend:
                st.markdown("### Particle legend")
                legend = [
                    {
                        "Symbol":           p.symbol,
                        "Name":             p.name,
                        "Mass (MeV/c^2)":   p.mass,
                        "Charge":           f"{p.charge}"
                    }
                    for p in PARTICLES.values()
                ]
                st.dataframe(pd.DataFrame(legend), hide_index=True, width="stretch")

        else:
            st.info("Run collision to see analysis")
    with tab3:
        st.subheader("Recent events")
        if st.session_state.event_log:
            for timestamp, message in st.session_state.event_log[:20]:
                st.text(f"{timestamp.strftime('%H:%M:%S.%f')[:-3]}  {message}")
        else:
            st.info("No events yet")
    with tab4:
        st.subheader("Export collision data")
        if st.session_state.collision_data:
            df = pd.DataFrame(st.session_state.collision_data)
            csv = df.to_csv(index=False)

            column_a, column_b = st.columns(2)
            column_a.download_button(
                label       = "Download CSV",
                data        = csv,
                file_name   = f"collisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            column_b.metric("Events logged", len(df))

            st.dataframe(df, width="stretch")
        else:
            st.info("No data to export")

if __name__ == "__main__":
    main()

# source /home/remi/Documents/Programing/FJFI-Python/particle_simulation/.venv/bin/activate.fish
# streamlit run /home/remi/Documents/Programing/FJFI-Python/particle_simulation/particle_sim.py --browser.gatherUsageStats false