import pytest # * pip install pytest in the active .venv
import numpy as np
from particle_sim import (
    RelativisticParticle,
    CollisionEvent,
    PARTICLES,
    initialize_session_state
)
import streamlit as st

class TestRelativisticParticle:
    """
    * Test relativistic physics calculations
    """
    def test_photon_properties(self):
        """
        * Photons should have v=c, gamma=inf and zero mass
        """
        photon = RelativisticParticle(PARTICLES["photon"], energy_tev=1.0)
        assert photon.velocity == 1.0
        assert photon.gamma == np.inf
        assert photon.momentum == 10**6 # * Energy in MeV

    def test_momentum_calculation(self):
        """
        * Verify momentum from E^2 = (pc)^2 + (mc^2)^2
        """
        proton = RelativisticParticle(PARTICLES["proton"], energy_tev=1.0)
        E_total = 10**6 + PARTICLES["proton"].mass # * E_total = 10^6 MeV + 983.3 MeV = 1 000 983 MeV ~ 1.001 TeV
        expected_p = np.sqrt(E_total**2 - PARTICLES["proton"].mass**2)
        assert np.isclose(proton.momentum, expected_p)

    def test_gamma_formula(self):
        """
        * Verify gamma = E_total / m
        """
        electron = RelativisticParticle(PARTICLES["electron"], energy_tev = 0.511)
        E_total = 5.11  * 10**5 + PARTICLES["electron"].mass
        expected_gamma = E_total / PARTICLES["electron"].mass
        assert np.isclose(electron.gamma, expected_gamma)

    def test_velocity_bounds(self):
        """
        * Velocity should be in (0,1> for massive particles
        """
        proton = RelativisticParticle(PARTICLES["proton"], energy_tev=0.1)
        assert 0 < proton.velocity <= 1.0


    def test_velocity_formula(self):
        """
        * Verify v = c * sqrt(1 - 1/gamma^2)
        """
        muon = RelativisticParticle(PARTICLES["muon"], energy_tev=1.0)
        expected_v = np.sqrt(1 - 1 / muon.gamma**2)
        assert np.isclose(muon.velocity, expected_v)

    def test_4momentum_structure(self):
        """
        * 4-momentum should be [E, 0, 0, 0]
        """
        electron = RelativisticParticle(PARTICLES["electron"], energy_tev=0.5)
        mom = electron.get_4momentum()
        assert len(mom) == 4
        assert mom[1] == 0 # * px
        assert mom[2] == 0 # * py
        assert mom[3] == 0 # * pz
        assert mom[0] > 0  # * E should be > 0

class TestCollisionEvent:
    """
    * Tests collision event calculations
    """
    def test_com_energy_calculation(self):
        """
        * CoM energy should be the total sum of energies both particles
        """
        p1 = RelativisticParticle(PARTICLES["proton"], energy_tev=1.0)
        p2 = RelativisticParticle(PARTICLES["proton"], energy_tev=1.0)
        collision = CollisionEvent(p1, p2)
        E1 = 10**6 + PARTICLES["proton"].mass
        E2 = 10**6 + PARTICLES["proton"].mass
        expected_com = (E1 + E2) / 10**6
        assert np.isclose(collision.com_energy, expected_com)

    def test_decay_products_structure(self):
        """
        * Generated products should have correct tuple structure
        """
        p1 = RelativisticParticle(PARTICLES["proton"], energy_tev=1.0)
        p2 = RelativisticParticle(PARTICLES["proton"], energy_tev=1.0)
        collision = CollisionEvent(p1, p2)
        products = collision.generate_decay_products()

        assert isinstance(products, list)
        assert len(products) > 0

        for particle_type, theta, phi, energy_frac in products:
            assert hasattr(particle_type, "name")   # * ParticleType instance
            assert 0 <= theta <= np.pi              # * Polar angle in bounds
            assert 0 <= phi <= 2 * np.pi            # * Azimuthal angle in bounds
            assert 0.1 <= energy_frac <= 1.0        # * energy is a fraction

    def test_product_multiplicity_by_energy(self):
        """
        * Higher energy should generally produce more particles
        """
        low_energy = CollisionEvent(
            RelativisticParticle(PARTICLES["photon"], energy_tev=0.0001),
            RelativisticParticle(PARTICLES["photon"], energy_tev=0.0001)
        )
        high_energy = CollisionEvent(
            RelativisticParticle(PARTICLES["photon"], energy_tev=5.0),
            RelativisticParticle(PARTICLES["photon"], energy_tev=5.0)
        )

        low_products = low_energy.generate_decay_products()
        high_products = high_energy.generate_decay_products()

        # * On average higher energies should produces more particles than lower energies
        assert len(high_products) >= len(low_products)

class TestSessionState:
    """
    * Tests Streamlit session state initialization
    """

    def test_initialize_creates_keys(self):
        """
        * All requiered keys should be initialized
        """
        st.session_state.clear() # * Reset
        initialize_session_state()

        assert "collision_data" in st.session_state
        assert "total_collisions" in st.session_state
        assert "total_particles_created" in st.session_state
        assert "event_log" in st.session_state
        assert "current_products" in st.session_state

    def test_initialize_default_values(self):
        """
        * Keys should have correct default types
        """
        st.session_state.clear()
        initialize_session_state()

        assert isinstance(st.session_state.collision_data, list)
        assert st.session_state.total_collisions == 0
        assert st.session_state.total_particles_created == 0
        assert isinstance(st.session_state.event_log, list)
        assert isinstance(st.session_state.current_products, list)