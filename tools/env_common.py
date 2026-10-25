#!/usr/bin/env python3
"""Shared helper: a MultiUserSimulationEnvironment shell carrying only dimensions.

The oracles call production helpers that take an `env` argument
(build_effective_channels, power_whitening, the BD null basis) but drive them
from explicit (K, Nt, Nr, NRF, d) tuples rather than from a configured
simulation. Constructing a real MultiUserSimulationEnvironment would run its
whole __init__ (channel model, pilot design, ...), so the oracles build an
uninitialized instance and set the six dimension attributes those helpers read.

This existed twice -- pipeline_ref.py and frontend_ref.py, differing only in
whether the assignments were tuple-packed -- and chain_ref.py and
exp_threshold_policy.py imported it *from pipeline_ref*, i.e. an experiment
script depended on a bench oracle for a one-line helper. One copy, here.

If a production helper starts reading another attribute, this shell raises
AttributeError instead of silently feeding a wrong oracle.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "mix_precode" / "classical"))
from multiuser_simulation_environment import MultiUserSimulationEnvironment as Env


def dummy_env(K, Nt, Nr, NRF, d):
    """Uninitialized Env carrying only the six dimension attributes."""
    env = object.__new__(Env)
    env.num_users = K
    env.num_tx_antennas = Nt
    env.num_rx_antennas = Nr
    env.num_rf_chains = NRF
    env.num_streams_per_user = d
    env.total_streams = K * d
    return env
