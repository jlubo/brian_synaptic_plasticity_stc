## Brian implementation of synaptic plasticity (calcium-induced, with synaptic tagging and capture) in neuronal networks 

This code implements a single current-based synapse which can undergo early-phase plasticity based on calcium dynamics and late-phase plasticity based on synaptic tagging and capture.

The early-phase dynamics are driven by neuronal spiking activity, and the late-phase dynamics depend on the early-phase through tag setting and protein synthesis (see [Luboeinski and Tetzlaff, 2021](https://doi.org/10.1038/s42003-021-01778-y), for details).

The simulation reproduces the outcome of a stand-alone simulator (see [this](https://github.com/jlubo/memory-consolidation-stc) repo) and of the Arbor simulator (see [this](https://github.com/jlubo/arbor_2N1S) repo).

### Usage

First of all, [Brian 2](https://briansimulator.org/) needs to be installed:
```
pip install brian2
```

To run the simulations and average over trials and batches, execute
```
python3 run.py
```

To run a single specific simulation, e.g., for late-phase dynamics, execute
```
python3 -c "import brianSynapseBasic as bsb; bsb.simulate('config_basic_early.json', '.', record_spikes = True)"
```

For tests and to obtain the line coverage, [pytest](https://pytest.org/) and [coverage.py](https://coverage.readthedocs.io/) need to be installed:
```
pip install -U pytest pytest-cov coverage
```

To run tests and determine the test coverage for the main module, execute
```
source run_tests
```
