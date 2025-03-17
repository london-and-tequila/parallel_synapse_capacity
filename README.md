# Parallel Synapses with Transmission Nonlinearities

## Overview
This repository contains the implementation code for the memory capacity analysis presented in:

> Song Y, Benna MK. Parallel Synapses with Transmission Nonlinearities Enhance Neuronal Classification Capacity. bioRxiv [Preprint]. 2024 Jul 4:2024.07.01.601490. doi: 10.1101/2024.07.01.601490. PMID: 39005326; PMCID: PMC11244940.

## Reproducing Figures
The figures from the manuscript can be reproduced by running the following Jupyter notebooks:

- `Fig2.ipynb` - Main results for Figure 2
- `Fig3.ipynb` - Main results for Figure 3
- `Fig5.ipynb` - Main results for Figure 5
- `SuppleFig_A_a-e.ipynb` - Supplementary Figure A, panels a-e
- `SuppleFig_B_a.ipynb` - Supplementary Figure B, panel a
- `SuppleFig_B_b_d-g.ipynb` - Supplementary Figure B, panels b, d-g
- `SuppleFig_B_c.ipynb` - Supplementary Figure B, panel c
- `SuppleFig_C_a_d.ipynb` - Supplementary Figure C, panels a and d
- `SuppleFig_D.ipynb` - Supplementary Figure D

## Memory Capacity Calculations

### Restricted Model
To calculate the memory capacity of the restricted model, run the gradient descent algorithm in the terminal:

```
python3 parallel_synapse_gradient.py N M P seed
```

where:
- `N` = number of input neurons
- `M` = number of parallel synapses
- `P` = number of patterns
- `seed` = random seed

Example:
```
python3 parallel_synapse_gradient.py 100 2 300 1
```
This runs with N=100 input neurons, M=2 parallel synapses, P=300 patterns, and seed=1.

### Unrestricted Model
To calculate the memory capacity of the unrestricted model, run the two-step algorithm:

```
./parallel_synapse_two_step N P seed
```

where:
- `N` = number of input neurons
- `P` = number of patterns
- `seed` = random seed

Example:
```
./parallel_synapse_two_step 100 300 1
```
This runs with N=100 input neurons, P=300 patterns, and seed=1.

## Contact
For additional information or questions, please contact:  
**Email**: yus027@ucsd.edu
