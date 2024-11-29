# AlgoPRO - All-Pairs Minimax Path Algorithm Optimization

This repository contains the implementation and analysis of an optimized algorithm for computing all-pairs minimax path distances in undirected dense graphs, achieving O(n²log n) complexity.

## Repository Structure

- `Algopro_Comparison.pdf` - Performance comparison with existing algorithms
- `Algopro_Theorem.pdf` - Theoretical proofs and mathematical analysis
- `CMakeLists.txt` - CMake build configuration for C++ implementation
- `eval_algo4.ipynb` - Jupyter notebook for algorithm evaluation and benchmarking
- `gen_testcases.ipynb` - Test case generation for algorithm validation
- `opt_algopro.py` - Python implementation of the optimized algorithm
- `opt.cpp` - C++ implementation of the core algorithm
- `opt.hpp` - Header file with algorithm declarations
- `test_cases.py` - Test case utilities and framework
- `test_opt.cpp` - Unit tests for C++ implementation
- `visualize_algo4.ipynb` - Visualization of algorithm behavior and results

## Requirements

- C++17 or higher
- Python 3.8+
- Jupyter Notebook
- CMake 3.10+

## Building and Running

```bash
# Build C++ implementation
mkdir build && cd build
cmake ..
make

# Run tests
./test_opt

# Python implementation
python opt_algopro.py
```

## Documentation

Refer to `Algopro_Theorem.pdf` for detailed theoretical analysis and proofs. Performance benchmarks and comparisons can be found in `Algopro_Comparison.pdf`.
