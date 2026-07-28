# K-ARC: Kinodynamic Adaptive Robot Coordination (Python Prototype)

This module provides a standalone Python proof-of-concept implementation of **Algorithm 1 (K-ARC Framework)** and **Algorithm 2 (SolveSubProblem)** from the paper *"K-ARC: Adaptive Robot Coordination for Multi-Robot Kinodynamic Planning"*[cite: 2].

The implementation uses **CasADi** and the **IPOPT** non-linear solver to execute segment-wise trajectory optimization, dynamic conflict detection, and prioritized local conflict resolution.

---

## 🛠️ Requirements & Setup

### Python Dependencies
Install the required scientific computing and optimization packages:

```bash
pip install numpy casadi matplotlib
