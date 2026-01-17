# 6G Cognitive Radio Networks: Intelligent Spectrum Allocation Using a Hybrid Genetic–Reinforcement Learning Framework

## Problem Statement
The transition to 6G networks introduces unprecedented demand for frequency resources, yet traditional static and rule-based spectrum allocation remains incapable of adapting to rapidly changing wireless environments and unpredictable Primary User (PU) activity. This rigidity leads to severe spectrum underutilization, characterized by **spectral holes**, and harmful interference that degrades the Quality of Service (QoS) for Secondary Users (SUs) in dense, heterogeneous environments.

While existing learning-based approaches attempt to address these dynamics, they frequently suffer from slow convergence and the **cold-start problem**, failing to balance global optimization with real-time tactical adaptation. Consequently, there is a critical need for an intelligent, autonomous mechanism capable of continuous real-time adaptation to ensure efficient coexistence and maximized throughput in highly dynamic 6G Cognitive Radio Networks (CRNs).

---

## Solution
This project implements a **Hybrid GA-RL Framework** to enable autonomous **Dynamic Spectrum Access (DSA)**.

### Optimization Layer (Genetic Algorithm)
The Genetic Algorithm (GA) performs a global search to evolve optimal reward structures and hyperparameters. This "pre-tunes" the system, bypassing the traditional RL cold-start phase.

### Execution Layer (Reinforcement Learning)
A Proximal Policy Optimization (PPO) agent uses the GA-evolved parameters to perform real-time channel hopping.

### Data-Driven Validation
The system is trained on the **Spectrum Dataset**, ensuring the model learns from realistic PU occupancy patterns rather than simplified mathematical simulations.

The hybrid approach ensures the network can both:
- Strategize for long-term efficiency (via GA)
- React to millisecond-level environmental changes (via RL)

---

## Tech Stack
- **Languages:** Python (Primary development)  
- **ML Frameworks:** PyTorch or TensorFlow (PPO implementation), PyGAD or DEAP (Genetic Algorithm)  
- **Data Handling:** Pandas, NumPy, and Scikit-learn  
- **Dataset:** Spectrum Dataset (Real-world signal environments)  
- **Simulation Environment:** OpenAI Gym/Gymnasium (Custom environment for CRN)
