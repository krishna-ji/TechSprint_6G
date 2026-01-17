"""
Genetic Algorithm Optimizer for PPO Hyperparameters in Cognitive Radio

Uses PyGAD to evolve optimal hyperparameters for the PPO agent, including:
- Learning rate, discount factor (gamma)
- Entropy coefficient, value function coefficient
- Reward weights for collision, throughput, and energy

Usage:
    python src/ga_optimizer.py --generations 10 --population 20
"""

import pygad
import numpy as np
import json
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.envs.cognitive_radio_env import CognitiveRadioEnv


class TrainingConfigForGA:
    """Lightweight config class for GA fitness evaluation."""
    
    def __init__(self):
        self.train_data_path = "data/generated/spectrum_train.npy"
        self.test_data_path = "data/generated/spectrum_test.npy"
        self.history_length = 10
        self.max_episode_steps = 2000  # Shorter episodes for GA
        self.seed = 42
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class GeneticHyperparameterOptimizer:
    """
    Genetic Algorithm optimizer for PPO hyperparameters.
    
    Evolves a population of hyperparameter configurations and selects
    the best performers based on evaluation reward.
    """
    
    def __init__(self, generations: int = 10, population_size: int = 10):
        self.generations = generations
        self.population_size = population_size
        self.best_solution = None
        self.best_fitness = -np.inf
        
        # Enhanced gene space for comprehensive optimization
        self.gene_space = [
            {'low': 1e-5, 'high': 1e-2},  # learning_rate (expanded range)
            {'low': 0.95, 'high': 0.999}, # gamma (longer term planning)
            {'low': 0.001, 'high': 0.05}, # ent_coef (balanced exploration)
            {'low': 0.1, 'high': 1.0},    # vf_coef
            {'low': 1.0, 'high': 10.0},   # w_collision (penalty weight)
            {'low': 5.0, 'high': 15.0},   # w_throughput (success reward)
            {'low': 0.05, 'high': 0.5},   # w_energy (switching cost)
            {'low': 5, 'high': 20},       # history_length (context window)
        ]
        
        self.gene_labels = [
            'learning_rate', 'gamma', 'ent_coef', 'vf_coef', 
            'w_collision', 'w_throughput', 'w_energy', 'history_length'
        ]
        
        self.config = TrainingConfigForGA()
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        """Evaluate fitness of a hyperparameter configuration."""
        # Decode enhanced solution
        lr, gamma, ent_coef, vf_coef, w_col, w_thr, w_en, hist_len = solution
        hist_len = int(hist_len)  # Ensure integer for history length
        
        try:
            # Create enhanced environment with these parameters
            env = CognitiveRadioEnv(
                data_path=self.config.train_data_path,
                history_length=hist_len,
                w_collision=w_col,
                w_throughput=w_thr,
                w_energy=w_en,
                max_episode_steps=self.config.max_episode_steps,
                seed=self.config.seed,
                use_enhanced_features=True  # Enable enhanced observations
            )
            
            # Train model for short duration
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr,
                gamma=gamma,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                n_steps=512,
                batch_size=64,
                device=self.config.device,  # Use CUDA if available
                verbose=0
            )
            model.learn(total_timesteps=3000)
            
            # Evaluate on collision rate (our true objective)
            obs, _ = env.reset()
            collisions = 0
            successes = 0
            
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            
            # Fitness = negative collision rate (we want to minimize collisions)
            collision_rate = info.get('collision_rate', 1.0)
            fitness = -collision_rate * 100  # Scale for GA
            
            env.close()
            
            print(f"  Solution {solution_idx}: LR={lr:.6f}, γ={gamma:.3f}, "
                  f"Collision={collision_rate:.2%}, Fitness={fitness:.2f}")
            
            return fitness
            
        except Exception as e:
            print(f"  GA Error (solution {solution_idx}): {e}")
            return -100.0

    def run(self):
        """Run the genetic algorithm optimization."""
        print("🧬 Starting Genetic Algorithm Optimization...")
        print(f"   Device: {self.config.device.upper()}")
        print(f"   Generations: {self.generations}")
        print(f"   Population: {self.population_size}")
        print(f"   Genes: learning_rate, gamma, ent_coef, vf_coef, w_collision, w_throughput, w_energy\n")
        
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=4,
            fitness_func=self.fitness_func,
            sol_per_pop=self.population_size,
            num_genes=len(self.gene_space),
            gene_space=self.gene_space,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=15,
            suppress_warnings=True
        )
        
        ga_instance.run()
        
        solution, solution_fitness, _ = ga_instance.best_solution()
        self.best_solution = solution
        self.best_fitness = solution_fitness
        
        print(f"\n✅ Best Solution Found!")
        print(f"   Fitness (negative collision %): {solution_fitness:.2f}")
        
        # Save best parameters
        best_params = {
            'learning_rate': float(solution[0]),
            'gamma': float(solution[1]),
            'ent_coef': float(solution[2]),
            'vf_coef': float(solution[3]),
            'w_collision': float(solution[4]),
            'w_throughput': float(solution[5]),
            'w_energy': float(solution[6]),
            'fitness': float(solution_fitness)
        }
        
        best_params_path = Path("models/best_params.json")
        best_params_path.parent.mkdir(exist_ok=True)
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
            
        print(f"💾 Saved best parameters to {best_params_path}")
        print(f"\n📊 Best Hyperparameters:")
        for key, value in best_params.items():
            if key != 'fitness':
                print(f"   {key}: {value:.6f}" if isinstance(value, float) else f"   {key}: {value}")
        
        return best_params


def main():
    """Run GA optimization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize PPO hyperparameters with GA")
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--population', type=int, default=10, help='Population size')
    args = parser.parse_args()
    
    optimizer = GeneticHyperparameterOptimizer(
        generations=args.generations,
        population_size=args.population
    )
    optimizer.run()


if __name__ == "__main__":
    main()
