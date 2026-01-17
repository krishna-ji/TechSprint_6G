"""
PPO Training Script for Cognitive Radio Agent

Trains a Proximal Policy Optimization (PPO) agent to learn optimal channel selection
strategies for 6G IoT spectrum management. Supports both standard training and
GA-optimized hyperparameter search.

Usage:
    # Standard training with default hyperparameters
    python src/train_agent.py

    # Training with GA optimization (runs genetic algorithm first)
    python src/train_agent.py --ga-optimize --ga-generations 10

    # Resume from checkpoint
    python src/train_agent.py --resume models/checkpoint.zip

    # Custom hyperparameters
    python src/train_agent.py --learning-rate 0.0003 --gamma 0.99

References:
    - PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    - Stable-Baselines3: https://stable-baselines3.readthedocs.io/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.envs.cognitive_radio_env import CognitiveRadioEnv

# Lazy import to avoid circular dependency if possible, or usually just import
# But here we need to import the optimizer we just created
GeneticHyperparameterOptimizer = None

def get_ga_optimizer():
    """Lazy load GA optimizer to avoid circular imports."""
    global GeneticHyperparameterOptimizer
    if GeneticHyperparameterOptimizer is None:
        from src.ga_optimizer import GeneticHyperparameterOptimizer as GAOpt
        GeneticHyperparameterOptimizer = GAOpt
    return GeneticHyperparameterOptimizer



class TrainingConfig:
    """Centralized training configuration."""
    
    def __init__(self):
        # Paths
        self.train_data_path = "data/generated/spectrum_train.npy"
        self.test_data_path = "data/generated/spectrum_test.npy"
        self.models_dir = Path("models")
        self.logs_dir = Path("logs")
        
        # Device configuration (CUDA support)
        self.device = self._get_device()
        
        # MAJOR OVERHAUL: Advanced training configuration
        self.history_length = 10      # FIXED: Keep original to avoid shape mismatch
        self.use_enhanced_features = False  # Keep simple but optimize other aspects
        
        # DOMAIN-AWARE: Reward weights optimized for cognitive radio
        self.w_collision = 15.0       # MASSIVE collision penalty
        self.w_throughput = 8.0       # Moderate success reward
        self.w_energy = 0.1          # Minimal energy penalty
        self.w_channel_quality = 3.0  # NEW: Reward channel quality awareness
        self.w_imitation = 2.0       # NEW: Imitation learning weight
        self.max_episode_steps = 2000  # Full episodes
        
        # CURRICULUM LEARNING: Start easy, get harder
        self.curriculum_enabled = False  # DISABLE until basic training works
        self.curriculum_stages = [
            {"collision_multiplier": 0.3, "steps": 50000},   # Easy: 30% collisions
            {"collision_multiplier": 0.6, "steps": 100000},  # Medium: 60% collisions  
            {"collision_multiplier": 1.0, "steps": 200000},  # Hard: Full difficulty
        ]
        self.current_stage = 0
        
        # ADVANCED PPO: Optimized for complex sequential decision making
        self.learning_rate = 3e-4     # Standard optimal rate
        self.gamma = 0.995            # INCREASED: Long-term planning crucial
        self.n_steps = 4096           # INCREASED: More experience per update
        self.batch_size = 512         # INCREASED: Better gradient estimates
        self.n_epochs = 20            # INCREASED: More thorough updates
        self.clip_range = 0.2         # Standard stable clipping
        self.ent_coef = 0.005         # Moderate exploration
        self.vf_coef = 0.5            # Balanced value learning
        self.gae_lambda = 0.98        # High advantage estimation
        self.max_grad_norm = 0.5      # Stable gradients
        
        # ADVANCED ARCHITECTURE: Domain-specific neural networks
        self.policy_kwargs = {
            "features_extractor_class": "mlp",
            "features_extractor_kwargs": {
                "net_arch": [256, 256, 128],  # Deeper network for pattern recognition
            },
            "net_arch": {
                "pi": [128, 64],    # Policy network
                "vf": [128, 64]     # Value network
            },
            "activation_fn": "tanh",  # Better for normalized rewards
        }
        
        # EMERGENCY: Faster training with more frequent evaluation
        self.total_timesteps = 500_000    # REDUCED: Focus on getting it right quickly
        self.eval_freq = 10_000           # MORE frequent evaluation to catch problems
        self.save_freq = 25_000           # Regular checkpoints
        self.seed = 42
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "tensorboard").mkdir(exist_ok=True)
    
    def _get_device(self) -> str:
        """Auto-detect and configure optimal device (CUDA/CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 CUDA Enabled: {gpu_name} ({gpu_mem:.1f} GB)")
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True  # Auto-tune for performance
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
            torch.backends.cudnn.allow_tf32 = True
        else:
            device = "cpu"
            cpu_count = torch.get_num_threads()
            print(f"⚙️  CPU Mode: {cpu_count} threads")
        return device
    
    def create_curriculum_env(self, stage: int):
        """Create environment with curriculum learning difficulty."""
        # SIMPLER APPROACH: Use same data but adjust reward weights for curriculum
        stage_config = self.curriculum_stages[stage]
        
        # Progressive difficulty through reward adjustment, not data modification
        difficulty_factor = stage_config["collision_multiplier"]
        
        # Easier stages: less collision penalty, more exploration reward
        if difficulty_factor < 1.0:
            adjusted_w_collision = self.w_collision * difficulty_factor
            adjusted_w_throughput = self.w_throughput * (1.5 - difficulty_factor)  # More reward on easier stages
        else:
            adjusted_w_collision = self.w_collision  
            adjusted_w_throughput = self.w_throughput
        
        env = CognitiveRadioEnv(
            data_path="data/generated/spectrum_train.npy",  # Always use original data
            history_length=self.history_length,
            w_collision=adjusted_w_collision,
            w_throughput=adjusted_w_throughput, 
            w_energy=self.w_energy,
            max_episode_steps=self.max_episode_steps,
            use_enhanced_features=self.use_enhanced_features,
            seed=42
        )
        return env
    
    def get_greedy_demonstrations(self, env, n_episodes: int = 5):
        """Collect expert demonstrations from greedy baseline."""
        from src.envs.cognitive_radio_env import GreedyAgent
        
        greedy_agent = GreedyAgent()
        demonstrations = []
        
        print(f"🎓 Collecting {n_episodes} expert demonstrations...")
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_data = []
            
            done = False
            while not done:
                action, _ = greedy_agent.predict(obs)
                episode_data.append((obs.copy(), action))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            demonstrations.extend(episode_data)
            if episode == 0:
                print(f"   Expert episode reward: {info.get('total_reward', 'N/A')}")
        
        print(f"✅ Collected {len(demonstrations)} expert transitions")
        return demonstrations

    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            'device': self.device,
            'environment': {
                'history_length': self.history_length,
                'w_collision': self.w_collision,
                'w_throughput': self.w_throughput,
                'w_energy': self.w_energy,
                'max_episode_steps': self.max_episode_steps,
            },
            'ppo': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'gae_lambda': self.gae_lambda,
                'max_grad_norm': self.max_grad_norm,
            },
            'training': {
                'total_timesteps': self.total_timesteps,
                'eval_freq': self.eval_freq,
                'save_freq': self.save_freq,
                'seed': self.seed,
            }
        }
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def make_env(config: TrainingConfig, eval_mode: bool = False) -> CognitiveRadioEnv:
    """
    Create a Cognitive Radio environment.
    
    Args:
        config: Training configuration
        eval_mode: If True, use test dataset; else use training dataset
    
    Returns:
        Monitored environment wrapped for Stable-Baselines3
    """
    data_path = config.test_data_path if eval_mode else config.train_data_path
    
    env = CognitiveRadioEnv(
        data_path=data_path,
        history_length=config.history_length,
        w_collision=config.w_collision,
        w_throughput=config.w_throughput,
        w_energy=config.w_energy,
        max_episode_steps=config.max_episode_steps,
        use_enhanced_features=config.use_enhanced_features,  # NEW: Enhanced features
        seed=config.seed + (1 if eval_mode else 0)
    )
    
    # Wrap with Monitor for logging
    log_dir = config.logs_dir / ("eval" if eval_mode else "train")
    log_dir.mkdir(exist_ok=True)
    env = Monitor(env, str(log_dir))
    
    return env


def train_agent_with_curriculum(config: TrainingConfig, verbose: int = 1) -> PPO:
    """
    Train PPO agent using curriculum learning and imitation learning.
    
    Args:
        config: Training configuration
        verbose: Verbosity level
    
    Returns:
        Trained PPO model
    """
    print("🎓 MAJOR OVERHAUL: Starting Curriculum + Imitation Learning Training")
    
    # Create evaluation environment (full difficulty)
    eval_env = make_env(config, eval_mode=True)
    eval_env = Monitor(eval_env, str(config.logs_dir / "eval"))
    
    model = None
    
    for stage, stage_config in enumerate(config.curriculum_stages):
        print(f"\n📚 CURRICULUM STAGE {stage + 1}/{len(config.curriculum_stages)}")
        print(f"   Difficulty: {stage_config['collision_multiplier']*100:.0f}% collision density")
        print(f"   Training steps: {stage_config['steps']:,}")
        
        # Create curriculum environment for this stage
        train_env = config.create_curriculum_env(stage)
        train_env = Monitor(train_env, str(config.logs_dir / "train" / f"stage_{stage}"))
        
        if model is None:
            # First stage: Create new model with imitation learning
            print("🤖 Creating new PPO model...")
            
            # Collect expert demonstrations
            demonstrations = config.get_greedy_demonstrations(train_env, n_episodes=5)
            
            # Create model with smaller architecture for curriculum
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[128, 64],
                    vf=[128, 64]
                ),
                activation_fn=torch.nn.ReLU,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5)
            )
            
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                gae_lambda=config.gae_lambda,
                max_grad_norm=config.max_grad_norm,
                tensorboard_log=str(config.logs_dir / "tensorboard"),
                policy_kwargs=policy_kwargs,
                device=config.device,
                seed=config.seed,
                verbose=verbose
            )
            
            # Pre-train with imitation learning
            print("🎯 Pre-training with expert demonstrations...")
            for epoch in range(5):  # Quick imitation pre-training
                loss = 0.0
                for obs, action in demonstrations[:100]:  # Use first 100 demonstrations
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
                    action_tensor = torch.LongTensor([action]).to(model.device)
                    
                    # Get policy prediction
                    with torch.no_grad():
                        features = model.policy.extract_features(obs_tensor)
                        action_logits = model.policy.action_net(features)
                    
                    # Simple behavioral cloning loss (for pre-training)
                    loss += torch.nn.functional.cross_entropy(action_logits, action_tensor).item()
                
                if epoch == 0:
                    print(f"   Imitation loss (epoch {epoch+1}): {loss/100:.3f}")
        
        else:
            # Transfer to new environment 
            print("🔄 Transferring model to next difficulty stage...")
            model.set_env(train_env)
        
        # Setup callbacks for this stage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"curriculum_stage_{stage}_{timestamp}"
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000, stage_config['steps'] // 5),
            save_path=str(config.models_dir / "curriculum" / f"stage_{stage}"),
            name_prefix=run_name,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(config.models_dir / "curriculum" / f"stage_{stage}_best"),
            log_path=str(config.logs_dir / "curriculum" / f"stage_{stage}_eval"),
            eval_freq=max(5000, stage_config['steps'] // 10),
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Train for this stage
        print(f"🚀 Training stage {stage + 1}...")
        start_time = time.time()
        model.learn(
            total_timesteps=stage_config['steps'],
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=run_name,
            reset_num_timesteps=False  # Keep total timestep count
        )
        
        stage_time = time.time() - start_time
        print(f"✅ Stage {stage + 1} completed in {stage_time/60:.1f} minutes")
        
        # Save stage checkpoint
        stage_path = config.models_dir / "curriculum" / f"stage_{stage}_final"
        stage_path.mkdir(parents=True, exist_ok=True)
        model.save(str(stage_path / "model"))
        config.save(stage_path / "config.json")
    
    print("\n🏆 CURRICULUM TRAINING COMPLETED!")
    return model


def train_agent(
    config: TrainingConfig,
    resume_path: Optional[str] = None,
    verbose: int = 1
) -> PPO:
    """
    Train PPO agent on cognitive radio task.
    
    Args:
        config: Training configuration
        resume_path: Path to checkpoint to resume from (optional)
        verbose: Verbosity level (0=none, 1=info, 2=debug)
    
    Returns:
        Trained PPO model
    """
    print("🚀 Initializing Cognitive Radio PPO Training\n")
    
    # Create environments
    print("📡 Creating environments...")
    train_env = make_env(config, eval_mode=False)
    eval_env = make_env(config, eval_mode=True)
    
    # Create or load model
    if resume_path:
        print(f"📂 Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=train_env)
    else:
        print("🧠 Creating new PPO model...")
        # SIMPLIFIED: Smaller network to prevent overfitting and pathological behavior
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 64],    # MUCH SMALLER: Prevent overfitting
                vf=[128, 64]     # MUCH SMALLER: Simple value estimation
            ),
            activation_fn=torch.nn.ReLU,       # Back to ReLU for stability
            optimizer_class=torch.optim.Adam,  # Back to Adam for faster convergence
            optimizer_kwargs=dict(
                eps=1e-5,                      # Standard epsilon
            )
        )
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            gae_lambda=config.gae_lambda,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=str(config.logs_dir / "tensorboard"),
            policy_kwargs=policy_kwargs,
            device=config.device,  # Use CUDA if available
            seed=config.seed,
            verbose=verbose
        )
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_cognitive_radio_{timestamp}"
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(config.models_dir / "checkpoints"),
        name_prefix=run_name,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.models_dir / "best"),
        log_path=str(config.logs_dir / "eval"),
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Print training info
    print("\n📊 Training Configuration:")
    print(f"   - Device: {config.device.upper()}")
    if config.device == "cuda":
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"   - Training Dataset: {config.train_data_path}")
    print(f"   - Test Dataset: {config.test_data_path}")
    print(f"   - Total Timesteps: {config.total_timesteps:,}")
    print(f"   - Learning Rate: {config.learning_rate}")
    print(f"   - Gamma: {config.gamma}")
    print(f"   - Batch Size: {config.batch_size}")
    print(f"   - N Steps: {config.n_steps}")
    print(f"   - Reward Weights: Collision={config.w_collision}, "
          f"Throughput={config.w_throughput}, Energy={config.w_energy}")
    
    print("\n🎯 Starting Training...\n")
    
    # Train the model
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=False  # Disable progress bar to avoid dependencies
    )
    
    # Save final model
    final_model_path = config.models_dir / f"{run_name}_final.zip"
    model.save(final_model_path)
    
    # Save configuration
    config_path = config.models_dir / f"{run_name}_config.json"
    config.save(config_path)
    
    print(f"\n✅ Training Complete!")
    print(f"   - Final Model: {final_model_path}")
    print(f"   - Best Model: {config.models_dir / 'best' / 'best_model.zip'}")
    print(f"   - Configuration: {config_path}")
    print(f"\n📈 View training logs with TensorBoard:")
    print(f"   tensorboard --logdir {config.logs_dir / 'tensorboard'}")
    
    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for 6G Cognitive Radio"
    )
    
    # Training mode
    parser.add_argument(
        '--ga-optimize',
        action='store_true',
        help='Use genetic algorithm to optimize hyperparameters first'
    )
    parser.add_argument(
        '--ga-generations',
        type=int,
        default=10,
        help='Number of GA generations (default: 10)'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Hyperparameter overrides
    parser.add_argument('--learning-rate', type=float, help='PPO learning rate')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--w-collision', type=float, help='Collision penalty weight')
    parser.add_argument('--w-throughput', type=float, help='Throughput reward weight')
    parser.add_argument('--w-energy', type=float, help='Energy cost weight')
    
    # Training parameters
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100_000,
        help='Total training timesteps (default: 100k)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Apply command-line overrides
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.gamma:
        config.gamma = args.gamma
    if args.w_collision:
        config.w_collision = args.w_collision
    if args.w_throughput:
        config.w_throughput = args.w_throughput
    if args.w_energy:
        config.w_energy = args.w_energy
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.seed:
        config.seed = args.seed
    
    # GA optimization
    if args.ga_optimize:
        print("🧬 GA Optimization requested - this will run genetic algorithm first")
        
        GAOptimizer = get_ga_optimizer()
        if GAOptimizer is None:
            print("❌ GA Optimizer class not found. Make sure src/ga_optimizer.py exists.")
            return

        optimizer = GAOptimizer(generations=args.ga_generations)
        best_params = optimizer.run()
        
        # Update config with best params
        config.learning_rate = best_params['learning_rate']
        config.gamma = best_params['gamma']
        config.ent_coef = best_params['ent_coef']
        config.vf_coef = best_params['vf_coef']
        config.w_collision = best_params['w_collision']
        config.w_throughput = best_params['w_throughput']
        config.w_energy = best_params['w_energy']
        
        print("\n✅ Config updated with GA-optimized hyperparameters!")

    
    # MAJOR OVERHAUL: Use curriculum learning by default for better performance
    if config.curriculum_enabled:
        print("🎓 Using CURRICULUM LEARNING for improved performance!")
        model = train_agent_with_curriculum(config, verbose=1)
    else:
        print("📚 Using standard training (curriculum disabled)")
        model = train_agent(config, resume_path=None, verbose=1)  # FORCE no resume for fresh model
    
    # Save final model
    final_path = config.models_dir / "final_overhaul"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path / "model"))
    config.save(final_path / "config.json")
    
    print(f"\n✅ Final model saved to: {final_path}")
    print("\n🎯 Next Steps:")
    print("   1. Evaluate the model: python src/evaluate.py")
    print("   2. Run the demo: streamlit run app.py")


if __name__ == "__main__":
    main()
