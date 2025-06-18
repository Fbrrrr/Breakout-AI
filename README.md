# Breakout AI using NEAT

A sophisticated implementation of the classic Breakout game where AI agents learn to play using NEAT (NeuroEvolution of Augmenting Topologies). This project demonstrates the power of evolutionary algorithms in game AI development.

## 🎮 Features

- **Evolutionary AI**: Uses NEAT algorithm to evolve neural networks that control the paddle
- **Real-time Visualization**: Watch AI agents learn and improve over generations
- **Advanced Physics**: Realistic ball physics with proper collision detection
- **Performance Metrics**: Comprehensive fitness tracking and statistics
- **Modular Architecture**: Clean, well-documented code structure
- **Configurable Parameters**: Easy to adjust game and evolution settings

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Fbrrrr/breakout-ai.git
   cd breakout-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training:**
   ```bash
   python BreakoutAI.py
   ```

## 📋 Requirements

- `neat-python==0.92` - NEAT algorithm implementation
- `pyxel==1.9.18` - Retro game engine for visualization
- `matplotlib==3.7.2` - Plotting and visualization
- `numpy==1.24.3` - Numerical computations

## 🎯 How It Works

### Game Mechanics

- **Objective**: AI-controlled paddle must keep the ball in play and destroy all blocks
- **Fitness Function**: Rewards block destruction, ball survival, and efficient movement
- **Timeout**: 30-second limit per game to prevent infinite loops

### Neural Network Architecture

- **Inputs (5)**: 
  - Paddle position (normalized)
  - Ball position X, Y (normalized)
  - Ball velocity X, Y (normalized)
- **Outputs (1)**: 
  - Paddle movement decision (-1: left, 0: stay, 1: right)

### Evolution Process

1. **Population**: 50 AI agents per generation
2. **Selection**: Best performers reproduce
3. **Mutation**: Network topology and weights evolve
4. **Generations**: Up to 300 generations (configurable)

## 🛠️ Configuration

### Game Settings

Edit the `GameConfig` class in `BreakoutAI.py`:

```python
@dataclass(frozen=True)
class GameConfig:
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 120
    BALL_SPEED: float = 2.0
    PADDLE_WIDTH: int = 20
    TIMEOUT_SECONDS: int = 30
    # ... more settings
```

### NEAT Parameters

Modify `config-feedforward.txt` to adjust evolution parameters:

```ini
[NEAT]
pop_size = 50
fitness_threshold = 1000.0

[DefaultGenome]
# Network structure
num_inputs = 5
num_outputs = 1
# Mutation rates
weight_mutate_rate = 0.8
conn_add_prob = 0.5
# ... more parameters
```

## 📊 Performance Metrics

The AI agents are evaluated using a comprehensive fitness function:

- **Block Destruction**: 100 points per block destroyed
- **Ball Survival**: 10 points per paddle hit
- **Completion Bonus**: 500 points for clearing all blocks
- **Time Bonus**: Faster completion yields higher scores
- **Movement Penalty**: Small penalty for excessive paddle movement

## 🔧 Code Structure

```
BreakoutAI.py
├── GameConfig          # Game constants and settings
├── Vector2D           # 2D vector mathematics
├── Ball               # Ball physics and collision
├── Paddle             # AI-controlled paddle
├── Block              # Destructible blocks
├── Game               # Individual game instance
├── GameVisualizer     # Multi-game visualization
└── NEAT Integration   # Evolution and training
```

## 📈 Monitoring Training

During training, you'll see:

- **Generation**: Current evolution cycle
- **Max Fitness**: Best performing agent's score
- **Avg Fitness**: Population average performance
- **Progress**: Number of completed games

## 🎮 Advanced Usage

### Loading Trained Models

```python
import pickle
import neat

# Load the trained winner
with open("winner.pkl", "rb") as f:
    winner = pickle.load(f)

# Create network and play
config = neat.Config(neat.DefaultGenome, ...)
net = neat.nn.FeedForwardNetwork.create(winner, config)
```

### Custom Fitness Functions

Modify the `_calculate_fitness()` method in the `Game` class:

```python
def _calculate_fitness(self) -> None:
    # Your custom fitness calculation
    self.genome.fitness = your_fitness_formula()
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Config File Not Found**: Check that `config-feedforward.txt` exists in the project directory

3. **Slow Training**: Reduce population size or increase timeout for faster iterations

4. **Poor AI Performance**: Adjust mutation rates or network structure in config file

### Performance Tips

- **Faster Training**: Reduce `pop_size` in config for quicker generations
- **Better AI**: Increase `max_stagnation` to allow more exploration
- **Stability**: Lower mutation rates for more stable evolution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Learning Resources

- [NEAT Algorithm](https://neat-python.readthedocs.io/) - Original NEAT documentation
- [Evolutionary Algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm) - General EA concepts
- [Game AI Programming](https://gameai.com/) - Game AI techniques

## 🎨 Visualization

The training process shows:
- Real-time gameplay of the best performing agent
- Generation statistics and progress
- Fitness evolution over time
- Population diversity metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NEAT-Python**: Excellent implementation of the NEAT algorithm
- **Pyxel**: Fantastic retro game engine for Python
- **Community**: Thanks to all contributors and users

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Search existing [issues](https://github.com/Fbrrrr/breakout-ai/issues)
3. Create a new issue with detailed information

---

**Happy AI Training! 🤖🎮**
