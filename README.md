# Breakout AI using NEAT

This project implements a Breakout game AI using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI controls the paddle in the Breakout game, trying to maximize its fitness by breaking as many blocks as possible.

## Requirements

- Python 3.x
- Pyxel
- NEAT-Python
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/breakout-ai.git
    cd breakout-ai
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the `BreakoutAI.py` script:
    ```bash
    python BreakoutAI.py
    ```

2. The AI will start training, and you will see the current generation, highest fitness, and average fitness displayed on the screen.

## File Descriptions

- `BreakoutAI.py`: The main script that runs the Breakout game and trains the AI using NEAT.
- `config-feedforward.txt`: Configuration file for NEAT.
- `requirements.txt`: Lists the required Python packages.

## Screenshots

![Screenshot](screenshot.png)

## Acknowledgements

- [NEAT-Python](https://neat-python.readthedocs.io/en/latest/) - Library for NEAT (NeuroEvolution of Augmenting Topologies) in Python.
- [Pyxel](https://github.com/kitao/pyxel) - A retro game engine for Python.

## License

This project is licensed under the MIT License.
