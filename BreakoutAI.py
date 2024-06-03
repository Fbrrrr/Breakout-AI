import os
import pickle
import random
import math
import neat
import pyxel
import matplotlib.pyplot as plt

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 120
BALL_SPEED_INCREMENT = 0
BLOCK_BREAK_TIMEOUT = 30  # 30 seconds timeout for breaking blocks
FPS = 60

BALL_COLORS = [8, 9, 10, 11, 12, 13, 14]
MEDIUM_LIGHT_GREY = 13  # Pyxel color palette

class CustomGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.mutation_count = 0

    @classmethod
    def parse_config(cls, param_dict):
        return neat.DefaultGenome.parse_config(param_dict)

    @classmethod
    def write_config(cls, f, config):
        neat.DefaultGenome.write_config(f, config)

    def configure_new(self, config):
        super().configure_new(config)

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)

    def mutate(self, config):
        super().mutate(config)
        self.mutation_count += 1

    def distance(self, other, config):
        return super().distance(other, config)

    def size(self):
        return len(self.nodes), sum(1 for conn in self.connections.values() if conn.enabled)

class Ball:
    def __init__(self, x, y, color):
        self.r = 3
        self.x = x
        self.y = y - self.r - 1
        self.speed = 2
        self.a = 0.3 * math.pi  # 90 degrees, can be adjusted as needed
        self.vx = self.speed * math.cos(self.a)
        self.vy = -self.speed * math.sin(self.a)
        self.color = color

    def hits_paddle(self, paddle):
        if (self.y + self.r >= SCREEN_HEIGHT - paddle.h and
            paddle.x - paddle.w // 2 < self.x < paddle.x + paddle.w // 2):
            self.vy = -abs(self.vy)  # Ensure the ball bounces upwards
            overlap = self.x - paddle.x
            self.vx += overlap * 0.1  # Add some horizontal deflection based on overlap
            return True
        return False

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def on_screen(self):
        if self.x - self.r < 0 or self.x + self.r > SCREEN_WIDTH:
            self.vx = -self.vx
        if self.y - self.r < 0:
            self.vy = -self.vy

    def hits_ground(self):
        return self.y + self.r >= SCREEN_HEIGHT

class Paddle:
    def __init__(self, genome, config, mutation_count):
        self.h = 5
        self.w = 20
        self.x = SCREEN_WIDTH // 2
        self.dx = 2
        self.brain = neat.nn.FeedForwardNetwork.create(genome, config)
        self.genome = genome
        self.genome.fitness = 0
        self.mutation_count = mutation_count

    def update(self, ball):
        inputs = [
            self.x / SCREEN_WIDTH,
            ball.x / SCREEN_WIDTH,
            ball.y / SCREEN_HEIGHT,
            ball.vx / 10,
            ball.vy / 10
        ]
        output = self.brain.activate(inputs)
        direction = output[0]

        if direction > 0.5:
            self.x += self.dx
        elif direction < -0.5:
            self.x -= self.dx

        self.x = max(self.w // 2, min(self.x, SCREEN_WIDTH - self.w // 2))

class Block:
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.hit = False

    def hits(self, ball):
        if (self.x < ball.x < self.x + self.w and
            self.y < ball.y < self.y + self.h):
            self.hit = True
            ball.vy = -ball.vy

class GameState:
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.mutation_count = genome.mutation_count
        self.ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, random.choice(BALL_COLORS))
        self.paddle = Paddle(genome, config, self.mutation_count)
        self.blocks = self.create_blocks()
        self.timeout_counter = 0
        self.finished = False
        self.bounces = 0  # Track the number of bounces off the paddle
        self.paddle_distance = 0  # Track the distance moved by the paddle

    def create_blocks(self):
        block_width = SCREEN_WIDTH // 10
        block_height = 10
        blocks = []
        for i in range(10):
            for j in range(5):
                block = Block(i * block_width, j * block_height, block_width, block_height, 8)
                blocks.append(block)
        return blocks

    def update(self):
        if self.finished:
            return

        ball = self.ball
        paddle = self.paddle

        ball.update()
        ball.on_screen()
        if ball.hits_paddle(paddle):
            self.bounces += 1
        previous_paddle_x = paddle.x
        paddle.update(ball)
        self.paddle_distance += abs(paddle.x - previous_paddle_x)

        for block in self.blocks:
            block.hits(ball)

        self.blocks = [block for block in self.blocks if not block.hit]
        self.ball = ball
        self.paddle = paddle

        if ball.hits_ground() or self.timeout_counter >= BLOCK_BREAK_TIMEOUT * FPS:
            self.genome.fitness = (len(self.blocks) * 10) + (self.bounces * 5) - (self.paddle_distance * 0.1)
            self.finished = True
        else:
            self.timeout_counter += 1

class BreakoutVisualizer:
    def __init__(self, genomes, config, generation):
        self.genomes = genomes
        self.config = config
        self.states = [GameState(genome, config) for genome_id, genome in genomes]
        self.finished_count = 0
        self.generation = generation
        self.highest_fitness = 0
        self.average_fitness = 0
        self.calculate_fitness()

    def calculate_fitness(self):
        total_fitness = 0
        for state in self.states:
            if state.genome.fitness > self.highest_fitness:
                self.highest_fitness = state.genome.fitness
            total_fitness += state.genome.fitness
        self.average_fitness = total_fitness / len(self.states)

    def update(self):
        finished_states = [state for state in self.states if state.finished]
        self.finished_count = len(finished_states)
        for state in self.states:
            state.update()
        self.calculate_fitness()  # Recalculate fitness values on each update

    def draw(self):
        pyxel.cls(0)
        for state in self.states:
            if not state.finished:
                ball = state.ball
                paddle = state.paddle
                pyxel.circ(ball.x, ball.y, ball.r, ball.color)
                pyxel.rect(paddle.x - paddle.w // 2, SCREEN_HEIGHT - paddle.h, paddle.w, paddle.h, 7)
                for block in state.blocks:
                    pyxel.rect(block.x, block.y, block.w, block.h, block.color)
        self.draw_stats()

    def draw_stats(self):
        pyxel.text(SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 20, f'Generation: {self.generation}', MEDIUM_LIGHT_GREY)
        pyxel.text(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 - 10, f'Highest Fitness: {self.highest_fitness}', MEDIUM_LIGHT_GREY)
        pyxel.text(SCREEN_WIDTH // 2 - 45, SCREEN_HEIGHT // 2, f'Average Fitness: {self.average_fitness:.2f}', MEDIUM_LIGHT_GREY)

def eval_genomes(genomes, config, generation):
    visualizer = BreakoutVisualizer(genomes, config, generation)
    
    while visualizer.finished_count < len(visualizer.states):
        pyxel.cls(0)
        visualizer.update()
        visualizer.draw()
        pyxel.flip()

def run(config_file):
    config = neat.Config(CustomGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    def eval_genomes_with_generation(genomes, config):
        generation = p.generation
        eval_genomes(genomes, config, generation)

    winner = p.run(eval_genomes_with_generation, 300)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    pyxel.init(SCREEN_WIDTH, SCREEN_HEIGHT, title="Breakout AI", fps=FPS)
    run(config_path)
