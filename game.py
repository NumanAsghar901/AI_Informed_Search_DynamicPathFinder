import pygame
import sys
import math
import random
import time
from queue import PriorityQueue

# ================== SPEED CONTROLS ==================
SEARCH_DELAY = 20
AGENT_DELAY = 100

# ================== CONFIG ==================
WIDTH, HEIGHT = 900, 700
PANEL_WIDTH = 250
FPS = 60

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (200,200,200)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
ORANGE = (255,165,0)

START_COLOR = (0,100,0)   # Dark Green
GOAL_COLOR  = (255,0,0)   # Red

EMPTY, WALL = 0, 1

# ================== HEURISTICS ==================
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ================== GRID ==================
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self.start = (0, 0)
        self.goal = (rows-1, cols-1)

    def randomize(self, density):
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in [self.start, self.goal]:
                    self.grid[r][c] = WALL if random.random() < density else EMPTY

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != WALL:
                    yield (nr, nc)

# ================== SEARCH ==================
def reconstruct(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def search(grid, start, goal, algorithm, heuristic, draw_callback):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    g_cost = {start: 0}
    visited = set()
    nodes_visited = 0

    while not frontier.empty():
        _, current = frontier.get()
        nodes_visited += 1

        if current == goal:
            return reconstruct(came_from, goal), nodes_visited

        visited.add(current)

        for n in grid.neighbors(current):
            new_cost = g_cost[current] + 1
            if n not in g_cost or new_cost < g_cost[n]:
                g_cost[n] = new_cost
                priority = new_cost + heuristic(n, goal) if algorithm == "A*" else heuristic(n, goal)
                frontier.put((priority, n))
                came_from[n] = current

        draw_callback(visited, frontier.queue)
        pygame.time.delay(SEARCH_DELAY)

    return None, nodes_visited

# ================== APP ==================
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.rows = 20
        self.cols = 20
        self.cell = min((WIDTH-PANEL_WIDTH)//self.cols, HEIGHT//self.rows)

        self.grid = Grid(self.rows, self.cols)
        self.grid.randomize(0.3)

        self.algorithm = "A*"
        self.heuristic = manhattan
        self.dynamic = True
        self.spawn_prob = 0.05

        self.agent_pos = self.grid.start
        self.path = []
        self.path_cost = 0
        self.nodes_visited = 0
        self.exec_time = 0

        self.place_start = False
        self.place_goal = False

    def draw_grid(self, visited=None, frontier=None):
        self.screen.fill(WHITE)

        for r in range(self.rows):
            for c in range(self.cols):
                color = BLACK if self.grid.grid[r][c] == WALL else WHITE
                pygame.draw.rect(self.screen, color,
                    (c*self.cell, r*self.cell, self.cell, self.cell))
                pygame.draw.rect(self.screen, GRAY,
                    (c*self.cell, r*self.cell, self.cell, self.cell), 1)

        if visited:
            for v in visited:
                pygame.draw.rect(self.screen, BLUE,
                    (v[1]*self.cell, v[0]*self.cell, self.cell, self.cell))

        if frontier:
            for _, f in frontier:
                pygame.draw.rect(self.screen, YELLOW,
                    (f[1]*self.cell, f[0]*self.cell, self.cell, self.cell))

        for p in self.path:
            pygame.draw.rect(self.screen, GREEN,
                (p[1]*self.cell, p[0]*self.cell, self.cell, self.cell))

        pygame.draw.rect(self.screen, START_COLOR,
            (self.grid.start[1]*self.cell, self.grid.start[0]*self.cell, self.cell, self.cell))
        pygame.draw.rect(self.screen, GOAL_COLOR,
            (self.grid.goal[1]*self.cell, self.grid.goal[0]*self.cell, self.cell, self.cell))
        pygame.draw.rect(self.screen, ORANGE,
            (self.agent_pos[1]*self.cell, self.agent_pos[0]*self.cell, self.cell, self.cell))

        self.draw_panel()
        pygame.display.flip()

    def draw_panel(self):
        x, y = WIDTH - PANEL_WIDTH + 10, 20
        info = [
            f"Algorithm: {self.algorithm}",
            f"Heuristic: {'Manhattan' if self.heuristic==manhattan else 'Euclidean'}",
            f"Dynamic Mode: {self.dynamic}",
            f"Rows x Cols: {self.rows} x {self.cols}",
            f"Nodes Visited: {self.nodes_visited}",
            f"Path Cost: {self.path_cost}",
            f"Time (ms): {self.exec_time:.2f}",
            "",
            "Controls:",
            "A - A*",
            "G - GBFS",
            "H - Switch Heuristic",
            "R - Random Map",
            "S - Start Search",
            "D - Toggle Dynamic",
            "1 - Place START",
            "2 - Place GOAL",
            "ESC - Exit Place Mode",
            "LMB - Add Wall",
            "RMB - Remove Wall",
            "Arrow UP/DOWN - Increase/Decrease Rows",
            "Arrow LEFT/RIGHT - Increase/Decrease Columns"
        ]
        for line in info:
            self.screen.blit(self.font.render(line, True, BLACK), (x, y))
            y += 22

    def start_search(self):
        self.agent_pos = self.grid.start
        self.path = []
        self.nodes_visited = 0
        start_time = time.time()

        path, self.nodes_visited = search(
            self.grid,
            self.agent_pos,
            self.grid.goal,
            self.algorithm,
            self.heuristic,
            self.draw_grid
        )

        self.exec_time = (time.time() - start_time) * 1000
        if path:
            self.path = path.copy()
            self.path_cost = len(path)

    def move_agent(self):
        if not self.path:
            return

        next_step = self.path.pop(0)

        if self.dynamic and random.random() < self.spawn_prob:
            r, c = random.randint(0,self.rows-1), random.randint(0,self.cols-1)
            if (r,c) not in [self.agent_pos, self.grid.goal]:
                self.grid.grid[r][c] = WALL
                if (r,c) in self.path:
                    self.start_search()
                    return

        self.agent_pos = next_step
        pygame.time.delay(AGENT_DELAY)

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a: self.algorithm = "A*"
                    if event.key == pygame.K_g: self.algorithm = "GBFS"
                    if event.key == pygame.K_h:
                        self.heuristic = euclidean if self.heuristic == manhattan else manhattan
                    if event.key == pygame.K_r: self.grid.randomize(0.3)
                    if event.key == pygame.K_d: self.dynamic = not self.dynamic
                    if event.key == pygame.K_s: self.start_search()
                    if event.key == pygame.K_1:
                        self.place_start, self.place_goal = True, False
                    if event.key == pygame.K_2:
                        self.place_goal, self.place_start = True, False
                    if event.key == pygame.K_ESCAPE:
                        self.place_start = self.place_goal = False

                    # ==== NEW FEATURE: Adjust grid size ====
                    if event.key == pygame.K_UP:      # Increase rows
                        self.rows += 1
                        self.cell = min((WIDTH-PANEL_WIDTH)//self.cols, HEIGHT//self.rows)
                        self.grid = Grid(self.rows, self.cols)
                        self.grid.randomize(0.3)
                        self.agent_pos = self.grid.start
                        self.path = []

                    if event.key == pygame.K_DOWN:    # Decrease rows
                        if self.rows > 2:
                            self.rows -= 1
                            self.cell = min((WIDTH-PANEL_WIDTH)//self.cols, HEIGHT//self.rows)
                            self.grid = Grid(self.rows, self.cols)
                            self.grid.randomize(0.3)
                            self.agent_pos = self.grid.start
                            self.path = []

                    if event.key == pygame.K_RIGHT:   # Increase columns
                        self.cols += 1
                        self.cell = min((WIDTH-PANEL_WIDTH)//self.cols, HEIGHT//self.rows)
                        self.grid = Grid(self.rows, self.cols)
                        self.grid.randomize(0.3)
                        self.agent_pos = self.grid.start
                        self.path = []

                    if event.key == pygame.K_LEFT:    # Decrease columns
                        if self.cols > 2:
                            self.cols -= 1
                            self.cell = min((WIDTH-PANEL_WIDTH)//self.cols, HEIGHT//self.rows)
                            self.grid = Grid(self.rows, self.cols)
                            self.grid.randomize(0.3)
                            self.agent_pos = self.grid.start
                            self.path = []

            # ==== Mouse controls ====
            if pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                r, c = my//self.cell, mx//self.cell
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.place_start and (r,c) != self.grid.goal:
                        self.grid.start = (r,c)
                        self.agent_pos = (r,c)
                    elif self.place_goal and (r,c) != self.grid.start:
                        self.grid.goal = (r,c)
                    elif (r,c) not in [self.grid.start, self.grid.goal]:
                        self.grid.grid[r][c] = WALL

            if pygame.mouse.get_pressed()[2]:
                mx, my = pygame.mouse.get_pos()
                r, c = my//self.cell, mx//self.cell
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if (r,c) not in [self.grid.start, self.grid.goal]:
                        self.grid.grid[r][c] = EMPTY

            self.move_agent()
            self.draw_grid()

# ================== RUN ==================
if __name__ == "__main__":
    App().run()