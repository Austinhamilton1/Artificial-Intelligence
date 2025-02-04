import pygame
import random
import copy
import math
from itertools import permutations
import numpy as np
import csv
import time

class Node:
    '''
    Node contains the information about a vertex in a graph
    '''
    def __init__(self, num, x, y, r):
        self.num = num
        #x and y position coordinates
        self.x = x
        self.y = y
        #radius for when we draw the Node
        self.radius = r
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return other.num == self.num
    def __hash__(self):
        return hash(self.num)
    def draw(self, screen):
        #draw the Node to the screen passed in
        pygame.draw.circle(screen, 'black', (self.x, self.y), self.radius)

class Graph:
    '''
    Graph is a representation of a mathematical graph G = (V, E)
    '''
    def __init__(self, nodes=None):
        if nodes == None:
            self.nodes = []
        else:
            self.nodes = copy.deepcopy(nodes)

    def set_edges(self):
        self.edges = list(permutations(self.nodes, 2))


    def random_cycle(self):
        '''
        random_cycle makes a random path among the nodes of the graph
        '''

        #we will choose a random node from the graph, and add it to the path until there are no
        #more nodes left
        path = []
        if len(self.nodes) > 1:
            current_list = copy.deepcopy(self.nodes)
            current_node = random.choice(current_list)
            current_list.remove(current_node)
            while len(current_list) > 0:
                new_node = random.choice(current_list)
                current_list.remove(new_node)
                path.append((current_node, new_node))
                current_node = new_node
            path.append((current_node, path[0][0]))
        return path

    def short_heuristic(self):
        path = []
        current_list = copy.deepcopy(self.nodes)
        current_node = random.choice(current_list)
        current_list.remove(current_node)
        while len(current_list) > 0:
            min_distance = math.inf
            new_node = None
            for possible in current_list:
                distance = self.distance((current_node, possible))
                if distance < min_distance:
                    new_node = possible
            current_list.remove(new_node)
            path.append((current_node, new_node))
            current_node = new_node
        path.append((current_node, path[0][0]))
        return path

    def density(self):
        e = len(self.edges)
        if e == 0:
            return 0
        sum = 0
        for edge in self.edges:
            sum += self.distance(edge)
        return sum / e
    
    @staticmethod
    def distance(edge):
        return math.sqrt((edge[0].x - edge[1].x)**2 + (edge[0].y - edge[1].y)**2)

    @staticmethod        
    def length(path):
        '''
        length loops through the nodes in the path and sums the distances between
        the nodes.
        returns -> float: length of the path
        '''
        length = 0
        for edge in path:
            length += Graph.distance(edge)
        return length
    
class ACOGraph(Graph):
    def __init__(self, nodes=None, init_pheromone=0.5):
        super(ACOGraph, self).__init__(nodes)
        super().set_edges()
        self.pheromones = {edge: init_pheromone for edge in self.edges}

class Solver:
    '''
    Solver is just a parent class for each of our solution algorithms
    Its holds data on the solution to make the code more concise and encapsulated
    '''
    def __init__(self, agent):
        self.agent = agent
        self.iter = iter(self.agent)
        self.min_distance = math.inf
        self.min_path = []
        self.solve_time = 0
        self.timer = 0

    def start(self):
        self.timer = time.perf_counter()

    def tick(self):
        return time.perf_counter() - self.timer

class BruteForce(Solver):
    '''
    BruteForce is a class that allows us to go through the permutations of a path
    It also stores the minimum path and distance of all the permutations
    '''
    def __init__(self, graph):
        self.num_nodes = len(graph.nodes)
        self.perms = permutations(graph.nodes)
        #setting min_distance at infinity ensures the algorithm will have a place to start
        super(BruteForce, self).__init__(self)

    def perm2path(self):
        current = next(self.perms)
        n = len(current)
        path = [(current[i], current[(i + 1) % n]) for i in range(n)]
        return path

    def __iter__(self):
        #we need to loop through from 0 to num_nodes! (factorial)
        self.n = 0
        self.max = math.perm(self.num_nodes)
        return self
    
    def __next__(self):
        if self.n <= self.max:
            path = copy.deepcopy(self.perm2path())
            distance = Graph.length(path)
            if distance < self.min_distance:
                self.min_distance = distance
                self.min_path = copy.deepcopy(path)
            self.solve_time = self.tick()
            return path
        raise StopIteration

    def __str__(self):
        return 'Brute Force'
    
class Annealing(Solver):
    '''
    The Annealing class lets us use simulated annealing to solve the problem
    '''
    def __init__(self, graph):
        self.current = graph.random_cycle()
        #after some experimentation, 50000 seemed to be a good number for the algorithm
        self.temp = 50000
        super(Annealing, self).__init__(self)

    def __iter__(self):
        #same solution is a counter that cuts the algorithm off after
        #it has generated the same solution multiple (500) times
        self.same_solution = 0
        return self
    
    def __next__(self):
        if self.same_solution < 500:
            successor = self.successor()
            #this is the cost difference between the new path and our current path
            # < 0 represents a better path, > 0 represents a worse path
            E = Graph.length(successor) - Graph.length(self.current)
            if E < 0:
                self.current = successor
                #reset the solution counter because a new solution has been chosen
                self.same_solution = 0
                self.solve_time = self.tick()
                distance = Graph.length(self.current)
                if distance < self.min_distance:
                    self.min_distance = distance
                    self.min_path = copy.deepcopy(self.current)
            #simulated annealing chooses a less optimal path with respect to temperature
            #the higher the temperature, the more likely it is to change
            elif random.uniform(0, 1) <= math.exp(-float(E) / float(self.temp)):
                self.current = successor
                self.same_solution = 0
                self.solve_time = self.tick()
            else:
                #this means we didn't choose a new solution so update the counter
                self.same_solution += 1
            self.cool()
            return self.current
        raise StopIteration
    
    def successor(self):
        '''
        successor chooses a new path to be the next path

        The successor is just a path with two random positions swapped
        '''
        i = random.randint(0, len(self.current) - 2)
        j = random.randint(0, len(self.current) - 2)

        while i == j:
            j = random.randint(0, len(self.current) - 2)

        i_value = self.current[i][0]
        j_value = self.current[j][0]
        if i_value == j_value:
            j_value = self.current[j][1]

        successor = []
        for edge in self.current:
            first = edge[0]
            second = edge[1]
            if first == i_value:
                first = j_value
            elif first == j_value:
                first = i_value
            if second == i_value:
                second = j_value
            elif second == j_value:
                second = i_value
            successor.append((first, second))

        return successor
    
    def cool(self):
        '''
        cool lowers the temperature function

        For this algorithm, I chose to use geometric reduction with a factor of 0.99
        '''
        self.temp *= 0.99

    def __str__(self):
        return 'Simulated Annealing'

class AntColony(Solver):
    def __init__(self, graph, num_ants, show_pheromones=False):
        self.graph = ACOGraph(graph.nodes, 0.0001)
        self.m = num_ants
        self.show_pheromones = show_pheromones
        super(AntColony, self).__init__(self)

    def evaporate(self):
        self.graph.pheromones = {edge: 0.95 * pheromone for edge, pheromone in self.graph.pheromones.items()}

    def update_pheromones(self):
        for ant in self.ants:
            tour = ant.get_tour()
            delta_t = 1 / Graph.length(tour)
            for edge in tour:
                self.graph.pheromones[edge] += delta_t

    def __iter__(self):
        self.n = 0
        self.max_iter = 500
        return self
    
    def __next__(self):
        if self.n < self.max_iter:
            self.ants = [Ant(self.graph, random.choice(self.graph.nodes)) for _ in range(self.m)]
            for ant in self.ants:
                ant.traverse()
                tour = ant.get_tour()
                distance = Graph.length(tour)
                if distance < self.min_distance:
                    self.min_distance = distance
                    self.min_path = copy.deepcopy(tour)
                    self.solve_time = self.tick()
            self.evaporate()
            self.update_pheromones()
            self.n += 1
            return self.min_path
        raise StopIteration
    
    def __str__(self):
        return "Ant Colony Optimization"

class Ant:
    def __init__(self, graph, start, alpha=1, beta=3):
        self.graph = graph
        self.start = start
        self.current = start
        self.alpha = alpha
        self.beta = beta
        self.tour = {self.current: None}
    
    def step(self):
        possible_edges = [e for e in self.graph.edges if e[0] == self.current and e[1] not in self.tour]
        possible_nodes = [e[1] for e in possible_edges]

        if len(possible_nodes) == 0:
            self.tour[self.current] = self.start
            return

        distances = np.array([Graph.distance(e) for e in possible_edges])
        pheromones = np.array([self.graph.pheromones[e] for e in possible_edges])

        preferences = pheromones**self.alpha / distances**self.beta
        probabilities = preferences / preferences.sum()

        new_node = np.random.choice(a=possible_nodes, size=1, p=probabilities)[0]
        self.tour[self.current] = new_node
        self.current = new_node

    def traverse(self):
        for _ in range(len(self.graph.nodes)):
            self.step()

    def get_tour(self):
        return [(k, v) for k, v in self.tour.items()]
    
class GeneticAlgorithm(Solver):
    def __init__(self, graph, pop_size):
        self.n = pop_size
        while self.n % 4 != 0:
            self.n += 1
        self.chromosome_length = len(graph.nodes)
        self.graph = graph
        self.m = 0.01
        self.population = [Individual(graph) for _ in range(self.n)]
        super(GeneticAlgorithm, self).__init__(self)

    def __iter__(self):
        self.max_gen = 700
        self.curr_gen = 0
        return self
    
    def __next__(self):
        if self.curr_gen < self.max_gen:
            new_gen = self.select()
            self.update_population(new_gen)
            for i in self.population:
                i.mutate(0.01)
            fittest = sorted(self.population, key=lambda i: i.fitness())[0]
            length = 1 / fittest.fitness()
            if length < self.min_distance:
                self.curr_gen = 0
                self.min_distance = length
                self.min_path = fittest.get_path()
                self.solve_time = self.tick()
            self.curr_gen += 1
            return self.min_path
        raise StopIteration
    
    def select(self):
        half = int(self.n / 2)
        return sorted(self.population, key=lambda individual: individual.fitness())[half:]

    def update_population(self, parents):
        gene_pool = copy.copy(parents)
        self.new_gen = copy.copy(gene_pool)
        while len(gene_pool) > 0:
            p1, p2 = random.sample(gene_pool, 2)
            gene_pool.remove(p1)
            gene_pool.remove(p2)
            self.breed(p1, p2)
        self.population = self.new_gen
        self.new_gen = []

    def breed(self, parent1, parent2):
        offspring1 = self.crossover(parent1, parent2)
        offspring2 = self.crossover(parent2, parent1)
        self.new_gen.append(offspring1)
        self.new_gen.append(offspring2)
        
    def crossover(self, individual1, individual2):
        parent1 = individual1.chromosome
        parent2 = individual2.chromosome
        start, end = sorted(random.sample(range(len(parent1)), 2))
        substring = parent1[start:end]

        offspring = [None] * self.chromosome_length
        offspring[start:end] = substring

        remaining_nodes = [node for node in parent2 if node not in offspring]
        
        j = 0
        for i in range(self.chromosome_length):
            if offspring[i] is None:
                offspring[i] = remaining_nodes[j]
                j += 1

        return Individual(self.graph, False, offspring)

    def __str__(self):
        return 'Genetic Algorithm'
    
class Individual:
    def __init__(self, graph, rand_init=True, chromosome=None):
        self.graph = graph
        if rand_init:
            self.chromosome = [edge[0] for edge in self.graph.random_cycle()]
        else:
            self.chromosome = chromosome

    def fitness(self):
        return 1 / Graph.length(self.get_path())
    
    def get_path(self):
        n = len(self.chromosome)
        path = [(self.chromosome[i], self.chromosome[(i+1) % n]) for i in range(n)]
        return path

    def mutate(self, m):
        for i in range(len(self.chromosome)):
            if random.uniform(0, 1) < m:
                j = random.randint(0, len(self.chromosome)-1)
                while(i == j):
                    j = random.randint(0, len(self.chromosome)-1)
                self.chromosome[i], self.chromosome[j] = self.chromosome[j], self.chromosome[i]

class Test:
    def __init__(self, screen, *args, log_file=None, test_func, log_fields):
        self.screen = screen
        self.graph = self.screen.graph
        self.agents = args
        self.i = 0
        self.log_file = log_file
        self.field_names = log_fields
        self.test_func = test_func
        self.data = []

    def run(self):
        done = False
        test_done = False
        self.screen.show_text(f'Nodes: {len(self.graph.nodes)}', 0)
        if self.i >= len(self.agents):
            self.screen.draw_graph(self.graph)
            self.screen.show_text('End of test.', 50)
            self.log()
            self.screen.show_text(f'Results added to {self.log_file}', 100)
            test_done = True
            done = True
            return test_done, done
        if isinstance(self.agents[self.i], AntColony):
            self.graph = self.agents[self.i].graph
        else:
            self.graph = self.screen.graph
        if self.agents[self.i].timer == 0:
            self.agents[self.i].start()
        self.screen.show_text(f'Current Min: {self.agents[self.i].min_distance}', 50)
        self.screen.show_text(f'Elapsed Time: {self.agents[self.i].tick()} s', 100)
        try:
            path = next(self.agents[self.i].iter)
            self.screen.draw_graph(self.graph, path)
            self.test_func()
        except StopIteration:
            self.screen.draw_graph(self.graph, self.agents[self.i].min_path)
            self.show()
            self.i += 1
            test_done = True
        return test_done, done
    
    def show(self):
        self.data.append({
            'Type': str(self.agents[self.i]),
            'Num Nodes': len(self.screen.graph.nodes),
            'Density': self.screen.graph.density(),
            'Time': self.agents[self.i].solve_time,
            'Performance': self.agents[self.i].min_distance
        })
        print(str(self))

    def log(self):
        if self.log_file is not None:
            self.data = sorted(self.data, key=lambda x: x['Type'])
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file, dialect='excel')
                writer.writerow(self.field_names)
                for mapping in self.data:
                    row = [mapping[key] for key in self.field_names]
                    writer.writerow(row)

    def __str__(self):
        message = f'''
        {'-' * 50}
        Type:\t\t{self.data[self.i]['Type']}
        Num Nodes:\t{self.data[self.i]['Num Nodes']}
        Density:\t{self.data[self.i]['Density']}
        Time:\t\t{self.data[self.i]['Time']} s
        Min Distance:\t{self.data[self.i]['Performance']}
        {'-' * 50}
        '''
        return message
    
class TimeTest(Test):
    def __init__(self, screen, *args, log_file=None):
        self.optimal_solution = 0
        field_names = ['Type', 'Density', 'Time', 'Num Nodes']
        super().__init__(screen, *args, log_file=log_file, test_func=self.tester, log_fields=field_names)

    def tester(self):
        if isinstance(self.agents[self.i], BruteForce):
            self.optimal_solution = self.agents[self.i].min_distance
        else:
            if self.agents[self.i].min_distance <= 1.05 * self.optimal_solution:
                raise StopIteration 
            
class PerformanceTest(Test):
    def __init__(self, screen, time, *args, log_file=None):
        self.time = time
        field_names = ['Type', 'Density', 'Performance', "Num Nodes"]
        super().__init__(screen, *args, log_file=log_file, test_func=self.tester, log_fields=field_names)
        self.test_func = self.tester

    def tester(self):
        if self.agents[self.i].tick() > self.time:
            raise StopIteration

class Screen:
    '''
    Screen is the representation of the pygame display object and some functions for it
    '''
    def __init__(self, width=500, height=500):
        #initialize the screen itself
        pygame.init()
        #the extra 50 is for text like shortest path, time, etc. to go
        self.screen = pygame.display.set_mode((width, height + 150))
        self.clock = pygame.time.Clock()
        #create the associated graph
        self.graph = Graph()
        #we'll use this to write to the window
        pygame.font.init()
        self.font = pygame.font.SysFont('Calibri', 30)
        self.draw_height = height
        self.draw_width = width
    def loop(self, test):
        self.tester = test
        while True:
            #poll for events caused by the user
            for event in pygame.event.get():
                #user presses the exit button
                if event.type == pygame.QUIT:
                    return
                #user presses the ESC key
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

            #fill the screen to "clear" it
            self.screen.fill('white')

            test_done, done = self.tester.run()
            
            #push the drawings from the buffer to the screen
            pygame.display.flip()

            if test_done:
                pygame.time.wait(5000)
            if done:
                return

            #limit the game to 60 FPS
            self.clock.tick(60)

    def populate_screen(self, num_nodes, min_distance=20):
        '''
        populate_screen fills a screen with num_nodes nodes ensuring they are min_distance apart
        screen -> pygame.display
        num_nodes -> int
        min_distance -> int
        '''

        #this would be close to the theoretical maximum
        #if the screen is a square/rectangle, the most efficient packing would be a lattice
        #there will be a max of sqrt(nodes) along the shortest side, so that's how big it could possibly be
        t_max = min(self.draw_width, self.draw_height) / math.sqrt(num_nodes)
        #0.75 is not a mathematically derived number, it's just a error factor that seemed to work
        #in all the cases I used
        if min_distance > 0.75 * t_max:
            raise ValueError(f'{min_distance} is too large for min_distance')
        
        self.graph.nodes = []
        
        current = 1

        #until we have generated enough nodes, keep making a random choice, looping through
        #the other nodes and checking to make sure it is at least min_distance away
        while len(self.graph.nodes) < num_nodes:
            x = random.randint(10, self.draw_width)
            y = random.randint(10, self.draw_height)
            is_valid = True
            for node in self.graph.nodes:
                distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                if distance < min_distance:
                    is_valid = False
                    break
            if is_valid:
                self.graph.nodes.append(Node(current, x, y, 3))
                current += 1
        self.graph.set_edges()

    def draw_graph(self, graph, path=None):
        '''
        draw_graph draws the nodes/edges of a graph to the screen
        graph -> Graph
        '''
        if len(graph.nodes) > 0:
            for node in graph.nodes:
                node.draw(self.screen)

        if path is None:
            return

        if isinstance(graph, ACOGraph) and self.tester.agents[self.tester.i].show_pheromones:
            self.show_pheromones(graph)

        if len(path) > 0:
            for edge in path:
                pygame.draw.line(self.screen, 'blue', (edge[0].x, edge[0].y), (edge[1].x, edge[1].y))

    def show_text(self, message, offset):
        '''
        show_text writes message to the pygame window
        message -> string
        offset -> how far from the bottom to write the message
        '''
        surface = self.font.render(message, False, 'black')
        self.screen.blit(surface, (0, self.draw_height + offset))

    def show_pheromones(self, graph):
        max_value = max(graph.pheromones.values())
        for edge in graph.pheromones.keys():
            surface = pygame.Surface((self.draw_width, self.draw_height), pygame.SRCALPHA)
            surface.set_alpha(int( graph.pheromones[edge] / max_value * 255))
            pygame.draw.line(surface, (255, 0, 0), (edge[0].x, edge[0].y), (edge[1].x, edge[1].y), 3)
            self.screen.blit(surface, (0, 0))

s = Screen()
s.populate_screen(50, 1)
test = PerformanceTest(s, 10, AntColony(s.graph, 50), AntColony(s.graph, 5), AntColony(s.graph, 10))
s.loop(test=test)
# for i in range(15):
#     i_s = str(i)
#     file = 'PerformanceTest_' + i_s + '.csv'
#     s.populate_screen(6, random.randint(10, 100))
#     test = TimeTest(s, BruteForce(s.graph), Annealing(s.graph), GeneticAlgorithm(s.graph, 10), AntColony(s.graph, 5), log_file=file)
#     s.loop(test=test) 

# for i in range(15):
#     i_s = str(i)
#     file1 = 'TimeTest50_' + i_s + '.csv'
#     file2 = 'TimeTest100_' + i_s + '.csv'
#     file3 = 'TimeTest500_' + i_s + '.csv'
#     s.populate_screen(50, random.randint(10, 30))
#     test = PerformanceTest(s, 30, Annealing(s.graph), GeneticAlgorithm(s.graph, 10), AntColony(s.graph, 5), log_file=file1)
#     s.loop(test=test)
#     s.populate_screen(100, random.randint(10, 20))
#     test = PerformanceTest(s, 60, Annealing(s.graph), GeneticAlgorithm(s.graph, 10), AntColony(s.graph, 5), log_file=file2)
#     s.loop(test=test)
#     s.populate_screen(500, 5)
#     test = PerformanceTest(s, 300, Annealing(s.graph), GeneticAlgorithm(s.graph, 10), AntColony(s.graph, 5), log_file=file3)
#     s.loop(test=test)