import pygame
import random
import copy
import math
from itertools import permutations

class Node:
    '''
    Node contains the information about a vertex in a graph
    '''
    def __init__(self, x, y, r):
        #x and y position coordinates
        self.x = x
        self.y = y
        #radius for when we draw the Node
        self.radius = r
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

    def random_cycle(self):
        '''
        random_cycle makes a random path among the nodes of the graph
        '''

        #we will choose a random node from the graph, and add it to the path until there are no
        #more nodes left
        path = []
        if len(self.nodes) > 1:
            current_list = copy.deepcopy(self.nodes)
            while len(current_list) > 0:
                node = random.choice(current_list)
                current_list.remove(node)
                path.append(node)
        return path

    @staticmethod        
    def length(path):
        '''
        length loops through the nodes in the path and sums the distances between
        the nodes.
        returns -> float: length of the path
        '''
        length = 0
        for i in range(len(path)):
            start = path[i]
            #the mod makes sure the last node is connected to the first
            end = path[(i + 1) % len(path)]
            distance = math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)
            length += distance
        return length

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
        self.time = 0

class BruteForce(Solver):
    '''
    BruteForce is a class that allows us to go through the permutations of a path
    It also stores the minimum path and distance of all the permutations
    '''
    def __init__(self, graph):
        self.num_nodes = len(graph.nodes)
        self.current_path = permutations(graph.nodes)
        #setting min_distance at infinity ensures the algorithm will have a place to start
        super(BruteForce, self).__init__(self)

    def __iter__(self):
        #we need to loop through from 0 to num_nodes! (factorial)
        self.n = 0
        self.max = math.perm(self.num_nodes)
        return self
    
    def __next__(self):
        if self.n <= self.max:
            path = copy.deepcopy(next(self.current_path))
            distance = Graph.length(path)
            if distance < self.min_distance:
                self.min_distance = distance
                self.min_path = copy.deepcopy(path)
            return path
        raise StopIteration
    
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
                distance = Graph.length(self.current)
                if distance < self.min_distance:
                    self.min_distance = distance
                    self.min_path = copy.deepcopy(self.current)
            #simulated annealing chooses a less optimal path with respect to temperature
            #the higher the temperature, the more likely it is to change
            elif random.uniform(0, 1) <= math.exp(-float(E) / float(self.temp)):
                self.current = successor
                self.same_solution = 0
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
        i = random.randint(0, len(self.current) - 1)
        j = random.randint(0, len(self.current) - 1)

        while i == j:
            j = random.randint(0, len(self.current) - 1)

        successor = copy.deepcopy(self.current)
        tmp = successor[i]
        successor[i] = successor[j]
        successor[j] = tmp

        return successor
    
    def cool(self):
        '''
        cool lowers the temperature function

        For this algorithm, I chose to use geometric reduction with a factor of 0.99
        '''
        self.temp *= 0.99

class Screen:
    '''
    Screen is the representation of the pygame display object and some functions for it
    '''
    def __init__(self, width=500, height=500):
        #initialize the screen itself
        pygame.init()
        #the extra 50 is for text like shortest path, time, etc. to go
        self.screen = pygame.display.set_mode((width, height + 100))
        self.clock = pygame.time.Clock()
        #create the associated graph
        self.graph = Graph()
        #we'll use this to write to the window
        pygame.font.init()
        self.font = pygame.font.SysFont('Calibri', 30)
    def loop(self, bruteforce=False, annealing=False):
        if bruteforce:
            #create a BruteForce object to solve for the optimal path
            self.solver = BruteForce(self.graph)
        elif annealing:
            #create an Annealing object to approximate the optimal path
            self.solver = Annealing(self.graph)
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

            self.solve()
            
            #push the drawings from the buffer to the screen
            pygame.display.flip()

            #limit the game to 60 FPS
            self.clock.tick(60)

    def populate_screen(self, num_nodes, min_distance=20):
        '''
        populate_screen fills a screen with num_nodes nodes ensuring they are min_distance apart
        screen -> pygame.display
        num_nodes -> int
        min_distance -> int
        '''

        #we need a way of bailing out of the function if the user enters a value too high for min_distance
        size = pygame.display.get_window_size()
        #this would be close to the theoretical maximum
        #if the screen is a square/rectangle, the most efficient packing would be a lattice
        #there will be a max of sqrt(nodes) along the shortest side, so that's how big it could possibly be
        t_max = min(size[0], size[1] - 100) / math.sqrt(num_nodes)
        #0.75 is not a mathematically derived number, it's just a error factor that seemed to work
        #in all the cases I used
        if min_distance > 0.75 * t_max:
            raise ValueError(f'{min_distance} is too large for min_distance')
        
        #until we have generated enough nodes, keep making a random choice, looping through
        #the other nodes and checking to make sure it is at least min_distance away
        while len(self.graph.nodes) < num_nodes:
            x = random.randint(10, size[0] - 10)
            y = random.randint(10, size[1] - 130)
            is_valid = True
            for node in self.graph.nodes:
                distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                if distance < min_distance:
                    is_valid = False
                    break
            if is_valid:
                self.graph.nodes.append(Node(x, y, 3))

    def draw_graph(self, graph, path=None):
        '''
        draw_graph draws the nodes/edges of a graph to the screen
        graph -> Graph
        '''
        if len(graph.nodes) > 0:
            for node in graph.nodes:
                node.draw(self.screen)

        if len(path) > 0:
            for i in range(len(path)):
                start = path[i]
                #modding by len(graph.path) makes sure the last node is connected to the first
                end = path[(i + 1) % len(path)]
                pygame.draw.line(self.screen, 'black', (start.x, start.y), (end.x, end.y))

    def show_text(self, message, offset):
        '''
        show_text writes message to the pygame window
        message -> string
        offset -> how far from the bottom to write the message
        '''
        surface = self.font.render(message, False, 'black')
        height = pygame.display.get_window_size()[1]
        self.screen.blit(surface, (0, height - offset))

    def solve(self):
        '''
        solve either finds or approximates the shortest path using one of the techniques in
        the parameter list
        bruteforce -> bool: solve with brute force
        annealing -> bool: approximate with annealing
        '''
        try:
            path = next(self.solver.iter)
            self.draw_graph(self.graph, path)
            self.solver.time += self.clock.get_time()
        except StopIteration:
            self.draw_graph(self.graph, self.solver.min_path)
        self.show_text(f'Current Min: {self.solver.min_distance}', 100)
        self.show_text(f'Time to Solve: {self.solver.time / 1000} s', 50)

s = Screen()
s.populate_screen(6, 10)
s.loop(annealing=True)