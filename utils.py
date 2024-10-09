import numpy as np
import matplotlib.pyplot as plt


# define function to compute euclidean distance between 2 points
def distance_euc(point_i, point_j):
    rounding = 0
    x_i, y_i = point_i[0], point_i[1]
    x_j, y_j = point_j[0], point_j[1]
    # use the numpy sqrt method
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    return round(distance, rounding)


class ProblemInstance:

    def __init__(self, name_tsp):

        # boolean if the problem have an sxisting optimal solution
        self.exist_opt = False

        # the optimal tour that was found before
        self.optimal_tour = None

        # distance matrix
        # Example distance matrix representing distances between 4 points
        #    A    B    C    D
        # A  0.0  2.5  3.0  4.2
        # B  2.5  0.0  1.8  3.6
        # C  3.0  1.8  0.0  2.9
        # D  4.2  3.6  2.9  0.0
        self.dist_matrix = None

        # read raw data
        with open(name_tsp) as f_o:
            data = f_o.read()
            self.lines = data.splitlines()

        # store metadata set information

        # here we expect the name of the problem
        self.name = self.lines[0].split(' ')[1]

        # here we expect the number of points in the considered instance
        self.nPoints = int(self.lines[3].split(' ')[1])

        # here the length of the best solution
        self.best_sol = float(self.lines[5].split(' ')[1])

        # read all data points and store them
        self.points = np.zeros((self.nPoints, 3))  # this is the structure where we will store the pts data
        for i in range(self.nPoints):
            line_i = self.lines[7 + i].split(' ')
            self.points[i, 0] = int(line_i[0])
            self.points[i, 1] = float(line_i[1])
            self.points[i, 2] = float(line_i[2])

        # create distance matrix bx calling create_dist_matrix
        self.create_dist_matrix()

        # TODO [optional]
        # if the problem is one with a optimal solution, that solution is loaded
        self.optimal_tour = np.zeros(self.nPoints, dtype=int)
        if name_tsp in ["./problems/eil76.tsp", "./problems/kroA100.tsp"]:
            # change the boolean to True since an optimal solution exist
            self.exist_opt = True
            # open the optimal solution file and read it
            file_object = open(name_tsp.replace(".tsp", ".opt.tour"))
            data = file_object.read()
            file_object.close()
            lines = data.splitlines()

            # read all data points and store them

            # initialize an array with 0s using np.zeros
            self.optimal_tour = np.zeros(self.nPoints, dtype=int)
            # read the points from the file and fill the array
            for i in range(self.nPoints):
                line_i = lines[5 + i].split(' ')
                self.optimal_tour[i] = int(line_i[0]) - 1

    # for display purposes
    def print_info(self):
        print("\n#############################\n")
        print('name: ' + self.name)
        print('nPoints: ' + str(self.nPoints))
        print('best_sol: ' + str(self.best_sol))
        print('exist optimal: ' + str(self.exist_opt))

    # for display purposes
    def plot_data(self, show_numbers=False):
        # define a figure with it's size
        plt.figure(figsize=(8, 8))

        # give it a title as the problem instance name
        plt.title(self.name)

        # scatter takes an x and y axis values that are in this case the points that we have
        plt.scatter(self.points[:, 1], self.points[:, 2])

        # if we want to add to the plot next to each point it's number or id
        if show_numbers:
            for i, txt in enumerate(np.arange(self.nPoints)):
                plt.annotate(txt, (self.points[i, 1], self.points[i, 2]))

        # display the plot/figure
        plt.show()

    def create_dist_matrix(self):  # TODO
        # initialize the matrix with dimension n x n
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        # loop over all the pints and compute the euclidean distance between every pair of points with index i and j
        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = distance_euc(self.points[i][1:3], self.points[j][1:3])

        # transpose of the matrix and fill missing other 0s
        # this was made to reduce the complexity
        self.dist_matrix += self.dist_matrix.T


# library to get time
from time import time as t


class SolverTSP:
    def __init__(self, algorithm_name, problem_instance, available_methods):
        # duration taken to find the solution
        self.duration = np.inf

        # lenght of the tour found by the solver
        self.found_length = np.inf

        # name of the algorithm
        self.algorithm_name = algorithm_name

        # initialization
        self.name_method = "initialized with " + algorithm_name

        # boolean to check if solved problem
        self.solved = False

        # TSP problem intance to be solved
        self.problem_instance = problem_instance

        # the found solution
        self.solution = None

        self.available_methods = available_methods

    # function that takes computes the solution by applying the function given as input
    def compute_solution(self, verbose=True, return_value=True):
        self.solved = False
        if verbose:
            print(f"###  solving with {self.algorithm_name}  ####")

        # starting time
        start_time = t()

        # available methods that was defined earlier have a pointer to the function (algorithm name) and take as input the problem instance
        self.solution = self.available_methods[self.algorithm_name](self.problem_instance)

        # check if solution is valid implemented
        assert self.check_if_solution_is_valid(self.solution), "Error the solution is not valid"
        # end time of the solution
        end_time = t()
        self.duration = np.around(end_time - start_time, 3)
        if verbose:
            print(f"###  solved  ####")
        self.solved = True
        # compute the length of the solution tour
        self.evaluate_solution()

        # compute the gap
        self._gap()

        if return_value:
            return self.solution

    # simple plot of the soluton found
    def plot_solution(self):
        assert self.solved, "You can't plot the solution, you need to compute it first!"
        plt.figure(figsize=(8, 8))
        self._gap()
        plt.title(f"{self.problem_instance.name} solved with {self.name_method} solver, gap {self.gap}")
        ordered_points = self.problem_instance.points[self.solution]
        plt.plot(ordered_points[:, 1], ordered_points[:, 2], 'b-')
        plt.show()

    # check if the solution contains all the point and visited exactly one
    def check_if_solution_is_valid(self, solution):
        rights_values = np.sum([self.check_validation(i, solution) for i in np.arange(self.problem_instance.nPoints)])
        if rights_values == self.problem_instance.nPoints:
            return True
        else:
            return False

            # check if a point/node inside a solution

    def check_validation(self, node, solution):
        if np.sum(solution == node) == 1:
            return 1
        else:
            return 0

    # compute the tour lengh by the help of distance matrix already computed
    def evaluate_solution(self, return_value=False):
        total_length = 0
        from_node_id = self.solution[0]  # starting_node

        # loop over all the nodes and add successive distances
        for node_id in self.solution[1:]:
            edge_distance = self.problem_instance.dist_matrix[from_node_id, node_id]
            total_length += edge_distance
            from_node_id = node_id

        # have a complete tour and add the distance to go back to starting node
        self.found_length = total_length + self.problem_instance.dist_matrix[self.solution[0], from_node_id]
        if return_value:
            return total_length

    # compute (solution - best solution)/best solution in %
    def _gap(self):
        self.evaluate_solution(return_value=False)
        self.gap = np.round(
            ((self.found_length - self.problem_instance.best_sol) / self.problem_instance.best_sol) * 100, 2)


def random_method(instance_):  # TODO
    return np.random.choice(np.arange(instance_.nPoints), size=instance_.nPoints,
                            replace=False)


def compute_length(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    total_length += dist_matrix[starting_node, from_node]
    return total_length