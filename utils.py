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


def ils(solution, instance, constant_temperature=0.95, iterations_for_each_temp=100):
    # initial setup
    # initialize the temperature T = tmax
    temperature = instance.best_sol / np.sqrt(instance.nPoints)

    # save current and initialiye best solution variables
    current_sol = np.array(solution)
    current_len = compute_length(solution, instance.dist_matrix)
    best_sol = np.array(solution)
    best_len = current_len

    # main loop
    while temperature > 0.001:
        for it in range(iterations_for_each_temp):
            # perturbation with Double Bridge
            next_sol_p, new_cost_p = DoubleBridge.perturbate_solution(current_sol,
                                                                      current_len,
                                                                      instance.dist_matrix)
            # print(new_cost_p)
            # local search
            next_sol, new_cost = local_search(next_sol_p, new_cost_p, instance)
            # print(new_cost, current_len)
            # print()
            # break
            # acceptance criterions
            if new_cost - current_len < 0:
                # print('updated sol')
                current_sol = next_sol
                current_len = new_cost
                if current_len < best_len:
                    # print("update best")
                    best_sol = current_sol
                    best_len = current_len
                    # print(best_len)
                # print()
            else:
                r = np.random.uniform(0, 1)
                if r < np.exp(- (new_cost - current_len) / temperature):
                    current_sol = next_sol
                    current_len = new_cost
        # decrease temprateure
        temperature *= constant_temperature
    return best_sol


# same code seen in previous tutorials
# in this case its 2opt, but for ILS it can be any local search algorithm
def local_search(solution, new_len, instance):
    matrix_dist = instance.dist_matrix
    new_tsp_sequence = np.copy(np.array(solution))
    uncross = 0
    seq_length = len(solution)
    try_again = True
    while uncross < 10:
        new_tsp_sequence = np.roll(new_tsp_sequence, np.random.randint(seq_length)).astype(np.int64)
        new_tsp_sequence, new_reward, uncr_ = step2opt(new_tsp_sequence, matrix_dist, new_len)
        uncross += uncr_
        if new_reward < new_len:
            new_len = new_reward
            try_again = True
        else:
            if try_again:
                try_again = False
            else:
                return new_tsp_sequence, new_len

    return new_tsp_sequence.tolist(), new_len


def step2opt(solution, matrix_dist, distance):
    seq_length = len(solution)
    tsp_sequence = np.copy(solution)
    uncrosses = 0
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            if gain_2opt(i, j, tsp_sequence, matrix_dist) > 0:
                new_distance = distance - gain_2opt(i, j, tsp_sequence, matrix_dist)
                tsp_sequence = swap2opt(tsp_sequence, i, j)
                # print(new_distance, distance)
                return tsp_sequence, new_distance, 1
    return tsp_sequence, distance, 1


def swap2opt(tsp_sequence, i, j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j] = np.flip(tsp_sequence[i:j], axis=0)
    return new_tsp_sequence.astype(np.int64)


def gain_2opt(i, j, tsp_sequence, matrix_dist):
    try:
        old_link_len = (matrix_dist[tsp_sequence[i],
        tsp_sequence[i - 1]]
                        + matrix_dist[tsp_sequence[j],
                tsp_sequence[j - 1]])
        changed_links_len = (matrix_dist[tsp_sequence[j],
        tsp_sequence[i]]
                             + matrix_dist[tsp_sequence[i - 1],
                tsp_sequence[j - 1]])
        return + old_link_len - changed_links_len
    except:
        print(i, j, tsp_sequence[i], tsp_sequence[j], tsp_sequence[i - 1], tsp_sequence[j - 1])


class DoubleBridge:

    @staticmethod
    def difference_cost(solution, a, b, c, d, matrix):
        n = matrix.shape[0]
        to_remove = matrix[solution[a - 1], solution[a]] + matrix[solution[b - 1], solution[b]] + matrix[
            solution[c - 1], solution[c]] + matrix[solution[d - 1], solution[d]]
        to_add = matrix[solution[a], solution[c - 1]] + matrix[solution[b], solution[d - 1]] + matrix[
            solution[c], solution[a - 1]] + matrix[solution[d], solution[b - 1]]
        return to_add - to_remove

    @staticmethod
    def perturbate_solution(solution, actual_cost, matrix):
        # generate 4 random indices
        a, b, c, d = np.sort(np.random.choice(matrix.shape[0], size=4, replace=False))
        # get new solution of double bridge
        B = solution[a:b]
        C = solution[b:c]
        D = solution[c:d]
        A = np.concatenate((solution[d:], solution[:a]))
        new_solution = np.concatenate((A, D, C, B))
        # double bridge gain computation
        new_length = actual_cost + DoubleBridge.difference_cost(solution, a, b, c, d, matrix)
        return new_solution, new_length


# while loop, calling step2opt until the new solution is shorter (an improvment)
# when the new solution is smaller than the previous one then you keep going in the while
# if this is not true, we return the solution we found
def loop2opt(solution, instance, max_num_of_uncrosses=10000):
    matrix_dist = instance.dist_matrix
    new_len = compute_length(solution, matrix_dist)
    new_tsp_sequence = np.copy(np.array(solution))
    uncross = 0
    try_again = True
    seq_length = new_len
    # TODO
    while uncross < max_num_of_uncrosses:
        new_tsp_sequence, new_reward, uncr_ = step2opt(new_tsp_sequence, matrix_dist, new_len)
        new_tsp_sequence = np.roll(new_tsp_sequence, np.random.randint(seq_length)).astype(np.int64)
        if new_reward < new_len:
            new_len = new_reward
            try_again = True
            uncross += uncr_
        else:
            if try_again:
                try_again = False
            else:
                return new_tsp_sequence.tolist(), new_len, uncross
    # END TODO
    return new_tsp_sequence.tolist(), new_len, uncross


def sa(solution, instance, constant_temperature=0.95, iterations_for_each_temp=100):
    # initial setup
    # initialize the temperature T = tmax
    temperature = instance.best_sol / np.sqrt(instance.nPoints)

    # initializing initial solution and len (x)
    current_sol = np.array(solution)
    # initial energey of x, E(x), energy is the tour lenght
    current_len = compute_length(solution, instance.dist_matrix)

    # variables to save the best sol
    best_sol = np.array(solution)
    best_len = current_len
    # main loop
    while temperature > 0.001: # lowest accepted temeperature, T > Tmin and E > Eth
        # TODO
        # equilibrium for the current temperature, or specify max iteration for each temperature
        for it in range(iterations_for_each_temp):
            # generate new candidate solution from neighbors
            next_sol, delta_E = random_sol_from_neigh(current_sol, instance)
            # if neighbor solution is an improvment: accept it and save it in current and best solution variables
            if delta_E < 0:
                current_sol = next_sol
                current_len += delta_E
                if current_len < best_len:
                    best_sol = current_sol
                    best_len = current_len
            else: # neighbor solution is not an improvment: compute exp(-deltaE/E) and generate a random number r in range [0, 1] and ecide if accept it or not
                r = np.random.uniform(0, 1)
                if r < np.exp(- delta_E / temperature):
                    current_sol = next_sol
                    current_len += delta_E

        # decrease temperature
        temperature *= constant_temperature
    # END TODO
    # return best tour solution
    return best_sol.tolist()

# genrate a random solution that is neghbor to the exisitng solution
def random_sol_from_neigh(solution, instance):
    # generate 2 random i and j
    i, j = np.random.choice(np.arange(1, len(solution) - 1), 2, replace=False)
    # sort i and j
    i, j = np.sort([i, j])
    solution = np.roll(solution, np.random.randint(len(solution)))
    # do an 2opt swap based on the 2 random i an j generated, and compute the gain of this 2opt
    return sa_swap2opt(solution, i, j), gain(i, j, solution, instance.dist_matrix)

# swap2opt step as seen in previous tutorial
def sa_swap2opt(tsp_sequence, i, j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j + 1] = np.flip(tsp_sequence[i:j + 1], axis=0)  # flip or swap ?
    return new_tsp_sequence

# swap2opt gain as seen in previous tutorial
def gain(i, j, tsp_sequence, matrix_dist):
    old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
        tsp_sequence[j], tsp_sequence[j + 1]])
    changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
        tsp_sequence[i], tsp_sequence[j + 1]])
    return - old_link_len + changed_links_len
