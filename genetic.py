from deap import base, creator, tools
import random
from setup import *
from scipy.optimize import minimize


def genetic_algorithm(d, bases, origin, basis, alpha_n = 0):
    r = 0.1
    reachable_points = d["e"]
    opt_n, n_bounds = d["n"]
    n_min, n_max = n_bounds
    singularities, s0 = d["S"]
    manipulability, m0 = d["M"]
    joints = d["joints"]
    L, norm = d["L"]
    Bs = d["Bs"]
    Be, walls = d["Be"]
    
    def create_individual():
        n_all = random.randint(n_min, n_max)
        vars = [[] for i in range(n_all)]
        pos = [i for i in range(n_all)]
        random.shuffle(pos)
        for i in range(n_all):
            j = random.choice(joints)
            if j == 0:
                axis1 = random.randint(0, 2)
                axis2 = random.randint(0, 2)
                length = 3 + random.uniform(0, 3)
                vars[pos[i]] = [0, axis1, length, axis2]
            elif j == 1:
                axis = random.randint(0, 2)
                length = random.uniform(0.1, 1)
                dmax = random.uniform(0.1, 7)
                vars[pos[i]] = [1, axis, length, dmax]
            else:
                axis = random.randint(0, 2)
                axis2 = random.randint(0, 2)
                angle = random.uniform(0, np.pi)
                length = random.uniform(0.1, 0.2)
                vars[pos[i]] = [2, axis, axis2, angle, length]
        vars = [el for var in vars for el in var]
        return vars
    
    def f(vars):
        links = []
        all = vars
        axis = ["x", "y", "z"]
        n = 0
        n_flip = 0
        lengths = 0
        n_points = len(reachable_points)
        if len(vars) < 4:
            return 1000,
        while len(vars) > 4:
            if vars[0] == 0:
                var = vars[:4]
                length = var[2]
                if type(var[3]) != int or type(var[1]) != int:
                    return 100000,
                links.append(RevoluteJoint(axis[var[1]], length, r, axis[var[3]]))
                if norm == 1:
                    lengths += abs(length)
                else:
                    lengths += length**2
                n+=1
                vars = vars[4:]
            elif vars[0] == 1:
                var = vars[:4]
                length = var[2]
                if type(var[1]) != int:
                    return 100000,
                dmax = var[3]
                links.append(PrismaticJoint(axis[var[1]], length, r, dmax))
                if norm == 1:
                    lengths += abs(length)
                    lengths += abs(dmax)
                else:
                    lengths += length**2
                    lengths += dmax**2
                
                n+=1
                vars = vars[4:]
            else:
                var = vars[:5]
                length = var[4]
                if type(var[2]) != int or type(var[1]) != int:
                    return 100000,
                links.append(Flip(axis[var[1]], var[3], length, r, axis[var[2]]))
                if norm == 1:
                    lengths += abs(length)
                else:
                    lengths += length**2
                n_flip+=1
                vars = vars[5:]
        thetas = [0 for i in range(n)]
        robot = Robot(bases, origin, basis, links, thetas)
        
        def objective_function(params, desired_point):
            i = 0
            for link in robot.links:
                if link.type != "flip":
                    link.set_params(params[i])
                    i+=1
            return np.linalg.norm(robot.forward_kinematics(True) - desired_point)
    
        error = 0
        manipulability_ind = np.inf
        singularities_index = np.inf
        collisions = 0
        self_collisions = 0
        optimized_joint_angles = [0 for i in range(robot.n)]
        for point in reachable_points:
            bounds = []
            desired_point = point
            for link in robot.links:
                if link.type == "revolute":
                    bounds.append((-2*np.pi, 2*np.pi))
                elif link.type == "prismatic":
                    bounds.append((0, link.dmax))
            result = minimize(objective_function, optimized_joint_angles, args=(desired_point,), bounds = bounds)
            optimized_joint_angles = result.x
            error += objective_function(optimized_joint_angles, desired_point)
            i = 0
            for link in robot.links:
                if link.type != "flip":
                    link.set_params(optimized_joint_angles[i])
                    i+=1
            robot_shape = robot.robot_shape()
            if Bs:
                if robot.self_collision(robot_shape):
                    self_collisions+=1
            if Be:
                if robot.check_collision2(walls):
                    collisions+=1
        
            J = robot.jacobian()
            if singularities:
                singularities_index = min(robot.singularity_check(J), singularities_index)
            if manipulability:
                manipulability_ind = min(robot.manipulability_index(J), manipulability_ind)
        
        error = error/n_points
        e = error

        print(e, n, manipulability_ind, singularities_index, n_flip, lengths, self_collisions, collisions)

        singularities_index = s0 - min(singularities_index, s0)
        manipulability_ind = m0 - min(manipulability_ind, m0)
        
        if e<0.0001:
            if collisions + self_collisions == 0:
                print(all)
        
        error = 10000*error
        if L:
            error += lengths
        if manipulability:
            error += manipulability_ind
        if singularities:
            error += singularities_index
        if Bs:
            error += 1000*self_collisions
        if Be:
            error += 1000*collisions
        if opt_n:
            error += alpha_n*n
        
        print(error, n)
        return error,

    # Set up DEAP framework for minimizing a single objective
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Register the individual creation function and population generator
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function for DEAP
    def evaluate(individual):
        return f(individual)

    # Register GA operations
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)  # Crossover operation
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # Mutation operation
    toolbox.register("select", tools.selTournament, tournsize=5)  # Selection operation

    # Set mutation probability, crossover probability, and number of generations
    CXPB, MUTPB, NGEN = 0.5, 0.2, 5

    # Initialize population
    population = toolbox.population(n=50)

    # Begin the genetic algorithm evolution process
    for generation in range(NGEN):
        print("Generation:", generation)
        print("Population size:", len(population))


        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new offspring
        population[:] = offspring

    # Print the best solution found
    best_individuals = tools.selBest(population, 5)
    print("Best individuals are:")
    for best_individual in best_individuals:
        print("Best individual is:", best_individual)
        print("Fitness of the best individual:", evaluate(best_individual))

    best_individual = tools.selBest(population, 1)[0]
    print("Best individual is:", best_individual)
    print("Fitness of the best individual:", evaluate(best_individual))

    return best_individual



def gradient(ind, d, bases, origin, basis, alpha_n = 0):
    r = 0.1
    reachable_points = d["e"]
    opt_n, n_bounds = d["n"]
    n_min, n_max = n_bounds
    singularities, s0 = d["S"]
    manipulability, m0 = d["M"]
    joints = d["joints"]
    L, norm = d["L"]
    Bs = d["Bs"]
    Be, walls = d["Be"]

    def numerical_gradient(individual, indices, h=1e-5):
        gradient = []
        for i in range(len(individual)):
            if i not in indices:
                gradient.append(0)
            else:
                # Create a copy of the individual for perturbation
                individual_plus_h = individual[:]
                individual_minus_h = individual[:]
                
                # Perturb the ith element
                individual_plus_h[i] += h
                individual_minus_h[i] -= h
                print(individual_plus_h)
                
                # Compute the numerical gradient using central difference
                grad_i = (evaluate(individual_plus_h)[0] - evaluate(individual_minus_h)[0]) / (2 * h)
                gradient.append(grad_i)
        return gradient

    def gradient_descent(individual, learning_rate=1, steps=10):
        indices = [4*i+2 for i in range(len(individual)//4)]
        for _ in range(steps):
            print("Step: ", _)
            gradient = numerical_gradient(individual, indices)  # Numerical gradient
            print(gradient)
            if np.linalg.norm(gradient) < 1e-5:
                break
            gradient = gradient / np.linalg.norm(gradient)  # Normalize the gradient
            print(gradient)
            f = np.inf
            prev = np.copy(individual)
            prev_f = evaluate(individual)
            while f > prev_f[0]:
                individual = np.copy(prev)
                for i in indices:
                    individual[i] = max(individual[i] - learning_rate * gradient[i], 0)
                f = evaluate(individual)[0]
                learning_rate *= 0.7
                print(f)
                print(individual)
                
        return individual
    
    def f(vars):
        links = []
        all = vars
        axis = ["x", "y", "z"]
        n = 0
        n_flip = 0
        lengths = 0
        n_points = len(reachable_points)
        if len(vars) < 4:
            return 1000,
        while len(vars) > 4:
            if vars[0] == 0:
                var = vars[:4]
                length = var[2]
                links.append(RevoluteJoint(axis[int(var[1])], length, r, axis[int(var[3])]))
                if norm == 1:
                    lengths += abs(length)
                else:
                    lengths += length**2
                n+=1
                vars = vars[4:]
            elif vars[0] == 1:
                var = vars[:4]
                length = var[2]
                dmax = var[3]
                links.append(PrismaticJoint(axis[int(var[1])], length, r, dmax))
                if norm == 1:
                    lengths += abs(length)
                    lengths += abs(dmax)
                else:
                    lengths += length**2
                    lengths += dmax**2
                
                n+=1
                vars = vars[4:]
            else:
                var = vars[:5]
                length = var[4]
                links.append(Flip(axis[int(var[1])], var[3], length, r, axis[int(var[2])]))
                if norm == 1:
                    lengths += abs(length)
                else:
                    lengths += length**2
                n_flip+=1
                vars = vars[5:]
        thetas = [0 for i in range(n)]
        robot = Robot(bases, origin, basis, links, thetas)
        
        def objective_function(params, desired_point):
            i = 0
            for link in robot.links:
                if link.type != "flip":
                    link.set_params(params[i])
                    i+=1
            return np.linalg.norm(robot.forward_kinematics(True) - desired_point)
    
        error = 0
        manipulability_ind = np.inf
        singularities_index = np.inf
        collisions = 0
        self_collisions = 0
        optimized_joint_angles = [0 for i in range(robot.n)]
        for point in reachable_points:
            bounds = []
            desired_point = point
            for link in robot.links:
                if link.type == "revolute":
                    bounds.append((-2*np.pi, 2*np.pi))
                elif link.type == "prismatic":
                    bounds.append((0, link.dmax))
            result = minimize(objective_function, optimized_joint_angles, args=(desired_point,), bounds = bounds)
            optimized_joint_angles = result.x
            error += objective_function(optimized_joint_angles, desired_point)
            i = 0
            for link in robot.links:
                if link.type != "flip":
                    link.set_params(optimized_joint_angles[i])
                    i+=1
            robot_shape = robot.robot_shape()
            if Bs:
                if robot.self_collision(robot_shape):
                    self_collisions+=1
            if Be:
                if robot.check_collision2(walls):
                    collisions+=1
        
            J = robot.jacobian()
            if singularities:
                singularities_index = min(robot.singularity_check(J), singularities_index)
            if manipulability:
                manipulability_ind = min(robot.manipulability_index(J), manipulability_ind)
        
        error = error/n_points
        e = error

        print(e, n, manipulability_ind, singularities_index, n_flip, lengths, self_collisions, collisions)

        singularities_index = s0 - min(singularities_index, s0)
        manipulability_ind = m0 - min(manipulability_ind, m0)
        
        if e<0.0001:
            if collisions + self_collisions == 0:
                print(all)
        
        error = 10000*error
        if L:
            error += lengths
        if manipulability:
            error += manipulability_ind
        if singularities:
            error += singularities_index
        if Bs:
            error += 1000*self_collisions
        if Be:
            error += 1000*collisions
        if opt_n:
            error += alpha_n*n
        
        print(error, n)
        return error,

    def evaluate(individual):
        return f(individual)


    individual = gradient_descent(ind)
    
    print("Best individual is:", individual)
    print("Fitness of the best individual:", evaluate(individual))

    return individual
