import numpy as np
import math
import random

# =========================
# HYPERCHAOTIC ITERATION
# =========================
def hyper_iter(x, y, z, w, params):
    a, b, c, d = params

    dx = a * (y - x) + b * z
    dy = (c - a) * x - x * z + d * y
    dz = x * y - b * z + math.sin(w)
    dw = -c * w + 0.1 * math.cos(x)

    # numerical integration step
    x += dx * 0.01
    y += dy * 0.01
    z += dz * 0.01
    w += dw * 0.01

    # stability (avoid overflow)
    x, y, z, w = np.clip([x, y, z, w], -50, 50)

    return x, y, z, w


# =========================
# FITNESS FUNCTION
# =========================
def fitness(params):
    x, y, z, w = 0.1, 0.2, 0.3, 0.4
    seq = []

    for _ in range(100):
        x, y, z, w = hyper_iter(x, y, z, w, params)
        seq.append(x)

    # higher std = better randomness
    return np.std(seq)


# =========================
# GA OPTIMIZATION
# =========================
def optimize_hyperchaotic_params(seed):

    random.seed(seed)
    np.random.seed(seed)

    population_size = 10

    # initial population (safe range)
    population = np.random.uniform(0.1, 2, (population_size, 4))

    for _ in range(5):

        # evaluate fitness
        fitness_values = [fitness(p) for p in population]

        # select best half
        idx = np.argsort(fitness_values)[-population_size // 2:]
        best = population[idx]

        new_population = []

        for _ in range(population_size):
            p1 = best[random.randint(0, len(best) - 1)]
            p2 = best[random.randint(0, len(best) - 1)]

            # crossover
            child = (p1 + p2) / 2

            # mutation
            child += np.random.normal(0, 0.05, 4)

            new_population.append(child)

        population = np.array(new_population)

    best_params = population[0]

    return {
        "params": best_params,
        "init_state": [0.1, 0.2, 0.3, 0.4]
    }