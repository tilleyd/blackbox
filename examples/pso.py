# Blackbox usage example for particle swarm optimization:
# Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. 1998 IEEE
# International Conference on Evolutionary Computation Proceedings. IEEE World
# Congress on Computational Intelligence (Cat. No.98TH8360), 69–73.
# Author: Duncan Tilley

def _evaluate_particles(context, x):
    import numpy as np
    ss = len(x)
    y = np.zeros(ss)
    for i in range(0, ss):
        y[i] = context.evaluate(x[i])
    return y

def bb_init(context, state, hyper):
    """
    Initializes the PSO's state.

    Args:
        context (Context): Representation of the current problem.
        state (dict): An empty dictionary to be initialized with the PSO state.
        hyper (dict): Hyperparameter configuration.
    """
    import numpy as np
    x = np.random.uniform(low=context.domain_min,
                          high=context.domain_max,
                          size=(hyper['swarm_size'], context.dimensions))
    v = np.zeros((hyper['swarm_size'], context.dimensions))
    y = _evaluate_particles(context, x)

    pbest_x = x.copy()
    pbest_y = y.copy()

    gbest_i = np.argmin(pbest_y)
    gbest_y = pbest_y[gbest_i]
    gbest_x = pbest_x[gbest_i].copy()

    state['x'] = x
    state['v'] = v
    state['pbest_x'] = pbest_x
    state['pbest_y'] = pbest_y
    state['gbest_x'] = gbest_x
    state['gbest_y'] = gbest_y

def bb_iterate(context, state, hyper, t):
    """
    Performs a single iteration of PSO.

    Args:
        context (Context): Representation of the current problem.
        state (dict): The current state of the optimizer.
        hyper (dict): Hyperparameter configuration.
        t (int): The current iteration.
    """
    import numpy as np
    x = state['x']
    v = state['v']
    pbest_x = state['pbest_x']
    gbest_x = state['gbest_x']

    # update velocities and positions
    pv = hyper['c1'] * np.random.uniform(0.0, 1.0, x.shape) * (pbest_x - x)
    gv = hyper['c1'] * np.random.uniform(0.0, 1.0, x.shape) * (gbest_x - x)
    v = hyper['w'] * v + pv + gv
    x = x + v
    y = _evaluate_particles(context, x)

    # update pbest
    pbest_y = state['pbest_y']
    mask = y < pbest_y
    pbest_x[mask] = x[mask]
    pbest_y[mask] = y[mask]

    # update gbest
    gbest_i = np.argmin(pbest_y)
    gbest_y = pbest_y[gbest_i]
    gbest_x = pbest_x[gbest_i].copy()

    state['x'] = x
    state['v'] = v
    state['pbest_x'] = pbest_x
    state['pbest_y'] = pbest_y
    state['gbest_x'] = gbest_x
    state['gbest_y'] = gbest_y
