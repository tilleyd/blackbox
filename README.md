# blackbox

A testbed for black-box optimization techniques. Allows the evaluation of
regular or population-based optimization techniques. Evaluations
can be logged for record keeping and further calculations.

## Evaluating

```
python3 -m blackbox <optimizer.py> <context.py> [key  val...]
```

Loads the optimizer, testbed context and passed hyper-parameters. Each function
defined by the context is evaluated and printed separately and optionally
logged.

#### `<optimizer.py>`

A python file that defines the global functions as shown below. See
`examples/pso.py` for an example of a particle swarm implementation.

```python
def bb_init(context, state, hyper):
    """
    Initializes your optimizer.

    Args:
        context (blackbox.testbed.Context): The object containing the blackbox
            function information.
        state (dict): An empty dictionary where your optimizer's state should
            be kept. Will be passed to bb_iterate.
        hyper (dict): A dictionary containing hyper-parameters.
    """
    # your initialization logic here

def bb_iterate(context, state, hyper, t):
    """
    Performs an iteration of your optimizer.

    Args:
        context (blackbox.testbed.Context): Same as bb_init.
        state (dict): A dictionary containing any state. Will be passed to
            the next bb_iterate call.
        hyper ([type]): Same as bb_init.
        t (int): The current iteration (starting at 1).
    """
    # your per-iteration logic here
```

#### `<context.py>`

A python file that defines the global `bb_context` list, containing a list of
`blackbox.testbed.Context` objects. See `examples/context.py` for an example.
This file defines the problem instance.

#### `[key val...]`

Key-value pairs that are passed to the `hyper` dictionary in `bb_init` and
`bb_iterate`. Note that all values are passed through `eval()` first, so
any valid python code is allowed.

The `log_dir <path>` is a special parameter, described under "Logging" below.

<br>

```
python3 -m blackbox <config.json> [key val...]
```

Loads the optimizer, testbed context and hyper-parameters from a JSON file.
Additional hyper-parameters can be specified. Each function defined by the
context is evaluated and printed separately and optionally logged.

#### `<config.json>`

A JSON file describing the optimizer, context and hyper-parameters. Has the
form as shown below. Hyper-parameters can be any scalar, python dictionary
or evaluatable string. See `examples/config.json` for an example:

```json
{
    "optimizer": "path/to/optimizer.py",
    "context": "path/to/context.py",
    "log_dir": "optional/log/directory",
    "hyper": {
        "any": 0.5,
        "hyper": {"x": 0.0, "y": 1.0},
        "parameters": "10**5"
    }
}
```

#### `[key val...]`

Key-value pairs that are passed to the `hyper` dictionary in `bb_init` and
`bb_iterate`. Note that all values are passed through `eval()` first, so
any valid python code is allowed. These values replace any keys defined in
the config file.

## Logging

When passing the `log_dir <path>` argument or including it in the config file,
the run will produce a log pickle in `<path>/<timestamp>/` for each evaluated
function. The pickles can easily be read back and extracted as the following:

```python
import pickle
with open('your/log/file.pkl', 'rb') as log_file:
    log = pickle.load(log_file)
for timestep in log:
    # state: dictionary containing the state after timestep t
    # evaluated_points: list of nd-points evaluated during timestep t
    # best_y: best found value
    # t: iteration count (0 -> initialization)
    state, evaluated_points, best_y, t = timestep
```

## Plotting

```
python3 -m blackbox plot <log.pkl ...>
```

Firstly creates a diversity plot using the mean distance from the population
center per iteration and secondly creates a value plot directly showing the best
value found after each iteration.

#### `<log.pkl ...>`

A list of log pickles. Each listed log will be included in the plot.
