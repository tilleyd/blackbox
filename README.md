# blackbox

A testbed for black-box optimization techniques. Allows the visualization and
evaluation of regular or population-based optimization techniques. Evaluations
can be logged for record keeping and further calculations.

## Usage

To run the example (an implementation of particle swarm optimization), do the
following:

```
$ git clone https://github.com/tilleyd/blackbox.git
$ cd blackbox
$ python3 -m blackbox examples/config.json
```

This will load the configuration from `examples/config.json`, which will start
an evaluation of PSO and write a pickled log object to `examples/log/`.

To load the log file and show simple metrics, do the following by replacing
`<timestamp>` and `<logfile>` with the relevant path:

```
$ python3 -m blackbox examples/log/<timestamp>/<logfile>.pickle
```

Alternatively, set `"mode": "visualize"` in the config file to change to
visualization. It will provide a iteration-by-iteration visualization similar
to the figure below.

![PSO Example](https://raw.githubusercontent.com/tilleyd/blackbox/master/examples/pso_vis.png)
