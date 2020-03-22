# blackbox

A testbed for black-box optimization techniques.

> **Note**: This is still in development.

## Usage

To run the example (an implementation of particle swarm optimization), do the
following:

```
$ git clone https://github.com/tilleyd/blackbox.git
$ cd blackbox
$ python3 -m blackbox examples/config.json
```

This will load the configuration from `examples/config.json`, which will start
a visualization of PSO on the modified Schwefel function. It will look similar
to the figure below.

![PSO Example](https://raw.githubusercontent.com/tilleyd/blackbox/master/examples/pso_vis.png)
