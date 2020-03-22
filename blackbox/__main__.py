# blackbox
# Manages the configuration and entrypoint for blackbox. Note that blackbox
# can be used as a pure python module, but is convenient to use as a
# standalone application.
# Author: Duncan Tilley

def bb_init(context, state):
    raise Exception('Your optimizer does not define bb_init(context, state)')

def bb_iterate(context, state, t):
    raise Exception('Your optimizer does not define bb_iterate(context, state, t')

# The first argument should contain the script

optimizer_fp = None
config_fp = None

# First check for any arguments overriding the optimizer or config file
import sys
import os.path
for arg in sys.argv[1:]:
    if optimizer_fp == None and arg.endswith('.py'):
        optimizer_fp = arg
    elif config_fp == None and arg.endswith('.json'):
        config_fp = arg

# Attempt to load the config file
config = {}
if config_fp != None:
    if not os.path.isfile(config_fp):
        print('Error: file \'%s\' does not exist' %(config_fp))
        exit()
    else:
        with open(config_fp) as config_file:
            import json
            config = json.load(config_file)
            print('Loaded parameters from \'%s\'' %(config_fp))
elif config == None:
    if os.path.isfile('config.json'):
        with open('config.json') as config_file:
            import json
            config = json.load(config_file)
            print('Loaded parameters from \'config.json\'')

# Make sure an optimizer is defined
if optimizer_fp == None:
    if 'optimizer' in config:
        optimizer_fp = config['optimizer']
    else:
        print('Error: no optimizer file specified')
        exit()

optimizer_name = os.path.split(optimizer_fp)[1][:-3] # filename without .py
optimizer = None
if os.path.isfile(optimizer_fp):
    # Note: importing the optimizer as a module rather than simply using eval()
    # In addition to being cleaner as it doesn't pollute the local scope, it
    # also allows stepping into the optimizer code with a debugger.
    import importlib
    spec = importlib.util.spec_from_file_location(optimizer_name, optimizer_fp)
    optimizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimizer)
else:
    print('Error: file \'%s\' does not exist' %(optimizer_fp))

# Load the hyperparameter config for the optimizer.
optimizer_config = {}
if 'hyper' in config and optimizer_name in config['hyper']:
    optimizer_config = config['hyper'][optimizer_name]
else:
    print('Warning: no hyperparameters found for \'%s\'' %(optimizer_name))

# Create the testbed and start
from blackbox.testbed import Testbed

# just the modified schwefel's function for now
def f(x):
    import numpy as np
    nx = len(x)
    sm = 0.0
    for i in range(0, nx):
        z = x[i] + 420.9687462275036
        if z < -500:
            zm = (abs(z) % 500) - 500
            t = z + 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        elif z > 500:
            zm = 500 - (z % 500)
            t = z - 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        else:
            sm += z * np.sin(np.sqrt(abs(z)))

    return 418.9829*nx - sm

tb = Testbed([f], 2, -100, 100)

mode = 'evaluate'
if 'mode' in config:
    mode = config['mode']
else:
    print('Warning: no mode found in config, defaulting to \'evaluate\'')

if mode == 'visualize':
    tb.visualize_all(optimizer.bb_init, optimizer.bb_iterate, optimizer_config)
elif mode == 'evaluate':
    results = tb.evaluate_all(optimizer.bb_init, optimizer.bb_iterate, optimizer_config)
    for i, r in enumerate(results):
        print('Function %i: found' %(i), r[0], '-> %f' %(r[1]))
else:
    print('Error: invalid mode \'%s\'' %(mode))
