# blackbox
# Manages the configuration and entrypoint for blackbox. Note that blackbox
# can be used as a pure python module, but is convenient to use as a
# standalone application.
# Author: Duncan Tilley

optimizer_fp = None
config_fp = None

def plot(logs):
    from blackbox.testbed import Log
    Log.show_metrics(logs)

def optimize(opt, contexts, hyper):
    # Evaluate all hyper params
    for k in hyper:
        if k != 'log_dir' and type(hyper[k]) == 'string':
            hyper[k] = eval(hyper[k])

    # Create the testbed and start
    from blackbox.testbed import Testbed
    log_dir = None if not ('log_dir' in hyper) else hyper['log_dir']
    tb = Testbed(contexts, log_dir)
    results = tb.evaluate(opt.bb_init, opt.bb_iterate, hyper)
    for i, r in enumerate(results):
        print('%i (%s): ' %(i, tb.contexts[i]._f.__name__), r[0], '-> %f' %(r[1]))

def help():
    print('''blackbox (https://github.com/tilleyd/blackbox)
Usage: python3 -m blackbox <mode> [...args]
Modes:
    plot <log.pickle ...>

    *.py <context> [hyper-params ...]

    *.json [hyper-params ...]''')

def read_log_file(filename):
    import pickle
    with open(filename, 'rb') as log_file:
        return pickle.load(log_file)

def read_optimizer_file(filename):
    import os
    import importlib
    optimizer_name = os.path.split(filename)[1][:-3] # filename without .py
    # NOTE: importing the optimizer as a module rather than simply using eval()
    # In addition to being cleaner as it doesn't pollute the local scope, it
    # also allows stepping into the optimizer code with a debugger.
    spec = importlib.util.spec_from_file_location(optimizer_name, filename)
    optimizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimizer)
    print('Loaded optimizer from \'%s\'' %(filename))
    return optimizer

def read_context_file(filename):
    import os
    import importlib
    context_name = os.path.split(filename)[1][:-3] # filename without .py
    spec = importlib.util.spec_from_file_location(context_name, filename)
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    print('Loaded context from \'%s\'' %(filename))
    return context.bb_contexts

def read_config_file(filename):
    import json
    with open(filename, 'r') as f:
        config = json.load(f)
    print('Loaded config from \'%s\'' %(filename))
    optimizer_fp = config['optimizer']
    context_fp = config['context']
    hyper = config['hyper']
    if 'log_dir' in config:
        # we cheat a bit and include the log_dir as a hyper-parameter
        hyper['log_dir'] = config['log_dir']
    return read_optimizer_file(optimizer_fp), read_context_file(context_fp), hyper

# First check for any arguments overriding the optimizer or config file or
# if loading a log file
import sys
import os.path
if len(sys.argv) < 2:
    print('Error: missing mode argument, try \'blackbox help\'')
    exit()

# Check the mode argument
if sys.argv[1] == 'plot':
    logs = []
    for arg in sys.argv[2:]:
        logs.append(read_log_file(arg))
    plot(logs)
elif sys.argv[1].endswith('.py'):
    opt = read_optimizer_file(sys.argv[1])
    context = read_context_file(sys.argv[2])
    hyper = {}
    for i in range(3, len(sys.argv), 2):
        hyper[sys.argv[i]] = sys.argv[i+1]
    optimize(opt, context, hyper)
elif sys.argv[1].endswith('.json'):
    opt, context, hyper = read_config_file(sys.argv[1])
    for i in range(2, len(sys.argv), 2):
        hyper[sys.argv[i]] = sys.argv[i+1]
    optimize(opt, context, hyper)
elif sys.argv[1] == 'help':
    help()
else:
    print('Error: invalid mode argument')

exit()
