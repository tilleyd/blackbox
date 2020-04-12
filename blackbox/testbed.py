# blackbox.testbed
# Framework for blackbox function execution.
# Author: Duncan Tilley

class Context(object):

    def __init__(self, function, dimensions, domain_min, domain_max):
        """
        Creates the context for a problem.

        Args:
            function (function): The evaluation function. Should take an
                nd-array as the only parameter.
            dimensions (int): The number of dimensions the problem is
                defined for.
            domain_min (float): The minimum search range boundary.
            domain_max (float): The maximum search range boundary.
        """
        import numpy as np
        self._f = function
        self._eval_callback = None
        self.dimensions = np.int(dimensions)
        self.domain_min = domain_min
        self.domain_max = domain_max

    def evaluate(self, x):
        """
        Evaluates the objective function and calls the eval_callback.

        Args:
            x (ndarray): Objective function parameters.

        Returns:
            float: Value of f(x).
        """
        v = self._f(x)
        if self._eval_callback:
            self._eval_callback(x, v)
        return v

    def set_eval_callback(self, callback):
        """
        Sets the eval_callback function that is called at each function
        evaluation.

        Args:
            callback (function): Callback with prototype (x, y), where y will
                be the value of f(x).
        """
        self._eval_callback = callback

class Testbed(object):

    def __init__(self, functions, dimensions, domain_min, domain_max, log_dir=None):
        """
        Creates a testbed for a set of functions.

        Args:
            functions ([function]): Array of functions to evaluate.
            dimensions (int or [int]): Number of dimensions per function.
            domain_min (float or [float]): Minimum search boundary per function.
            domain_max (float or [float]): Maximum search boundary per function.
            log_dir (string, optional): Where to log runs. If None, no logging.
        """
        import collections as coll
        import numpy as np

        # check dimension shape
        if not isinstance(dimensions, (coll.Sequence, np.ndarray)):
            dimensions = np.full(len(functions), dimensions)
        else:
            dimensions = np.array(dimensions)
        assert(dimensions.shape == (len(functions),))

        # check domain shape
        if not isinstance(domain_min, (coll.Sequence, np.ndarray)):
            domain_min = np.full(len(functions), domain_min)
        else:
            domain_min = np.array(domain_min)
        assert(domain_min.shape == (len(functions),))
        if not isinstance(domain_max, (coll.Sequence, np.ndarray)):
            domain_max = np.full(len(functions), domain_max)
        else:
            domain_max = np.array(domain_max)
        assert(domain_max.shape == (len(functions),))

        # create a context for each function
        contexts = []
        for i in range(0, len(functions)):
            contexts.append(Context(functions[i], dimensions[i], domain_min[i], domain_max[i]))

        self.__init__(contexts, log_dir)

    def __init__(self, contexts, log_dir=None):
        """
        Creates a testbed from given contexts.

        Args:
            contexts ([Context]): A list of context objects.
            log_dir (string, optional): Where to log runs. If None, no logging.
        """
        self.contexts = contexts

        # check and possibly create the log directory
        self.log_dir = None
        if log_dir != None:
            import os
            if os.path.exists(log_dir):
                if os.path.isdir(log_dir):
                    self.log_dir = log_dir
                else:
                    raise Exception('\'%s\' is not a directory' %(log_dir))
            else:
                os.mkdir(log_dir)
                self.log_dir = log_dir

    def evaluate(self, f_init, f_iterate, hyper, T=1000, MFE=0, show_metrics=False):
        """
        Iterates the problem functions and evaluates each.

        Args:
            f_init (function): The optimizer's initialization function.
            f_iterate (function): The optimizer's iterate function.
            hyper (dict): Hyperparameter config passed directly to f_init.
            T (int): Max number of iterations.
            MFE (int): Max function evaluations.
            show_metrics (bool): If true, will pause and show metric graphs
                after evaluating each function.

        Returns:
            list<(ndarry<float>, float)>: Array of tuples, where each tuple
                represents the best found parameter-value pair for each function.
        """
        eval_num = 0
        best_x, best_y = None, float('inf')
        eval_points = []
        def eval_callback(x, y):
            # count the evaluation and track the evaluated positions
            nonlocal eval_num
            nonlocal eval_points
            nonlocal best_x
            nonlocal best_y
            eval_num += 1
            if y < best_y:
                best_x = x
                best_y = y
            eval_points.append(x)

        log = []
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.log_dir:
            # Create the logging directory log_dir/timestamp/
            import os
            os.mkdir(os.path.join(self.log_dir, timestamp))

        if MFE <= 0:
            MFE = float('inf')
        results = []

        for (i, context) in enumerate(self.contexts):
            context.set_eval_callback(eval_callback)
            best_x, best_y = None, float('inf')
            state = {}
            eval_num = 0
            f_init(context, state, hyper)
            log.append((state, eval_points, best_y, 0))
            t = 1
            while t <= T and eval_num < MFE:
                eval_points = []
                f_iterate(context, state, hyper, t)
                log.append((state, eval_points, best_y, t))
                t += 1
            results.append((best_x, best_y))

            if self.log_dir:
                # Pickle the log to the file log_dir/timestamp/i_f.pickle
                import os
                filename = '%d_%s.pkl' %(i, context._f.__name__)
                filename = os.path.join(self.log_dir, timestamp, filename)
                with open(filename, 'wb') as logfile:
                    import pickle
                    pickle.dump(log, logfile)

            if show_metrics:
                Testbed.show_metrics(log)

        return results

class Log(object):

    def calculate_divergence(log):
        """
        Calculates the divergence series as the average distance from the
        population center at each timestep.

        Args:
            log (object): The log object created after a run or read from a
                log pickle.

        Returns:
            ndarry<float> containing the divergence metric at each iteration.
        """
        import numpy as np
        div = np.zeros(len(log))
        for i in range(0, len(log)):
            x = log[i][1]
            centroid = np.mean(x, axis=0)
            sm = 0
            for j in range(0, len(x)):
                d = x - centroid
                sm += np.sqrt(np.sum(d*d))
            sm /= len(x)
            div[i] = sm
        return div

    def show_metrics(logs):
        """
        Displays plots of convergence and performance metrics of a run.

        Args:
            logs ([object]): The log objects created after a run or read from a
                log pickle.
        """
        import matplotlib.pyplot as plt

        divergences = []
        values = []
        for log in logs:
            d = Log.calculate_divergence(log)
            v = [l[2] for l in log]
            divergences.append(d)
            values.append(v)

        for d in divergences:
            plt.plot(d)
        plt.title('Population Divergence')
        plt.ylabel('Average Distance from Center')
        plt.xlabel('Iteration')
        plt.show()

        for v in values:
            plt.plot(v)
        plt.title('Best Function Value')
        plt.ylabel('Function Value')
        plt.xlabel('Iteration')
        plt.show()
