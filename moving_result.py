import matplotlib.pyplot as plt
from collections import deque

class Params():
    """Moving result associated parameters

    """
    def __init__(self, d=None):
        self.params = {}
        if d is not None:
            if type(d) is str:
                p = d.split('/')
                for i in range(len(p)):
                    s = p[i].split(':')
                    self.params[s[0]] = float(s[1])
            elif type(d) is dict:
                self.params = d

    def add(self, name, value):
        self.params[name] = value 

    def get(self,name):
        if name not in self.params:
            return None
        else: 
            return self.params[name]

    def __repr__(self):
        s = ''
        if self.params is None:
            return s
        for k,v in self.params.items():
            s += f'{k}:{v}/'
        return s[:-1] 

class MovingResult():
    """Manage moving results easily with built-in plot.
    
    """
    def __init__(self, size=100, name='score', save_raw=True, restore=None, params=None):  
        if restore is not None:
            self.restore(restore)
        else:
            self.save_raw = save_raw
            self.params = Params(params)
            self.buffer_size = size
            self.name = name.capitalize()
            self.reset()

    def add(self, y, it=None):
        if it is None:
            it = 1 
            if len(self.buffer) > 0:
                it = self.buffer[-1][0] + 1  
        self.buffer.append((it,y))
        self.buffer_window.append((it,y))
        if len(self.buffer_window) >= self.buffer_size:
            y = np.mean([v[1] for v in self.buffer_window])
            self.buffer_moving.append((self.buffer_window[-1][0], y))
            
    def plot(self):
        x_moving, y_moving = zip(*self.buffer_moving)

        plt.figure(figsize=(10,5), dpi=80)
        if self.save_raw:
            x,y = zip(*self.buffer)
            plt.plot(list(x), list(y), alpha=0.2, label=self.name.capitalize())
            
        plt.plot(list(x_moving), list(y_moving), label=f'Moving {self.name} ({self.buffer_size})')
        plt.legend()
        plt.xlabel('Iteration #')
        plt.ylabel(f'{self.name}')
        plt.show()
    
    def reset(self):
        self.buffer_window = deque(maxlen=self.buffer_size)
        if self.save_raw:
            self.buffer = []
        else:
            self.buffer = deque(maxlen=1)
        self.buffer_moving = []
        
    def restore(self,filename):
        if not os.path.exists(filename):
            return None
        with open(filename) as file:
            line = file.readline() # params 
            s = line.split(',')
            self.name = s[0]
            self.buffer_size = int(s[1])
            self.save_raw = True if s[2].rstrip() == 'True' else False
            self.params = Params(s[3].rstrip()) if len(s) > 3 else Params()
            self.reset()
            line = file.readline()
            while not line.startswith('MOVING'):
                s = line.split(',')
                self.buffer.append((int(s[0]),float(s[1])))
                line = file.readline()
            line = file.readline()
            while line:
                s = line.split(',')
                self.buffer_moving.append((int(s[0]),float(s[1])))
                line = file.readline()
                
    def save(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filename, "w") as file:
            file.write(f'{self.name},{self.buffer_size},{self.save_raw},{self.params}\n')
            for x,y in self.buffer:
                file.write(f'{x},{y}\n')
            file.write('MOVING\n')
            for x,y in self.buffer_moving:
                file.write(f'{x},{y}\n')
                
    @property
    def x(self):
        return [p[0] for p in self.buffer_moving]
    
    @property
    def y(self):
        return [p[1] for p in self.buffer_moving]

    @property
    def last(self):
        if len(self.buffer_window) < self.buffer_size:
            return 0.0
        return self.buffer_moving[-1][1]