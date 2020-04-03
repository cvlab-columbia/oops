import csv
import json
import os
import posixpath
import random
from glob import glob

basepath = 'YOUR PATH HERE'
data = []
for fn in glob(posixpath.join(basepath, '*.csv')):
    with open(fn) as f:
        data.extend(csv.DictReader(f))
for d in data:
    d['fn'] = os.path.join(os.path.basename(os.path.dirname(d['fn'])), os.path.basename(d['fn']))
    for k in ('y', 'y_hat'):
        d[k] = int(d[k])
    for k in ('t_start', 't_end'):
        d[k] = float(d[k])
    d['y_hat_vec'] = eval(d['y_hat_vec'])
    if 'x' in d:
        d['x'] = eval(d['x'])
random.shuffle(data)
data = [d for d in data if d['y'] >= 0]
with open(os.path.join(basepath, 'fails_processed.json'),
          'w') as f:
    json.dump(data, f)
pass
