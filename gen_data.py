import random

points = 100000
features = 1000
avg_len = 10

for i in xrange(points):
    sign = 1 if random.randrange(2) == 0 else -1
    fs = []
    for j in xrange(features):
        if random.randrange(features) < avg_len:
            s = sign * (1 if j % 2 == 0 else -1)
            fs.append((j, s * random.random()))
    print sign, ' '.join([str(a) + ':' + str(b) for a, b in fs])
                
