#!/usr/bin/env python3

import csv

with open("faithful.csv", 'w') as c:
    with open("faithful.txt", "r") as f:
        for line in f.readlines()[1:]:
            x = line.split('      ')
            c.write('%s\n' % x[0])