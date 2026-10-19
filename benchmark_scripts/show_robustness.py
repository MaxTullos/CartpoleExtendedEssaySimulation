import csv, sys
from collections import defaultdict
d = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results_patient'
which = sys.argv[2] if len(sys.argv) > 2 else 'final'
agg = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f'{d}/robustness.csv')):
    if r['which'] != which:
        continue
    key = (int(r['mode']), r['regime'], r['algo'])
    agg[key][float(r['test_angle'])].append(int(r['frames_survived']))
angles = sorted({a for k in agg for a in agg[k]})
print('mean fraction of trials surviving full 1800 frames, by test angle (deg), brain=' + which)
print('mode regime   algo  ' + ' '.join(f'{a:+4.0f}' for a in angles))
for key in sorted(agg):
    cells = []
    for a in angles:
        v = agg[key][a]
        cells.append(f'{sum(1 for x in v if x >= 1800) / len(v):4.1f}')
    print(key[0], key[1].ljust(8), key[2].ljust(5), ' '.join(cells))
