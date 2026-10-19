import csv, sys
d = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results'
rows = list(csv.DictReader(open(f'{d}/summary.csv')))
keys = ['gens_run_mean', 'cpu_s_mean', 'best_ever_mean', 'last10_best_mean_mean', 'gen_to_50pct_mean',
        'gen_to_90pct_mean', 'robust_final_mean_fitness_mean', 'robust_final_survive_frac_mean',
        'champ_hidden_nodes_mean']
print('mode regime   algo n plat r90 | ' + ' | '.join(k.replace('_mean', '') for k in keys))
for r in rows:
    vals = []
    for k in keys:
        v = r[k]
        vals.append(v if v == 'NA' else f'{float(v):.2f}')
    print(r['mode'], r['regime'].ljust(8), r['algo'].ljust(4), r['n_trials'], r['n_plateau_stop'],
          r['n_reached_90pct'], '|', ' | '.join(vals))
