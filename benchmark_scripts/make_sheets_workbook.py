"""Builds a Google-Sheets-friendly workbook (.xlsx, stdlib only) plus CSV copies from benchmark output.

Usage: python3 benchmark_scripts/make_sheets_workbook.py [results_dir]
"""
import csv, os, sys, zipfile, statistics
from collections import defaultdict
from xml.sax.saxutils import escape

D = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results_topologies'
REG = {
    'zero': 'Fixed 0 deg (degenerate)',
    'random15': 'Random +/-15 deg',
    'scaled': 'Scaled Range (1:±10°, 2:±5°, 3:±2.5°)'
}
POLES = {1: 'Single', 2: 'Double', 3: 'Triple'}
MAX_CURVE_GENS = 400


def num(x, nd=2):
    if x in ('NA', '', None):
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def algo_sort_key(algo):
    if algo == 'NEAT':
        return (-1, 0, 0)
    if algo.startswith('GA '):
        parts = algo[3:].split('x')
        try:
            return (0, int(parts[0]), int(parts[1]))
        except Exception:
            return (0, 99, 99)
    return (1, 0, 0)


def get_topology_spec(algo, mode):
    if algo == 'NEAT':
        return 'Evolved Minimal DAG'
    if not algo.startswith('GA '):
        return algo
    parts = algo[3:].split('x')
    w, d = int(parts[0]), int(parts[1])
    in_count = 4 + (mode - 1) * 2
    p = in_count * w + w
    for _ in range(d - 1):
        p += w * w + w
    p += w * 1 + 1
    return f'{w}w x {d}d ({p} weights)'


def read(name):
    path = os.path.join(D, name)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


summary = read('summary.csv')
trials = read('trials.csv')
robust = read('robustness.csv')
gens = read('generations.csv')

# ── Tab: Summary ────────────────────────────────────────────────────────────
summary.sort(key=lambda r: (r['regime'] == 'zero', int(r['mode']), algo_sort_key(r['algo'])))
sum_rows = [[
    'Poles', 'Start-angle regime', 'Algorithm', 'Trials', 'Solved (reached 90% of max)',
    'Solved %', 'Gens to 90% (mean)', 'Gens to 90% (sd)', 'Sim frames to 90% (millions)',
    'Typical final strength (last-10-gen best, mean)', 'Strength sd across trials',
    'Generations run (mean)', 'CPU seconds per run (mean)', 'Champion hidden nodes (mean)',
    'Champion survives fixed test angles (%)', 'How to read this row']]
for r in summary:
    reg = r['regime']
    note = ('Do-nothing exploit: zero force survives from exact upright; not real balancing'
            if reg == 'zero' else 'Meaningful: must recover from varied starting perturbations')
    solved = int(r['n_reached_90pct']); n = int(r['n_trials'])
    f90 = num(r['frames_to_90pct_mean'], 0)
    sum_rows.append([
        POLES[int(r['mode'])], REG.get(reg, reg), r['algo'], n, solved, round(100.0 * solved / n, 0),
        num(r['gen_to_90pct_mean'], 1), num(r['gen_to_90pct_sd'], 1),
        None if f90 is None else round(f90 / 1e6, 2),
        num(r['last10_best_mean_mean'], 0), num(r['last10_best_mean_sd'], 0),
        num(r['gens_run_mean'], 0), num(r['cpu_s_mean'], 1), num(r['champ_hidden_nodes_mean'], 1),
        round(100 * float(r['robust_final_survive_frac_mean']), 0), note])

# ── Pre-aggregate Robustness & Curves ───────────────────────────────────────
agg = defaultdict(lambda: defaultdict(list))
rob_fitness_map = defaultdict(list)
for r in robust:
    if r['which'] != 'final':
        continue
    m = int(r['mode'])
    reg = r['regime']
    algo = r['algo']
    angle = float(r['test_angle'])
    agg[(reg, m, algo)][angle].append(int(r['frames_survived']))
    rob_fitness_map[(reg, m, algo)].append(float(r['fitness']))

rob_rows = [['Poles', 'Training regime', 'Algorithm', 'Test Point / Multiplier', 'Test Angle (deg)',
             'Survives Full 1800 Frames (%)', 'Status']]
multipliers = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
for key in sorted(agg, key=lambda k: (k[0] == 'zero', k[1], algo_sort_key(k[2]))):
    reg, mode, algo = key
    angles_sorted = sorted(agg[key].keys())
    for i, a in enumerate(angles_sorted):
        v = agg[key][a]
        pct = round(100.0 * sum(1 for x in v if x >= 1800) / max(1, len(v)), 1)
        mult_label = f'{multipliers[i]:+.1f}x training bound' if i < len(multipliers) else ''
        status = 'In-bounds (interpolation)' if abs(a) <= (10.0 if mode == 1 else (5.0 if mode == 2 else 2.5)) else 'Out-of-bounds (generalization)'
        rob_rows.append([POLES[mode], REG.get(reg, reg), algo, mult_label, round(a, 3), pct, status])

# ── Tab: Robustness Matrix ──────────────────────────────────────────────────
mat_rows = [['Poles', 'Training regime', 'Algorithm',
             '-1.5x (Out)', '-1.2x (Out)', '-1.0x (Edge)', '-0.8x (In)', '-0.5x (In)', '-0.2x (In)',
             '0.0 deg (Center)',
             '+0.2x (In)', '+0.5x (In)', '+0.8x (In)', '+1.0x (Edge)', '+1.2x (Out)', '+1.5x (Out)']]
for key in sorted(agg, key=lambda k: (k[0] == 'zero', k[1], algo_sort_key(k[2]))):
    reg, mode, algo = key
    angles_sorted = sorted(agg[key].keys())
    row = [POLES[mode], REG.get(reg, reg), algo]
    for a in angles_sorted:
        v = agg[key][a]
        pct = round(100.0 * sum(1 for x in v if x >= 1800) / max(1, len(v)), 0)
        row.append(pct)
    mat_rows.append(row)

# ── Learning curves dictionary ──────────────────────────────────────────────
curves = defaultdict(lambda: defaultdict(list))
for r in gens:
    curves[(r['regime'], int(r['mode']), r['algo'])][r['trial_id']].append((int(r['gen']), float(r['best'])))


def curve_tab_by_mode(mode):
    cells = sorted([k for k in curves if k[1] == mode], key=lambda k: algo_sort_key(k[2]))
    series = {}
    longest = 0
    for k in cells:
        per_trial = []
        for tid, lst in curves[k].items():
            lst.sort()
            per_trial.append([b for _, b in lst])
        if per_trial:
            longest = max(longest, max(len(t) for t in per_trial))
        series[k] = per_trial
    G = min(longest, MAX_CURVE_GENS)
    rows = [['Generation'] + [k[2] for k in cells]]
    for g in range(G):
        row = [g + 1]
        for k in cells:
            tr = series[k]
            vals = [t[g] if g < len(t) else t[-1] for t in tr]
            row.append(round(sum(vals) / max(1, len(vals)), 1))
        rows.append(row)
    return rows


# ── Top Performer Tabs Builder ──────────────────────────────────────────────
def build_top_mode_tab(mode, top_n_ga):
    """Builds a dedicated executive tab for a mode: Leaderboard + Robustness Profile + Curves."""
    m_trials = [t for t in trials if t['mode'] == str(mode) and t['regime'] == 'scaled']
    by_algo = defaultdict(list)
    for t in m_trials:
        by_algo[t['algo']].append(t)

    # Rank GA algorithms
    ga_ranked = []
    for algo, ts in by_algo.items():
        if algo == 'NEAT':
            continue
        solved = [t for t in ts if int(t['gen_to_90pct']) > 0]
        rate = len(solved) / len(ts) * 100
        gens_list = [int(t['gen_to_90pct']) for t in solved]
        mean_g = statistics.mean(gens_list) if gens_list else 999
        rob_surv = statistics.mean([float(t['robust_bestever_survive_frac']) for t in ts]) * 100
        rob_fit = statistics.mean([float(t['robust_bestever_mean_fitness']) for t in ts])
        ga_ranked.append((algo, rate, mean_g, rob_surv, rob_fit))

    if mode == 1:
        # Mode 1: almost all solve 100%, rank by robustness & speed
        ga_ranked.sort(key=lambda x: (x[1], x[3], x[4], -x[2]), reverse=True)
    else:
        # Modes 2 & 3: rank by solve rate, then fewest gens to solve, then robustness
        ga_ranked.sort(key=lambda x: (x[1], -x[2], x[3], x[4]), reverse=True)

    top_ga_names = [r[0] for r in ga_ranked[:top_n_ga]]
    selected_algos = ['NEAT'] + top_ga_names

    bound_deg = 10.0 if mode == 1 else (5.0 if mode == 2 else 2.5)
    rows = []
    header_rows = set()

    # Section 1: Leaderboard
    rows.append([f'{POLES[mode]} Pendulum Benchmark: NEAT vs Top {top_n_ga} GA Topologies'])
    header_rows.add(len(rows))
    rows.append([f'Initial Disturbance: Uniform [-{bound_deg:.1f}°, +{bound_deg:.1f}°], Population: 150, Max Frames: 1,800'])
    header_rows.add(len(rows))
    rows.append([])

    lb_headers = [
        'Rank', 'Algorithm', 'Architecture Spec', 'Trials', 'Solved (>=90%)',
        'Solve Rate (%)', 'Gens to 90% (Mean)', 'Gens to 90% (Median)',
        'Gens to 90% (Min)', 'Gens to 90% (Max)', 'Gens to 90% (SD)',
        'Frames to 90% (M)', 'Mean Gens Run', 'Last-10-Gen Strength',
        'Champ Hidden Nodes', 'Champ Enabled Weights',
        'Test Angle Survival (%)', 'Robustness Mean Fitness', 'CPU Seconds (Mean)'
    ]
    rows.append(lb_headers)
    header_rows.add(len(rows))

    for idx, algo in enumerate(selected_algos):
        ts = by_algo.get(algo, [])
        if not ts:
            continue
        n = len(ts)
        solved = [t for t in ts if int(t['gen_to_90pct']) > 0]
        n_sol = len(solved)
        rate = round(100.0 * n_sol / n, 1)
        gens_list = [int(t['gen_to_90pct']) for t in solved]
        mean_g = round(statistics.mean(gens_list), 1) if gens_list else None
        med_g = round(statistics.median(gens_list), 1) if gens_list else None
        min_g = min(gens_list) if gens_list else None
        max_g = max(gens_list) if gens_list else None
        sd_g = round(statistics.stdev(gens_list), 1) if len(gens_list) > 1 else 0.0 if gens_list else None
        f90_list = [int(t['frames_to_90pct']) for t in solved if int(t['frames_to_90pct']) > 0]
        f90_m = round(statistics.mean(f90_list) / 1e6, 2) if f90_list else None
        gens_run_m = round(statistics.mean([int(t['gens_run']) for t in ts]), 1)
        last10_m = round(statistics.mean([float(t['last10_best_mean']) for t in ts]), 1)
        hn_m = round(statistics.mean([int(t['champ_hidden_nodes']) for t in ts]), 1)
        conns_m = round(statistics.mean([int(t['champ_enabled_conns']) for t in ts]), 1)
        surv_pct = round(100.0 * statistics.mean([float(t['robust_bestever_survive_frac']) for t in ts]), 1)
        fit_m = round(statistics.mean([float(t['robust_bestever_mean_fitness']) for t in ts]), 1)
        cpu_m = round(statistics.mean([float(t['cpu_s']) for t in ts]), 1)
        rank_str = 'Benchmark (Winner)' if algo == 'NEAT' else f'GA #{idx}'
        arch_spec = get_topology_spec(algo, mode)
        rows.append([
            rank_str, algo, arch_spec, n, n_sol, rate,
            mean_g, med_g, min_g, max_g, sd_g,
            f90_m, gens_run_m, last10_m, hn_m, conns_m,
            surv_pct, fit_m, cpu_m
        ])

    rows.append([])
    rows.append([])

    # Section 2: Robustness Profile
    rows.append(['Out-of-Distribution Robustness Profile: Survival Rate (%) Across Fixed Perturbation Angles'])
    header_rows.add(len(rows))
    rob_col_headers = [
        'Algorithm', '-1.5x (Out)', '-1.2x (Out)', '-1.0x (Edge)', '-0.8x (In)', '-0.5x (In)', '-0.2x (In)',
        '0.0° (Center)',
        '+0.2x (In)', '+0.5x (In)', '+0.8x (In)', '+1.0x (Edge)', '+1.2x (Out)', '+1.5x (Out)',
        'Overall Survival (%)', 'Robust Mean Fitness'
    ]
    rows.append(rob_col_headers)
    header_rows.add(len(rows))

    for algo in selected_algos:
        key = ('scaled', mode, algo)
        angles_sorted = sorted(agg[key].keys())
        r_row = [algo]
        for a in angles_sorted:
            v = agg[key][a]
            pct = round(100.0 * sum(1 for x in v if x >= 1800) / max(1, len(v)), 0)
            r_row.append(pct)
        ts = by_algo.get(algo, [])
        surv_pct = round(100.0 * statistics.mean([float(t['robust_bestever_survive_frac']) for t in ts]), 1) if ts else None
        fit_m = round(statistics.mean([float(t['robust_bestever_mean_fitness']) for t in ts]), 1) if ts else None
        r_row.extend([surv_pct, fit_m])
        rows.append(r_row)

    rows.append([])
    rows.append([])

    # Section 3: Learning Curves (Ready for Charting)
    rows.append(['Learning Curves: Mean Best-Ever Fitness per Generation (Ready for Line Charting)'])
    header_rows.add(len(rows))
    curve_headers = ['Generation'] + selected_algos
    rows.append(curve_headers)
    header_rows.add(len(rows))

    series = {}
    longest = 0
    for algo in selected_algos:
        k = ('scaled', mode, algo)
        per_trial = []
        for tid, lst in curves[k].items():
            lst.sort()
            per_trial.append([b for _, b in lst])
        if per_trial:
            longest = max(longest, max(len(t) for t in per_trial))
        series[algo] = per_trial

    G = min(longest, MAX_CURVE_GENS)
    for g in range(G):
        c_row = [g + 1]
        for algo in selected_algos:
            tr = series.get(algo, [])
            if not tr:
                c_row.append(None)
                continue
            vals = [t[g] if g < len(t) else t[-1] for t in tr]
            c_row.append(round(sum(vals) / max(1, len(vals)), 1))
        rows.append(c_row)

    return rows, header_rows


# ── Tab: Trials (raw rows for pivot tables) ─────────────────────────────────
rob_by_trial = {}
for r in robust:
    if r['which'] == 'final':
        rob_by_trial.setdefault(r['trial_id'], []).append(int(r['frames_survived']))
trial_rows = [['Trial', 'Poles', 'Start-angle regime', 'Algorithm', 'Repeat', 'Generations run',
               'Gen reached 90% (blank=never)', 'Best-ever fitness',
               'Last-10-gen best (mean)', 'CPU seconds', 'Sim frames (millions)',
               'Champion hidden nodes', 'Survives test angles (%)']]
for r in sorted(trials, key=lambda r: (r['regime'] == 'zero', int(r['mode']), algo_sort_key(r['algo']), int(r['rep']))):
    g90 = int(r['gen_to_90pct'])
    surv = rob_by_trial.get(r['trial_id'], [])
    trial_rows.append([
        r['trial_id'], POLES[int(r['mode'])], REG.get(r['regime'], r['regime']), r['algo'], int(r['rep']),
        int(r['gens_run']), None if g90 < 0 else g90, num(r['best_ever'], 1), num(r['last10_best_mean'], 1),
        num(r['cpu_s'], 1), round(int(r['agent_frames']) / 1e6, 2), int(r['champ_hidden_nodes']),
        round(100.0 * sum(1 for x in surv if x >= 1800) / max(1, len(surv)), 0)])

# ── Tab: README ─────────────────────────────────────────────────────────────
readme = [
    ['Cartpole Benchmark: Topologies & Algorithms Comparison'],
    [''],
    ['Tab', 'What it shows'],
    ['Summary', 'Overall results per configuration. Comparing NEAT vs 63 GA topologies across all modes.'],
    ['Top Single (NEAT + 10 GA)', 'Executive leaderboard, robustness matrix, and ready-to-chart learning curves for Single Pendulum (NEAT vs Top 10 GAs).'],
    ['Top Double (NEAT + 10 GA)', 'Executive leaderboard, robustness matrix, and ready-to-chart learning curves for Double Pendulum (NEAT vs Top 10 GAs).'],
    ['Top Triple (NEAT + 5 GA)', 'Executive leaderboard, robustness matrix, and ready-to-chart learning curves for Triple Pendulum (NEAT vs Top 5 GAs).'],
    ['Robustness Matrix', 'Survival rates (%) tested from -1.5x up to +1.5x of training disturbance bound.'],
    ['Robustness Details', 'Per-angle survival rates and in-bounds vs generalization tagging.'],
    ['Curves Single', 'Complete learning curves for Single Pendulum (all 64 algorithms).'],
    ['Curves Double', 'Complete learning curves for Double Pendulum (all 64 algorithms).'],
    ['Curves Triple', 'Complete learning curves for Triple Pendulum (all 64 algorithms).'],
    ['Trials', 'Raw trial executions (5,526 rows) for Pivot Table analysis.'],
    [''],
    ['Experimental Design'],
    ['Single Pendulum (Mode 1)', '10 trials per GA topology, 20 trials for NEAT, starting angle ~ U(-10 deg, +10 deg)'],
    ['Double Pendulum (Mode 2)', '25 trials per GA topology, 50 trials for NEAT, starting angle ~ U(-5 deg, +5 deg)'],
    ['Triple Pendulum (Mode 3)', '50 trials per GA topology, 100 trials for NEAT, starting angle ~ U(-2.5 deg, +2.5 deg)'],
    ['Topologies compared', 'NEAT, plus GA with widths (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 100) and depths (1, 2, 3), plus depths (4, 5) for widths 1-16'],
    ['Population size', '150 individuals in every generation across all runs'],
    ['Stopping rule', 'Patient plateau detection (window 25, threshold 0.5%, patience 20, min 250 gens, max 500)'],
]

top_single_rows, top_single_hdr = build_top_mode_tab(1, 10)
top_double_rows, top_double_hdr = build_top_mode_tab(2, 10)
top_triple_rows, top_triple_hdr = build_top_mode_tab(3, 5)

SHEETS = [
    ('README', readme, {1, 3, 15}),
    ('Summary', sum_rows, {1}),
    ('Top Single (NEAT + 10 GA)', top_single_rows, top_single_hdr),
    ('Top Double (NEAT + 10 GA)', top_double_rows, top_double_hdr),
    ('Top Triple (NEAT + 5 GA)', top_triple_rows, top_triple_hdr),
    ('Robustness Matrix', mat_rows, {1}),
    ('Robustness Details', rob_rows, {1}),
    ('Curves Single', curve_tab_by_mode(1), {1}),
    ('Curves Double', curve_tab_by_mode(2), {1}),
    ('Curves Triple', curve_tab_by_mode(3), {1}),
    ('Trials', trial_rows, {1}),
]


# ── Minimal xlsx writer ─────────────────────────────────────────────────────
def col(n):
    s = ''
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def sheet_xml(rows, header_rows):
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           '<cols><col min="1" max="25" width="22" customWidth="1"/></cols><sheetData>']
    for ri, row in enumerate(rows, start=1):
        out.append(f'<row r="{ri}">')
        style = ' s="1"' if ri in header_rows else ''
        for ci, v in enumerate(row):
            ref = f'{col(ci)}{ri}'
            if v is None or v == '':
                continue
            if isinstance(v, (int, float)):
                out.append(f'<c r="{ref}"{style}><v>{v}</v></c>')
            else:
                out.append(f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>')
        out.append('</row>')
    out.append('</sheetData></worksheet>')
    return ''.join(out)


def write_xlsx(path):
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
           '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
           '<Default Extension="xml" ContentType="application/xml"/>'
           '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
           '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i, item in enumerate(SHEETS, start=1):
        name = item[0]
        ct.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        wb.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
    n = len(SHEETS) + 1
    rels.append(f'<Relationship Id="rId{n}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    ct.append('</Types>'); wb.append('</sheets></workbook>'); rels.append('</Relationships>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
              '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
              '<fills count="2"><fill><patternFill patternType="none"/></fill>'
              '<fill><patternFill patternType="gray125"/></fill></fills>'
              '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
              '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
              '</styleSheet>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                 '</Relationships>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ''.join(ct))
        z.writestr('_rels/.rels', root_rels)
        z.writestr('xl/workbook.xml', ''.join(wb))
        z.writestr('xl/_rels/workbook.xml.rels', ''.join(rels))
        z.writestr('xl/styles.xml', styles)
        for i, item in enumerate(SHEETS, start=1):
            name, rows = item[0], item[1]
            header_rows = item[2] if len(item) > 2 else {1}
            z.writestr(f'xl/worksheets/sheet{i}.xml', sheet_xml(rows, header_rows))


out_xlsx = os.path.join(D, 'cartpole_benchmark_for_sheets.xlsx')
write_xlsx(out_xlsx)
csv_dir = os.path.join(D, 'sheets_csv')
os.makedirs(csv_dir, exist_ok=True)
for item in SHEETS:
    name, rows = item[0], item[1]
    with open(os.path.join(csv_dir, name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus') + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        for row in rows:
            w.writerow(['' if v is None else v for v in row])
print('wrote', out_xlsx)
print('csv copies in', csv_dir)
