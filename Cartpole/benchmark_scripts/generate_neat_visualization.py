"""Generates an interactive HTML/SVG visualization of the top 3 NEAT Triple Pendulum champions."""
import json, os

json_path = 'benchmark_results_topologies/top3_neat_triple.json'
out_html = 'benchmark_results_topologies/neat_triple_visualizer.html'

with open(json_path) as f:
    data = json.load(f)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top 3 NEAT Triple Pendulum Neural Topologies</title>
<style>
  :root {{
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-blue: #38bdf8;
    --accent-indigo: #818cf8;
    --accent-green: #34d399;
    --accent-red: #f87171;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    padding: 24px;
    line-height: 1.5;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}

  header {{
    margin-bottom: 24px;
    border-b: 1px solid var(--border-color);
    padding-bottom: 16px;
  }}

  h1 {{
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .badge {{
    font-size: 13px;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-blue);
    border: 1px solid rgba(56, 189, 248, 0.3);
    font-weight: 600;
  }}

  p.subtitle {{
    color: var(--text-secondary);
    font-size: 14px;
  }}

  .tabs {{
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
  }}

  .tab-btn {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 10px 18px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .tab-btn:hover {{
    border-color: var(--accent-blue);
    color: #fff;
  }}

  .tab-btn.active {{
    background: rgba(56, 189, 248, 0.15);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }}

  .stat-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
  }}

  .stat-label {{
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 4px;
  }}

  .stat-val {{
    font-size: 22px;
    font-weight: 700;
    color: #fff;
  }}

  .canvas-wrapper {{
    background: #090d16;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    position: relative;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
  }}

  svg {{
    width: 100%;
    height: 540px;
    display: block;
  }}

  .legend {{
    display: flex;
    gap: 20px;
    margin-top: 14px;
    font-size: 12px;
    color: var(--text-secondary);
    justify-content: center;
    align-items: center;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 12px;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .legend-color {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}

  .legend-line {{
    width: 24px;
    height: 3px;
    border-radius: 2px;
  }}

  .analysis-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
  }}

  .analysis-card h3 {{
    font-size: 16px;
    color: #fff;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .analysis-card p {{
    color: #cbd5e1;
    font-size: 14px;
    margin-bottom: 10px;
  }}

  .tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid var(--accent-blue);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    color: #fff;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    z-index: 100;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>
      <span>Evolved Topologies: Top 3 NEAT Triple Pendulum Champions</span>
      <span class="badge">Mode 3: θ₀ ~ U(-2.5°, +2.5°)</span>
    </h1>
    <p class="subtitle">Interactive DAG network inspection showing minimal structure, direct reflex pathways, and coupled non-linear hidden nodes.</p>
  </header>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchChamp(0)">Champion 1 (Rank 1: 3 Hidden Nodes, 100% Robust)</button>
    <button class="tab-btn" onclick="switchChamp(1)">Champion 2 (Rank 2: 6 Hidden Nodes, 85% Robust)</button>
    <button class="tab-btn" onclick="switchChamp(2)">Champion 3 (Rank 3: 6 Hidden Nodes, 29 Conns)</button>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Fitness Score</div>
      <div class="stat-val" id="stat-fitness">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Robustness Survival</div>
      <div class="stat-val" id="stat-robust" style="color: var(--accent-green)">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Generations to Solve</div>
      <div class="stat-val" id="stat-gens">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Hidden Nodes</div>
      <div class="stat-val" id="stat-hidden" style="color: var(--accent-blue)">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Active Connections</div>
      <div class="stat-val" id="stat-conns" style="color: var(--accent-indigo)">-</div>
    </div>
  </div>

  <div class="canvas-wrapper">
    <svg id="network-svg"></svg>
    <div class="legend">
      <div class="legend-item"><span class="legend-color" style="background:#38bdf8"></span> Input (8)</div>
      <div class="legend-item"><span class="legend-color" style="background:#fbbf24"></span> Bias (1)</div>
      <div class="legend-item"><span class="legend-color" style="background:#818cf8"></span> Hidden Node</div>
      <div class="legend-item"><span class="legend-color" style="background:#34d399"></span> Output (Motor Force)</div>
      <div class="legend-item" style="margin-left:16px"><span class="legend-line" style="background:#38bdf8"></span> Positive Weight (+)</div>
      <div class="legend-item"><span class="legend-line" style="background:#f87171"></span> Negative Weight (-)</div>
    </div>
  </div>

  <div class="analysis-card">
    <h3 id="analysis-title">Architectural Analysis</h3>
    <div id="analysis-body"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {json.dumps(data)};
let currentIdx = 0;

const analyses = [
  `<p><strong>The Differential Velocity Detector:</strong> Champion 1 achieved a <strong>100% out-of-distribution robustness</strong> survival rate with only <strong>3 hidden nodes and 16 weights</strong>.</p>
   <p>Notice how inputs <strong>Pole 2 Angular Velocity (5)</strong> and <strong>Pole 3 Angular Velocity (7)</strong> feed directly into <strong>Hidden Node 369</strong> with nearly equal and opposite weights (<strong>-11.54</strong> vs <strong>+11.87</strong>). This computes the relative shear angular velocity differential between the 2nd and 3rd links, which then passes through <strong>Hidden Node 409</strong> to produce a decisive <strong>-14.45 gain</strong> directly to Motor Force.</p>
   <p>Simultaneously, direct connections from <strong>Cart Velocity (+16.34)</strong> and <strong>Pole 1 Angle (-7.45)</strong> handle immediate base stabilization.</p>`,

  `<p><strong>Hierarchical Multi-Stage Balancing:</strong> Champion 2 reached <strong>84.6% robustness</strong> using 6 hidden nodes in an intricate multi-stage hierarchy.</p>
   <p>Rather than a single chain, it branches into two specialized clusters: Node 17 handles lower-joint damping from Pole 1 Angle (+0.84) and Cart Position (-1.60), while Nodes 234 &rarr; 291 &rarr; 533 form an angular acceleration damper connecting Pole 2 and Pole 3.</p>`,

  `<p><strong>High-Bandwidth Control Loop:</strong> Champion 3 reached solution fitness in only <strong>132 generations</strong> by evolving 29 connections across 6 hidden nodes.</p>
   <p>It contains multiple parallel pathways coupling Cart Position and Velocities through Nodes 11, 57, and 284, yielding high agility but lower generalization to extreme perturbation angles.</p>`
];

function switchChamp(idx) {{
  currentIdx = idx;
  document.querySelectorAll('.tab-btn').forEach((b, i) => {{
    b.classList.toggle('active', i === idx);
  }});
  renderChamp(idx);
}}

function renderChamp(idx) {{
  const c = DATA.champions[idx];
  document.getElementById('stat-fitness').innerText = c.fitness.toFixed(1);
  document.getElementById('stat-robust').innerText = c.robustness_pct.toFixed(1) + '%';
  document.getElementById('stat-gens').innerText = c.generations_to_solve;
  document.getElementById('stat-hidden').innerText = c.hidden_nodes_count;
  document.getElementById('stat-conns').innerText = c.enabled_connections_count;
  document.getElementById('analysis-title').innerText = `Architectural Analysis: Champion ${{idx + 1}}`;
  document.getElementById('analysis-body').innerHTML = analyses[idx];

  const svg = document.getElementById('network-svg');
  svg.innerHTML = '';
  const W = svg.clientWidth || 1160;
  const H = 540;

  // Determine positions
  const pos = {{}};
  const inputs = c.nodes.filter(n => n.type === 'INPUT');
  const bias = c.nodes.find(n => n.type === 'BIAS');
  const output = c.nodes.find(n => n.type === 'OUTPUT');
  const hidden = c.nodes.filter(n => n.type === 'HIDDEN');

  // Input column (X = 140)
  inputs.forEach((n, i) => {{
    pos[n.id] = {{ x: 150, y: 40 + i * (H - 120) / (inputs.length - 1), node: n }};
  }});

  // Bias node
  if (bias) {{
    pos[bias.id] = {{ x: 150, y: H - 35, node: bias }};
  }}

  // Output node (X = W - 140)
  if (output) {{
    pos[output.id] = {{ x: W - 140, y: H / 2, node: output }};
  }}

  // Hidden nodes (spread across middle X: 350 to W - 320)
  const hCols = Math.min(hidden.length, 3);
  hidden.forEach((n, i) => {{
    const col = i % hCols;
    const row = Math.floor(i / hCols);
    const totalRows = Math.ceil(hidden.length / hCols);
    const colX = 380 + (col * (W - 740) / Math.max(1, hCols - 1));
    const rowY = 80 + (row * (H - 160) / Math.max(1, totalRows));
    pos[n.id] = {{ x: colX, y: rowY + (col % 2 === 1 ? 40 : 0), node: n }};
  }});

  // Draw connections
  c.connections.forEach(cg => {{
    const p1 = pos[cg.from];
    const p2 = pos[cg.to];
    if (!p1 || !p2) return;

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const isPos = cg.weight >= 0;
    const strokeColor = isPos ? '#38bdf8' : '#f87171';
    const strokeW = Math.max(1.2, Math.min(7.0, Math.abs(cg.weight) * 0.45));
    const opacity = Math.max(0.3, Math.min(0.9, Math.abs(cg.weight) * 0.15 + 0.3));

    // Curved bezier path
    const dx = p2.x - p1.x;
    const pathD = `M ${{p1.x}} ${{p1.y}} C ${{p1.x + dx * 0.4}} ${{p1.y}}, ${{p2.x - dx * 0.4}} ${{p2.y}}, ${{p2.x}} ${{p2.y}}`;
    line.setAttribute('d', pathD);
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', strokeColor);
    line.setAttribute('stroke-width', strokeW);
    line.setAttribute('stroke-opacity', opacity);
    line.setAttribute('stroke-linecap', 'round');
    line.style.cursor = 'pointer';

    line.addEventListener('mouseenter', (e) => {{
      line.setAttribute('stroke-width', strokeW + 3);
      line.setAttribute('stroke-opacity', 1.0);
      showTooltip(e, `<b>${{p1.node.label}} &rarr; ${{p2.node.label}}</b><br>Weight: ${{cg.weight.toFixed(4)}}<br>Innovation: ${{cg.innovation}}`);
    }});
    line.addEventListener('mousemove', moveTooltip);
    line.addEventListener('mouseleave', () => {{
      line.setAttribute('stroke-width', strokeW);
      line.setAttribute('stroke-opacity', opacity);
      hideTooltip();
    }});

    svg.appendChild(line);
  }});

  // Draw nodes
  Object.values(pos).forEach(p => {{
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.style.cursor = 'pointer';

    let fillColor = '#818cf8';
    let r = 10;
    if (p.node.type === 'INPUT') {{ fillColor = '#38bdf8'; r = 9; }}
    else if (p.node.type === 'BIAS') {{ fillColor = '#fbbf24'; r = 8; }}
    else if (p.node.type === 'OUTPUT') {{ fillColor = '#34d399'; r = 14; }}
    else {{ fillColor = '#a855f7'; r = 11; }}

    // Halo glow
    const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    halo.setAttribute('cx', p.x);
    halo.setAttribute('cy', p.y);
    halo.setAttribute('r', r + 4);
    halo.setAttribute('fill', fillColor);
    halo.setAttribute('opacity', '0.2');
    g.appendChild(halo);

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', p.x);
    circle.setAttribute('cy', p.y);
    circle.setAttribute('r', r);
    circle.setAttribute('fill', fillColor);
    circle.setAttribute('stroke', '#0f172a');
    circle.setAttribute('stroke-width', '2.5');
    g.appendChild(circle);

    // Text Label
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('y', p.y + 4);
    text.setAttribute('fill', '#e2e8f0');
    text.setAttribute('font-size', '11px');
    text.setAttribute('font-weight', '600');
    text.setAttribute('font-family', 'sans-serif');

    if (p.node.type === 'INPUT' || p.node.type === 'BIAS') {{
      text.setAttribute('x', p.x - r - 8);
      text.setAttribute('text-anchor', 'end');
    }} else if (p.node.type === 'OUTPUT') {{
      text.setAttribute('x', p.x + r + 8);
      text.setAttribute('text-anchor', 'start');
    }} else {{
      text.setAttribute('x', p.x);
      text.setAttribute('y', p.y - r - 6);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#c084fc');
    }}
    text.textContent = p.node.label;
    g.appendChild(text);

    g.addEventListener('mouseenter', (e) => {{
      halo.setAttribute('r', r + 8);
      halo.setAttribute('opacity', '0.45');
      showTooltip(e, `<b>${{p.node.label}}</b><br>ID: ${{p.node.id}}<br>Type: ${{p.node.type}}`);
    }});
    g.addEventListener('mousemove', moveTooltip);
    g.addEventListener('mouseleave', () => {{
      halo.setAttribute('r', r + 4);
      halo.setAttribute('opacity', '0.2');
      hideTooltip();
    }});

    svg.appendChild(g);
  }});
}}

const tooltip = document.getElementById('tooltip');
function showTooltip(e, html) {{
  tooltip.innerHTML = html;
  tooltip.style.opacity = '1';
  moveTooltip(e);
}}
function moveTooltip(e) {{
  const wrapper = document.querySelector('.canvas-wrapper').getBoundingClientRect();
  tooltip.style.left = (e.clientX - wrapper.left + 15) + 'px';
  tooltip.style.top = (e.clientY - wrapper.top + 15) + 'px';
}}
function hideTooltip() {{
  tooltip.style.opacity = '0';
}}

window.addEventListener('resize', () => renderChamp(currentIdx));
renderChamp(0);
</script>
</body>
</html>
"""

with open(out_html, 'w') as f:
    f.write(html_content)

print(f"Generated {out_html}")
