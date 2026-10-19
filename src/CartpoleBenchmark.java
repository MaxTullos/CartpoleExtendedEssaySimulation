import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.*;
import java.util.concurrent.*;

/**
 * Headless benchmark: Fixed-topology GA vs NEAT on the Cartpole simulation.
 *
 * Reuses CartAgent / NeuralNetwork / NEATGenome / NEATSpecies unchanged and replicates the
 * evolution logic of Cartpole.java (evolveFixedTopology / evolveNEAT) without any Swing.
 *
 * Training stops when progress plateaus (see PLATEAU settings) or maxGens is hit.
 *
 * Usage (all optional, key=value):
 *   trials=5 threads=7 out=benchmark_results maxGens=400 pop=150 maxFrames=1800
 *   algos=GA,NEAT modes=1,2,3 regimes=zero,random15 hidden=6
 *   window=10 eps=0.02 patience=5 minGens=30
 */
public class CartpoleBenchmark {

    // ── Context handed to CartAgent ──────────────────────────────────────────
    static class Ctx implements SimulationContext {
        final int mode;
        volatile double angle;
        Ctx(int mode) { this.mode = mode; }
        public int pendulumMode() { return mode; }
        public boolean useAlternativeFitness() { return false; }
        public boolean useCenterReward() { return true; }
        public double getCurrentGenAngleDeg() { return angle; }
    }

    // ── Settings (GUI defaults unless overridden) ────────────────────────────
    static int threads = 7, maxGens = 500, popSize = 150, maxFrames = 1800;
    static String outDir = "benchmark_results_topologies";
    static List<String> algos = buildDefaultAlgos();
    static List<Integer> modes = Arrays.asList(1, 2, 3);
    static List<String> regimes = Arrays.asList("scaled");
    static int trialsMode1 = 10, trialsMode2 = 25, trialsMode3 = 50;

    // GA (GUI defaults)
    static double elitismPct = 5.0, mutationPct = 5.0, mutMag = 0.5;
    static int tourney = 4;
    // NEAT (GUI defaults)
    static double compat = 3.0, addNode = 0.03, addConn = 0.05, wMut = 0.80, xover = 0.75, survive = 0.20;
    static int stagnation = 15;
    // Patient plateau detection defaults (increased bar for GA Mode 3)
    static int window = 25, patience = 20, minGens = 250;
    static double eps = 0.005;

    static List<String> buildDefaultAlgos() {
        List<String> list = new ArrayList<>();
        list.add("NEAT");
        int[] widths = {1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 100};
        int[] depths = {1, 2, 3};
        for (int w : widths) {
            for (int d : depths) {
                list.add("GA " + w + "x" + d);
            }
        }
        int[] widths1to16 = {1, 2, 4, 6, 8, 10, 12, 14, 16};
        int[] deepDepths = {4, 5};
        for (int w : widths1to16) {
            for (int d : deepDepths) {
                list.add("GA " + w + "x" + d);
            }
        }
        return list;
    }

    static int[] getTopologyForAlgo(String algo, int inputCount) {
        if (algo.equals("NEAT")) return null;
        String spec = algo.startsWith("GA ") ? algo.substring(3).trim() : algo.trim();
        String[] parts = spec.split("x");
        int width = Integer.parseInt(parts[0].trim());
        int depth = Integer.parseInt(parts[1].trim());
        int[] t = new int[depth + 2];
        t[0] = inputCount;
        for (int i = 0; i < depth; i++) {
            t[i + 1] = width;
        }
        t[t.length - 1] = 1;
        return t;
    }

    static int paramCount(int[] t) {
        int p = 0;
        for (int i = 0; i < t.length - 1; i++) {
            p += t[i] * t[i + 1] + t[i + 1];
        }
        return p;
    }

    static double getRangeMax(int mode) {
        switch (mode) {
            case 1: return 10.0;
            case 2: return 5.0;
            case 3: return 2.5;
            default: return 10.0;
        }
    }

    static int getTrialsForAlgoMode(String algo, int mode) {
        int base;
        switch (mode) {
            case 1: base = trialsMode1; break;
            case 2: base = trialsMode2; break;
            case 3: base = trialsMode3; break;
            default: base = trialsMode1; break;
        }
        if (algo.equals("NEAT")) {
            return base * 2;
        }
        return base;
    }

    static double[] getTestAngles(int mode) {
        double b = getRangeMax(mode);
        return new double[]{
            -1.5 * b, -1.2 * b, -1.0 * b, -0.8 * b, -0.5 * b, -0.2 * b,
            0.0,
            0.2 * b, 0.5 * b, 0.8 * b, 1.0 * b, 1.2 * b, 1.5 * b
        };
    }

    static final ThreadMXBean MX = ManagementFactory.getThreadMXBean();
    static PrintWriter trialsW, gensW, robW;

    public static void main(String[] args) throws Exception {
        for (String a : args) {
            String[] kv = a.split("=", 2);
            if (kv.length != 2) continue;
            String v = kv[1];
            switch (kv[0]) {
                case "threads": threads = Integer.parseInt(v); break;
                case "out": outDir = v; break;
                case "maxGens": maxGens = Integer.parseInt(v); break;
                case "pop": popSize = Integer.parseInt(v); break;
                case "maxFrames": maxFrames = Integer.parseInt(v); break;
                case "algos": algos = Arrays.asList(v.split(",")); break;
                case "modes": modes = new ArrayList<>(); for (String s : v.split(",")) modes.add(Integer.parseInt(s)); break;
                case "regimes": regimes = Arrays.asList(v.split(",")); break;
                case "trials1": trialsMode1 = Integer.parseInt(v); break;
                case "trials2": trialsMode2 = Integer.parseInt(v); break;
                case "trials3": trialsMode3 = Integer.parseInt(v); break;
                case "window": window = Integer.parseInt(v); break;
                case "eps": eps = Double.parseDouble(v); break;
                case "patience": patience = Integer.parseInt(v); break;
                case "minGens": minGens = Integer.parseInt(v); break;
                default: System.err.println("Unknown arg " + a);
            }
        }
        new File(outDir).mkdirs();
        File tf = new File(outDir, "trials.csv");
        boolean exists = tf.exists() && tf.length() > 0;
        Set<String> completed = Collections.synchronizedSet(new HashSet<>());
        if (exists) {
            try (BufferedReader br = new BufferedReader(new FileReader(tf))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length > 0) completed.add(parts[0]);
                }
            }
        }

        trialsW = new PrintWriter(new FileWriter(tf, true));
        gensW = new PrintWriter(new FileWriter(new File(outDir, "generations.csv"), true));
        robW = new PrintWriter(new FileWriter(new File(outDir, "robustness.csv"), true));
        if (!exists) {
            trialsW.println("trial_id,algo,mode,regime,rep,gens_run,stop_reason,wall_s,cpu_s,agent_frames,"
                    + "best_ever,best_ever_gen,final_gen_best,final_gen_mean,last10_best_mean,"
                    + "gen_to_50pct,gen_to_90pct,frames_to_90pct,champ_hidden_nodes,champ_enabled_conns,"
                    + "robust_final_mean_fitness,robust_final_survive_frac,robust_bestever_mean_fitness,robust_bestever_survive_frac");
            gensW.println("trial_id,algo,mode,regime,gen,angle_deg,best,q3,median,q1,mean,cum_cpu_s,cum_agent_frames,species,champ_hidden,champ_conns");
            robW.println("trial_id,algo,mode,regime,which,test_angle,frames_survived,fitness");
        }

        // Build and shuffle trial list so long jobs spread across threads
        List<Object[]> jobs = new ArrayList<>();
        for (String algo : algos) {
            for (int m : modes) {
                for (String r : regimes) {
                    int n = getTrialsForAlgoMode(algo, m);
                    for (int rep = 1; rep <= n; rep++) {
                        String id = algo + "-m" + m + "-" + r + "-" + rep;
                        if (!completed.contains(id)) {
                            jobs.add(new Object[]{algo, m, r, rep});
                        }
                    }
                }
            }
        }
        Collections.shuffle(jobs, new Random(12345));

        System.out.printf("Total jobs remaining to run: %d (already completed: %d)%n", jobs.size(), completed.size());

        ExecutorService ex = Executors.newFixedThreadPool(threads);
        final int total = jobs.size();
        final int[] done = {0};
        long t0 = System.currentTimeMillis();
        List<Future<?>> fs = new ArrayList<>();
        for (Object[] j : jobs) {
            fs.add(ex.submit(() -> {
                try {
                    runTrial((String) j[0], (Integer) j[1], (String) j[2], (Integer) j[3]);
                } catch (Throwable t) { t.printStackTrace(); }
                synchronized (done) {
                    done[0]++;
                    System.out.printf("[%d/%d] %s mode=%d %s rep=%d  (%.0fs elapsed)%n", done[0], total,
                            j[0], j[1], j[2], j[3], (System.currentTimeMillis() - t0) / 1000.0);
                }
            }));
        }
        for (Future<?> f : fs) f.get();
        ex.shutdown();
        trialsW.close(); gensW.close(); robW.close();
        Summarize.run(outDir);
        System.out.printf("DONE in %.1f s%n", (System.currentTimeMillis() - t0) / 1000.0);
    }

    // ── One full training run ────────────────────────────────────────────────
    static void runTrial(String algo, int mode, String regime, int rep) {
        final String id = algo + "-m" + mode + "-" + regime + "-" + rep;
        final boolean neat = algo.equals("NEAT");
        final double varDeg = regime.equals("scaled") ? getRangeMax(mode) : (regime.equals("random15") ? 15.0 : 0.0);
        Random rand = new Random();
        Ctx ctx = new Ctx(mode);
        ctx.angle = (rand.nextDouble() * 2 - 1) * varDeg;

        int inputCount = 4 + (mode - 1) * 2;
        List<CartAgent> pop = new ArrayList<>();
        NEATInnovationTracker tracker = null;
        List<NEATSpecies> species = new ArrayList<>();
        if (neat) {
            tracker = new NEATInnovationTracker(inputCount + 2);
            for (int i = 0; i < popSize; i++) {
                NEATGenome g = new NEATGenome(inputCount, 1, tracker);
                g.mutateWeights(1.0, 1.0);
                pop.add(new CartAgent(g, ctx));
            }
        } else {
            int[] top = getTopologyForAlgo(algo, inputCount);
            for (int i = 0; i < popSize; i++) pop.add(new CartAgent(new NeuralNetwork(top), ctx));
        }

        long cpu0 = MX.getCurrentThreadCpuTime();
        long wall0 = System.nanoTime();
        long cumFrames = 0;
        double bestEver = -Double.MAX_VALUE; int bestEverGen = 0; Brain bestEverBrain = null;
        List<Double> bests = new ArrayList<>();
        StringBuilder genBuf = new StringBuilder();
        int gen1of50 = -1, gen1of90 = -1; long frames90 = -1;
        int plateauCount = 0; String stop = "maxGens";
        int gen; Brain finalBrain = null;
        double lastBest = 0, lastMean = 0; int champHidden = 0, champConns = 0;

        for (gen = 1; gen <= maxGens; gen++) {
            // Evaluate every agent for up to maxFrames (agents are independent)
            long genFrames = 0;
            for (CartAgent a : pop) {
                int f = 0;
                while (a.alive && f < maxFrames) { a.update(); f++; }
                genFrames += f;
            }
            cumFrames += genFrames;

            double[] fit = new double[pop.size()];
            double sum = 0; CartAgent champ = pop.get(0);
            for (int i = 0; i < fit.length; i++) {
                fit[i] = pop.get(i).fitness; sum += fit[i];
                if (fit[i] > champ.fitness) champ = pop.get(i);
            }
            double[] sorted = fit.clone(); Arrays.sort(sorted);
            double best = sorted[sorted.length - 1], mean = sum / fit.length, median = sorted[sorted.length / 2];
            double q1 = sorted[sorted.length / 4], q3 = sorted[sorted.length * 3 / 4];
            bests.add(best);
            lastBest = best; lastMean = mean; finalBrain = champ.brain;
            if (best > bestEver) { bestEver = best; bestEverGen = gen; bestEverBrain = champ.brain.copy(); }
            if (gen1of50 < 0 && best >= 0.5 * maxFrames) gen1of50 = gen;
            if (gen1of90 < 0 && best >= 0.9 * maxFrames) { gen1of90 = gen; frames90 = cumFrames; }

            if (champ.brain instanceof NEATGenome) {
                NEATGenome ng = (NEATGenome) champ.brain;
                champHidden = 0; champConns = 0;
                for (NEATGenome.NodeGene n : ng.nodes) if (n.type == NEATGenome.NodeType.HIDDEN) champHidden++;
                for (NEATGenome.ConnectionGene c : ng.connections) if (c.enabled) champConns++;
            } else {
                int[] top = getTopologyForAlgo(algo, inputCount);
                champHidden = 0;
                for (int i = 1; i < top.length - 1; i++) champHidden += top[i];
                champConns = paramCount(top);
            }

            double cpuS = (MX.getCurrentThreadCpuTime() - cpu0) / 1e9;
            genBuf.append(String.format(Locale.ROOT, "%s,%s,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d%n",
                    id, algo, mode, regime, gen, ctx.angle, best, q3, median, q1, mean, cpuS, cumFrames,
                    neat ? species.size() : 0, champHidden, champConns));

            // Plateau check: trailing window-mean of per-generation best vs one window earlier
            int n = bests.size();
            if (n >= 2 * window) {
                double now = avg(bests, n - window, n), prev = avg(bests, n - 2 * window, n - window);
                double rel = (now - prev) / Math.max(1.0, Math.abs(prev));
                plateauCount = (rel < eps) ? plateauCount + 1 : 0;
                if (plateauCount >= patience && gen >= minGens) { stop = "plateau"; break; }
            }
            if (gen == maxGens) break;

            // Evolve (new angle chosen BEFORE offspring are constructed, as in the GUI)
            ctx.angle = (rand.nextDouble() * 2 - 1) * varDeg;
            pop = neat ? evolveNEAT(pop, species, tracker, ctx, rand) : evolveGA(pop, ctx, rand);
        }
        if (gen > maxGens) gen = maxGens;
        double cpuTotal = (MX.getCurrentThreadCpuTime() - cpu0) / 1e9;
        double wallTotal = (System.nanoTime() - wall0) / 1e9;
        int n = bests.size();
        double last10 = avg(bests, Math.max(0, n - 10), n);

        // Robustness of final champion and best-ever champion on unseen fixed angles
        StringBuilder robBuf = new StringBuilder();
        double[] rf = robustness(id, algo, mode, regime, "final", finalBrain, robBuf);
        double[] rb = robustness(id, algo, mode, regime, "bestever", bestEverBrain, robBuf);

        String trialRow = String.format(Locale.ROOT,
                "%s,%s,%d,%s,%d,%d,%s,%.3f,%.3f,%d,%.3f,%d,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%.3f,%.4f,%.3f,%.4f%n",
                id, algo, mode, regime, rep, bests.size(), stop, wallTotal, cpuTotal, cumFrames,
                bestEver, bestEverGen, lastBest, lastMean, last10, gen1of50, gen1of90, frames90,
                champHidden, champConns, rf[0], rf[1], rb[0], rb[1]);
        synchronized (CartpoleBenchmark.class) {
            gensW.print(genBuf); gensW.flush();
            robW.print(robBuf); robW.flush();
            trialsW.print(trialRow); trialsW.flush();
        }
    }

    // ── GA evolution (mirrors Cartpole.evolveFixedTopology) ──────────────────
    static List<CartAgent> evolveGA(List<CartAgent> pop, Ctx ctx, Random rand) {
        pop.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        List<CartAgent> np = new ArrayList<>();
        int elites = (int) (popSize * (elitismPct / 100.0));
        for (int i = 0; i < elites && i < pop.size(); i++) {
            CartAgent e = new CartAgent(pop.get(i).brain.copy(), ctx);
            e.isElite = true;
            np.add(e);
        }
        while (np.size() < popSize) {
            CartAgent pa = tournament(pop, rand), pb = tournament(pop, rand);
            NeuralNetwork child = ((NeuralNetwork) pa.brain).breed((NeuralNetwork) pb.brain, mutationPct, mutMag);
            np.add(new CartAgent(child, ctx));
        }
        return np;
    }

    static CartAgent tournament(List<CartAgent> pop, Random rand) {
        CartAgent best = pop.get(rand.nextInt(popSize));
        for (int i = 0; i < tourney - 1; i++) {
            CartAgent c = pop.get(rand.nextInt(popSize));
            if (c.fitness > best.fitness) best = c;
        }
        return best;
    }

    // ── NEAT evolution (mirrors Cartpole.evolveNEAT) ─────────────────────────
    static List<CartAgent> evolveNEAT(List<CartAgent> pop, List<NEATSpecies> species,
                                      NEATInnovationTracker tracker, Ctx ctx, Random rand) {
        tracker.resetGenerationCache();
        for (NEATSpecies sp : species) sp.clearMembers();
        for (CartAgent agent : pop) {
            NEATGenome g = (NEATGenome) agent.brain;
            boolean placed = false;
            for (NEATSpecies sp : species) {
                if (g.compatibilityDistance(sp.representative, 1.0, 1.0, 0.4) < compat) {
                    sp.add(g); sp.fitnesses.add(agent.fitness); placed = true; break;
                }
            }
            if (!placed) {
                NEATSpecies ns = new NEATSpecies(g);
                ns.add(g); ns.fitnesses.add(agent.fitness);
                species.add(ns);
            }
        }
        species.removeIf(sp -> sp.members.isEmpty());
        species.removeIf(sp -> sp.updateStagnation(stagnation) && species.size() > 1);

        double totalAdj = 0;
        for (NEATSpecies sp : species) totalAdj += sp.totalAdjustedFitness();
        for (NEATSpecies sp : species) sp.cull(survive);

        List<CartAgent> np = new ArrayList<>();
        for (NEATSpecies sp : species) {
            int offspring = (int) Math.round((sp.totalAdjustedFitness() / Math.max(1e-9, totalAdj)) * popSize);
            offspring = Math.max(1, offspring);
            if (sp.members.size() >= 5) {
                NEATGenome champ = sp.selectParent(rand);
                np.add(new CartAgent(champ.copy(), ctx));
                offspring--;
            }
            for (int i = 0; i < offspring && np.size() < popSize; i++) {
                NEATGenome child;
                if (sp.members.size() > 1 && rand.nextDouble() < xover) {
                    child = sp.selectParent(rand).crossover(sp.selectParent(rand));
                } else {
                    child = (NEATGenome) sp.selectParent(rand).copy();
                }
                child.mutateWeights(wMut, mutMag);
                if (rand.nextDouble() < addNode) child.mutateAddNode(tracker);
                if (rand.nextDouble() < addConn) child.mutateAddConnection(tracker);
                np.add(new CartAgent(child, ctx));
            }
            sp.updateRepresentative(rand);
        }
        while (np.size() < popSize) {
            NEATSpecies sp = species.get(rand.nextInt(species.size()));
            if (sp.members.isEmpty()) continue;
            NEATGenome child = (NEATGenome) sp.selectParent(rand).copy();
            child.mutateWeights(wMut, mutMag);
            if (rand.nextDouble() < addConn) child.mutateAddConnection(tracker);
            np.add(new CartAgent(child, ctx));
        }
        return np;
    }

    // ── Robustness: run a champion at fixed unseen angles ────────────────────
    static double[] robustness(String id, String algo, int mode, String regime, String which,
                               Brain brain, StringBuilder buf) {
        double sumFit = 0; int full = 0;
        double[] testAngles = getTestAngles(mode);
        for (double ang : testAngles) {
            Ctx c = new Ctx(mode); c.angle = ang;
            CartAgent a = new CartAgent(brain.copy(), c);
            int f = 0;
            while (a.alive && f < maxFrames) { a.update(); f++; }
            sumFit += a.fitness;
            if (f >= maxFrames) full++;
            buf.append(String.format(Locale.ROOT, "%s,%s,%d,%s,%s,%.3f,%d,%.3f%n", id, algo, mode, regime, which, ang, f, a.fitness));
        }
        return new double[]{sumFit / testAngles.length, (double) full / testAngles.length};
    }



    static double avg(List<Double> l, int from, int to) {
        double s = 0; for (int i = from; i < to; i++) s += l.get(i);
        return s / Math.max(1, to - from);
    }

    // ── Aggregation into summary.csv (mean ± sd per cell) ────────────────────
    static class Summarize {
        static void run(String dir) throws IOException {
            List<String[]> rows = new ArrayList<>();
            String header;
            try (BufferedReader br = new BufferedReader(new FileReader(new File(dir, "trials.csv")))) {
                header = br.readLine(); String line;
                while ((line = br.readLine()) != null) rows.add(line.split(","));
            }
            String[] cols = header.split(",");
            Map<String, Integer> idx = new HashMap<>();
            for (int i = 0; i < cols.length; i++) idx.put(cols[i], i);
            String[] metrics = {"gens_run", "wall_s", "cpu_s", "agent_frames", "best_ever", "final_gen_best", "last10_best_mean",
                    "gen_to_50pct", "gen_to_90pct", "frames_to_90pct", "champ_hidden_nodes", "champ_enabled_conns",
                    "robust_final_mean_fitness", "robust_final_survive_frac", "robust_bestever_mean_fitness", "robust_bestever_survive_frac"};
            Map<String, List<String[]>> cells = new TreeMap<>();
            for (String[] r : rows)
                cells.computeIfAbsent(r[idx.get("mode")] + "|" + r[idx.get("regime")] + "|" + r[idx.get("algo")], k -> new ArrayList<>()).add(r);
            try (PrintWriter pw = new PrintWriter(new FileWriter(new File(dir, "summary.csv")))) {
                StringBuilder h = new StringBuilder("mode,regime,algo,n_trials,n_plateau_stop,n_reached_90pct");
                for (String m : metrics) h.append(',').append(m).append("_mean,").append(m).append("_sd");
                pw.println(h);
                for (Map.Entry<String, List<String[]>> e : cells.entrySet()) {
                    String[] k = e.getKey().split("\\|");
                    List<String[]> rs = e.getValue();
                    int plateau = 0, reached = 0;
                    for (String[] r : rs) {
                        if (r[idx.get("stop_reason")].equals("plateau")) plateau++;
                        if (Integer.parseInt(r[idx.get("gen_to_90pct")]) > 0) reached++;
                    }
                    StringBuilder sb = new StringBuilder(k[0] + "," + k[1] + "," + k[2] + "," + rs.size() + "," + plateau + "," + reached);
                    for (String m : metrics) {
                        List<Double> v = new ArrayList<>();
                        for (String[] r : rs) {
                            double x = Double.parseDouble(r[idx.get(m)]);
                            if (m.equals("gen_to_50pct") || m.equals("gen_to_90pct") || m.equals("frames_to_90pct")) { if (x < 0) continue; }
                            v.add(x);
                        }
                        if (v.isEmpty()) { sb.append(",NA,NA"); continue; }
                        double mean = 0; for (double x : v) mean += x; mean /= v.size();
                        double var = 0; for (double x : v) var += (x - mean) * (x - mean);
                        double sd = v.size() > 1 ? Math.sqrt(var / (v.size() - 1)) : 0;
                        sb.append(String.format(Locale.ROOT, ",%.3f,%.3f", mean, sd));
                    }
                    pw.println(sb);
                }
            }
        }
    }
}
