import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class ExtractTopNEAT {


    public static void main(String[] args) throws Exception {
        int threads = 7;
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        List<ChampionRecord> champions = Collections.synchronizedList(new ArrayList<>());
        int targetChamps = 3;
        System.out.println("Searching for top 3 NEAT Triple Pendulum champions...");

        int totalAttempts = 20;
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < totalAttempts; i++) {
            final int seed = 1000 + i * 37;
            futures.add(ex.submit(() -> {
                if (champions.size() >= targetChamps) return;
                try {
                    ChampionRecord rec = trainNEAT(seed);
                    if (rec != null && rec.fitness >= 1790.0 && rec.robustSurviveFrac >= 0.70) {
                        synchronized (champions) {
                            if (champions.size() < targetChamps) {
                                champions.add(rec);
                                System.out.printf("Found Champion %d: Fitness=%.1f, Robust=%.1f%%, Gens=%d, Hidden=%d, Conns=%d%n",
                                        champions.size(), rec.fitness, rec.robustSurviveFrac * 100, rec.gensRun,
                                        rec.hiddenCount, rec.connCount);
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }));
        }

        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception ignored) {}
        }
        ex.shutdown();

        if (champions.size() < 3) {
            System.out.println("Could not find 3 high-robustness champions in initial batch, searching more...");
            ExecutorService ex2 = Executors.newFixedThreadPool(threads);
            List<Future<?>> futures2 = new ArrayList<>();
            for (int i = totalAttempts; i < totalAttempts + 30; i++) {
                final int seed = 2000 + i * 41;
                futures2.add(ex2.submit(() -> {
                    if (champions.size() >= targetChamps) return;
                    try {
                        ChampionRecord rec = trainNEAT(seed);
                        if (rec != null && rec.fitness >= 1700.0) {
                            synchronized (champions) {
                                if (champions.size() < targetChamps) {
                                    champions.add(rec);
                                    System.out.printf("Found Champion %d: Fitness=%.1f, Robust=%.1f%%, Gens=%d, Hidden=%d, Conns=%d%n",
                                            champions.size(), rec.fitness, rec.robustSurviveFrac * 100, rec.gensRun,
                                            rec.hiddenCount, rec.connCount);
                                }
                            }
                        }
                    } catch (Exception ignored) {}
                }));
            }
            for (Future<?> f : futures2) {
                try { f.get(); } catch (Exception ignored) {}
            }
            ex2.shutdown();
        }

        // Sort champions by robustSurviveFrac desc, fitness desc
        champions.sort((a, b) -> {
            int c = Double.compare(b.robustSurviveFrac, a.robustSurviveFrac);
            if (c != 0) return c;
            return Double.compare(b.fitness, a.fitness);
        });

        writeJson(champions, "benchmark_results_topologies/top3_neat_triple.json");
        System.out.println("Successfully saved top 3 NEAT Triple Pendulum champions to benchmark_results_topologies/top3_neat_triple.json");
    }

    static class ChampionRecord {
        int id;
        double fitness;
        double robustSurviveFrac;
        int gensRun;
        int hiddenCount;
        int connCount;
        NEATGenome genome;
    }

    static ChampionRecord trainNEAT(int seed) {
        Random rand = new Random(seed);
        CartpoleBenchmark.Ctx ctx = new CartpoleBenchmark.Ctx(3);
        ctx.angle = (rand.nextDouble() * 2 - 1) * 2.5;

        int inputCount = 8;
        NEATInnovationTracker tracker = new NEATInnovationTracker(inputCount + 2);
        List<CartAgent> pop = new ArrayList<>();
        for (int i = 0; i < 150; i++) {
            NEATGenome g = new NEATGenome(inputCount, 1, tracker);
            g.mutateWeights(1.0, 1.0);
            pop.add(new CartAgent(g, ctx));
        }

        List<NEATSpecies> species = new ArrayList<>();
        double bestEver = 0;
        NEATGenome bestGenome = null;
        List<Double> bests = new ArrayList<>();
        int plateauCount = 0;

        for (int gen = 1; gen <= 350; gen++) {
            double genBest = 0;
            NEATGenome genBestG = null;
            for (CartAgent a : pop) {
                int frames = 0;
                while (a.alive && frames < 1800) { a.update(); frames++; }
                if (a.fitness > genBest) {
                    genBest = a.fitness;
                    genBestG = (NEATGenome) a.brain;
                }
            }
            bests.add(genBest);
            if (genBest > bestEver) {
                bestEver = genBest;
                bestGenome = (NEATGenome) genBestG.copy();
            }

            int n = bests.size();
            if (n >= 40) {
                double now = 0; for (int k = n - 20; k < n; k++) now += bests.get(k); now /= 20.0;
                double prev = 0; for (int k = n - 40; k < n - 20; k++) prev += bests.get(k); prev /= 20.0;
                double rel = (now - prev) / Math.max(1.0, Math.abs(prev));
                plateauCount = (rel < 0.01) ? plateauCount + 1 : 0;
                if (plateauCount >= 10 && gen >= 100) break;
            }

            if (bestEver >= 1799.0 && gen >= 120 && plateauCount >= 5) break;

            ctx.angle = (rand.nextDouble() * 2 - 1) * 2.5;
            pop = CartpoleBenchmark.evolveNEAT(pop, species, tracker, ctx, rand);
        }

        if (bestGenome == null) return null;

        // Evaluate robustness
        double[] testAngles = {-3.75, -3.0, -2.5, -2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0, 2.5, 3.0, 3.75};
        int fullSurv = 0;
        for (double ang : testAngles) {
            CartpoleBenchmark.Ctx c = new CartpoleBenchmark.Ctx(3); c.angle = ang;
            CartAgent a = new CartAgent(bestGenome.copy(), c);
            int f = 0;
            while (a.alive && f < 1800) { a.update(); f++; }
            if (f >= 1800) fullSurv++;
        }
        double robFrac = (double) fullSurv / testAngles.length;

        int hidden = 0, conns = 0;
        for (NEATGenome.NodeGene ng : bestGenome.nodes) {
            if (ng.type == NEATGenome.NodeType.HIDDEN) hidden++;
        }
        for (NEATGenome.ConnectionGene cg : bestGenome.connections) {
            if (cg.enabled) conns++;
        }

        ChampionRecord r = new ChampionRecord();
        r.fitness = bestEver;
        r.robustSurviveFrac = robFrac;
        r.gensRun = bests.size();
        r.hiddenCount = hidden;
        r.connCount = conns;
        r.genome = bestGenome;
        return r;
    }

    static void writeJson(List<ChampionRecord> champs, String path) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("{");
            pw.println("  \"champions\": [");
            for (int i = 0; i < Math.min(3, champs.size()); i++) {
                ChampionRecord cr = champs.get(i);
                pw.println("    {");
                pw.printf(Locale.ROOT, "      \"rank\": %d,%n", i + 1);
                pw.printf(Locale.ROOT, "      \"fitness\": %.2f,%n", cr.fitness);
                pw.printf(Locale.ROOT, "      \"robustness_pct\": %.1f,%n", cr.robustSurviveFrac * 100.0);
                pw.printf(Locale.ROOT, "      \"generations_to_solve\": %d,%n", cr.gensRun);
                pw.printf(Locale.ROOT, "      \"hidden_nodes_count\": %d,%n", cr.hiddenCount);
                pw.printf(Locale.ROOT, "      \"enabled_connections_count\": %d,%n", cr.connCount);

                // Nodes
                pw.println("      \"nodes\": [");
                for (int j = 0; j < cr.genome.nodes.size(); j++) {
                    NEATGenome.NodeGene ng = cr.genome.nodes.get(j);
                    String label = getNodeLabel(ng.id, ng.type);
                    pw.printf(Locale.ROOT, "        {\"id\": %d, \"type\": \"%s\", \"label\": \"%s\"}%s%n",
                            ng.id, ng.type.name(), label, (j < cr.genome.nodes.size() - 1 ? "," : ""));
                }
                pw.println("      ],");

                // Connections
                pw.println("      \"connections\": [");
                List<NEATGenome.ConnectionGene> cl = new ArrayList<>();
                for (NEATGenome.ConnectionGene cg : cr.genome.connections) {
                    if (cg.enabled) cl.add(cg);
                }
                for (int j = 0; j < cl.size(); j++) {
                    NEATGenome.ConnectionGene cg = cl.get(j);
                    pw.printf(Locale.ROOT, "        {\"from\": %d, \"to\": %d, \"weight\": %.4f, \"innovation\": %d}%s%n",
                            cg.from, cg.to, cg.weight, cg.innovation, (j < cl.size() - 1 ? "," : ""));
                }
                pw.println("      ]");
                pw.printf("    }%s%n", (i < Math.min(3, champs.size()) - 1 ? "," : ""));
            }
            pw.println("  ]");
            pw.println("}");
        }
    }

    static String getNodeLabel(int id, NEATGenome.NodeType type) {
        switch (type) {
            case INPUT:
                String[] inLbls = {"Cart X", "Cart V", "Pole 1 Angle", "Pole 1 AngV", "Pole 2 Angle", "Pole 2 AngV", "Pole 3 Angle", "Pole 3 AngV"};
                if (id >= 0 && id < inLbls.length) return inLbls[id];
                return "In " + id;
            case BIAS:
                return "Bias (1.0)";
            case OUTPUT:
                return "Motor Force";
            case HIDDEN:
                return "Hidden " + id;
            default:
                return "Node " + id;
        }
    }
}
