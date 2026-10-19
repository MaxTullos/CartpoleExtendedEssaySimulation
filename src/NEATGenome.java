import java.io.Serializable;
import java.util.*;

/**
 * A NEAT genome: a list of NodeGenes and ConnectionGenes.
 *
 * The network is an arbitrary directed acyclic graph (DAG).
 * Evaluation is done via topological sort so cycles introduced
 * by add-connection mutations are safely handled.
 */
public class NEATGenome implements Brain {
    private static final long serialVersionUID = 1L;

    // ── Node types ───────────────────────────────────────────────────────────
    public enum NodeType { INPUT, HIDDEN, OUTPUT, BIAS }

    public static class NodeGene implements Serializable {
        private static final long serialVersionUID = 1L;
        public int id;
        public NodeType type;
        public double lastActivation; // for visualization

        public NodeGene(int id, NodeType type) {
            this.id = id;
            this.type = type;
        }

        public NodeGene copy() {
            NodeGene c = new NodeGene(id, type);
            c.lastActivation = lastActivation;
            return c;
        }
    }

    // ── Connection genes ──────────────────────────────────────────────────────
    public static class ConnectionGene implements Serializable {
        private static final long serialVersionUID = 1L;
        public int from, to;
        public double weight;
        public boolean enabled;
        public int innovation;

        public ConnectionGene(int from, int to, double weight, boolean enabled, int innovation) {
            this.from = from;
            this.to = to;
            this.weight = weight;
            this.enabled = enabled;
            this.innovation = innovation;
        }

        public ConnectionGene copy() {
            return new ConnectionGene(from, to, weight, enabled, innovation);
        }
    }

    // ── Genome fields ─────────────────────────────────────────────────────────
    public List<NodeGene>       nodes       = new ArrayList<>();
    public List<ConnectionGene> connections = new ArrayList<>();

    /** Number of input nodes (excludes bias). */
    public int inputCount;
    /** Number of output nodes. */
    public int outputCount;

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * Creates a minimal genome: inputCount inputs + 1 bias → outputCount outputs,
     * all directly connected (no hidden nodes).
     */
    public NEATGenome(int inputCount, int outputCount, NEATInnovationTracker tracker) {
        this.inputCount  = inputCount;
        this.outputCount = outputCount;

        // Input nodes  (ids 0 .. inputCount-1)
        for (int i = 0; i < inputCount; i++) {
            nodes.add(new NodeGene(i, NodeType.INPUT));
        }
        // Bias node (id inputCount)
        nodes.add(new NodeGene(inputCount, NodeType.BIAS));

        // Output nodes (ids inputCount+1 .. inputCount+outputCount)
        for (int i = 0; i < outputCount; i++) {
            nodes.add(new NodeGene(inputCount + 1 + i, NodeType.OUTPUT));
        }

        // Connect every input (and bias) to every output
        Random rand = new Random();
        int totalInputsWithBias = inputCount + 1;
        for (int in = 0; in < totalInputsWithBias; in++) {
            for (int out = 0; out < outputCount; out++) {
                int outId = inputCount + 1 + out;
                connections.add(new ConnectionGene(
                        in, outId,
                        rand.nextGaussian(),
                        true,
                        tracker.getInnovation(in, outId)
                ));
            }
        }
    }

    /** Private constructor for copy / crossover. */
    private NEATGenome(int inputCount, int outputCount) {
        this.inputCount  = inputCount;
        this.outputCount = outputCount;
    }

    // ── Brain interface ───────────────────────────────────────────────────────

    @Override
    public double[] feedForward(double[] inputs) {
        // Map node id → activation value
        Map<Integer, Double> activation = new HashMap<>();

        // Set input activations
        for (int i = 0; i < inputCount; i++) {
            activation.put(i, inputs[i]);
        }
        // Bias is always 1.0
        activation.put(inputCount, 1.0);

        // Topological sort (Kahn's algorithm on enabled connections)
        // Build in-degree map and adjacency list
        Map<Integer, List<int[]>> adj = new HashMap<>(); // from → list of [to, connIdx]
        Map<Integer, Integer> inDegree = new HashMap<>();

        for (NodeGene ng : nodes) {
            adj.put(ng.id, new ArrayList<>());
            inDegree.put(ng.id, 0);
        }
        for (int ci = 0; ci < connections.size(); ci++) {
            ConnectionGene cg = connections.get(ci);
            if (!cg.enabled) continue;
            adj.get(cg.from).add(new int[]{cg.to, ci});
            inDegree.put(cg.to, inDegree.getOrDefault(cg.to, 0) + 1);
        }

        // Seeds: nodes with in-degree 0 (inputs + bias) go first
        Queue<Integer> queue = new LinkedList<>();
        for (NodeGene ng : nodes) {
            if (inDegree.get(ng.id) == 0) {
                queue.add(ng.id);
            }
        }

        // Accumulate weighted sums for each node
        Map<Integer, Double> weightedSum = new HashMap<>();
        for (NodeGene ng : nodes) {
            weightedSum.put(ng.id, 0.0);
        }

        while (!queue.isEmpty()) {
            int cur = queue.poll();
            double act;
            if (cur < inputCount) {
                act = activation.getOrDefault(cur, 0.0);
            } else if (nodes.get(getNodeIndex(cur)).type == NodeType.BIAS) {
                act = 1.0;
            } else {
                act = Math.tanh(weightedSum.getOrDefault(cur, 0.0));
            }
            activation.put(cur, act);

            // Update lastActivation for visualization
            NodeGene ng = nodes.get(getNodeIndex(cur));
            ng.lastActivation = act;

            for (int[] edge : adj.get(cur)) {
                int to = edge[0];
                int ci = edge[1];
                weightedSum.put(to, weightedSum.getOrDefault(to, 0.0)
                        + act * connections.get(ci).weight);
                inDegree.put(to, inDegree.get(to) - 1);
                if (inDegree.get(to) == 0) {
                    queue.add(to);
                }
            }
        }

        // Collect output activations
        double[] outputs = new double[outputCount];
        for (int i = 0; i < outputCount; i++) {
            int outId = inputCount + 1 + i;
            double sum = weightedSum.getOrDefault(outId, 0.0);
            outputs[i] = Math.tanh(sum);
            NodeGene ng = nodes.get(getNodeIndex(outId));
            if (ng != null) {
                ng.lastActivation = outputs[i];
                activation.put(outId, outputs[i]);
            }
        }
        return outputs;
    }

    private int getNodeIndex(int id) {
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).id == id) return i;
        }
        return -1;
    }

    @Override
    public Brain copy() {
        NEATGenome g = new NEATGenome(inputCount, outputCount);
        for (NodeGene ng : nodes)       g.nodes.add(ng.copy());
        for (ConnectionGene cg : connections) g.connections.add(cg.copy());
        return g;
    }

    // ── Mutations ─────────────────────────────────────────────────────────────

    /** Perturb or replace connection weights. */
    public void mutateWeights(double rate, double magnitude) {
        Random rand = new Random();
        for (ConnectionGene cg : connections) {
            if (rand.nextDouble() < rate) {
                if (rand.nextDouble() < 0.1) {
                    cg.weight = rand.nextGaussian(); // full replacement
                } else {
                    cg.weight += rand.nextGaussian() * magnitude;
                }
            }
        }
    }

    /**
     * Split an enabled connection in two by inserting a new hidden node.
     * Old connection disabled; two new connections added.
     */
    public void mutateAddNode(NEATInnovationTracker tracker) {
        List<ConnectionGene> enabled = new ArrayList<>();
        for (ConnectionGene cg : connections) {
            if (cg.enabled) enabled.add(cg);
        }
        if (enabled.isEmpty()) return;

        Random rand = new Random();
        ConnectionGene split = enabled.get(rand.nextInt(enabled.size()));
        split.enabled = false;

        int newNodeId = tracker.getOrCreateSplitNode(split.from, split.to);

        // Avoid duplicates: if this node already exists (same gen, same split), don't re-add
        boolean nodeExists = false;
        for (NodeGene ng : nodes) {
            if (ng.id == newNodeId) { nodeExists = true; break; }
        }
        if (!nodeExists) {
            nodes.add(new NodeGene(newNodeId, NodeType.HIDDEN));
        }

        // in-connection: weight 1.0 (preserves old output roughly)
        int innA = tracker.getInnovation(split.from, newNodeId);
        connections.add(new ConnectionGene(split.from, newNodeId, 1.0, true, innA));

        // out-connection: old weight (preserves mapping)
        int innB = tracker.getInnovation(newNodeId, split.to);
        connections.add(new ConnectionGene(newNodeId, split.to, split.weight, true, innB));
    }

    /**
     * Add a new connection between two currently unconnected nodes.
     * Guards against creating input→input, output→input, or self-loops.
     */
    public void mutateAddConnection(NEATInnovationTracker tracker) {
        Random rand = new Random();

        // Collect candidate from/to node ids
        List<Integer> fromCandidates = new ArrayList<>();
        List<Integer> toCandidates   = new ArrayList<>();

        for (NodeGene ng : nodes) {
            if (ng.type != NodeType.OUTPUT) fromCandidates.add(ng.id);
            if (ng.type != NodeType.INPUT && ng.type != NodeType.BIAS) toCandidates.add(ng.id);
        }

        // Build existing connection set for fast lookup
        Set<Long> existing = new HashSet<>();
        for (ConnectionGene cg : connections) {
            existing.add(((long) cg.from << 32) | (cg.to & 0xFFFFFFFFL));
        }

        // Try a handful of random pairs
        for (int attempt = 0; attempt < 20; attempt++) {
            int from = fromCandidates.get(rand.nextInt(fromCandidates.size()));
            int to   = toCandidates  .get(rand.nextInt(toCandidates.size()));
            if (from == to) continue;

            long k = ((long) from << 32) | (to & 0xFFFFFFFFL);
            if (existing.contains(k)) continue;

            int inn = tracker.getInnovation(from, to);
            connections.add(new ConnectionGene(from, to, rand.nextGaussian(), true, inn));
            return;
        }
    }

    // ── Crossover ─────────────────────────────────────────────────────────────

    /**
     * Produce a child genome from this (assumed fitter parent) and {@code other}.
     * Matching genes: random choice of parent weight.
     * Disjoint/excess genes: inherited from the fitter parent (this).
     */
    public NEATGenome crossover(NEATGenome other) {
        NEATGenome child = new NEATGenome(inputCount, outputCount);

        // Build innovation→gene map for the other parent
        Map<Integer, ConnectionGene> otherMap = new HashMap<>();
        for (ConnectionGene cg : other.connections) {
            otherMap.put(cg.innovation, cg);
        }

        Random rand = new Random();
        Set<Integer> nodeIds = new HashSet<>();

        for (ConnectionGene cg : this.connections) {
            ConnectionGene childCg;
            if (otherMap.containsKey(cg.innovation)) {
                // Matching gene: random from either parent
                ConnectionGene otherCg = otherMap.get(cg.innovation);
                childCg = rand.nextBoolean() ? cg.copy() : otherCg.copy();
                // Re-disable if either parent has it disabled (75% chance)
                if (!cg.enabled || !otherCg.enabled) {
                    childCg.enabled = rand.nextDouble() > 0.75;
                }
            } else {
                // Disjoint/excess: inherit from fitter parent (this)
                childCg = cg.copy();
            }
            child.connections.add(childCg);
            nodeIds.add(childCg.from);
            nodeIds.add(childCg.to);
        }

        // Add all needed nodes from this parent
        for (NodeGene ng : this.nodes) {
            if (nodeIds.contains(ng.id) || ng.type == NodeType.INPUT
                    || ng.type == NodeType.BIAS || ng.type == NodeType.OUTPUT) {
                child.nodes.add(ng.copy());
            }
        }

        return child;
    }

    // ── Compatibility distance ────────────────────────────────────────────────

    /**
     * Computes the NEAT compatibility distance δ between this genome and another.
     *
     * δ = (c1*E + c2*D) / N  + c3 * W̄
     */
    public double compatibilityDistance(NEATGenome other,
                                        double c1, double c2, double c3) {
        // Map innovation → connection for each genome
        Map<Integer, ConnectionGene> mine  = new HashMap<>();
        Map<Integer, ConnectionGene> theirs = new HashMap<>();

        int maxInnovMine  = 0;
        int maxInnovTheir = 0;

        for (ConnectionGene cg : this.connections) {
            mine.put(cg.innovation, cg);
            if (cg.innovation > maxInnovMine) maxInnovMine = cg.innovation;
        }
        for (ConnectionGene cg : other.connections) {
            theirs.put(cg.innovation, cg);
            if (cg.innovation > maxInnovTheir) maxInnovTheir = cg.innovation;
        }

        int matchingCount  = 0;
        double weightDiffSum = 0;
        int disjoint = 0;
        int excess   = 0;
        int threshold = Math.min(maxInnovMine, maxInnovTheir);

        Set<Integer> allInn = new HashSet<>();
        allInn.addAll(mine.keySet());
        allInn.addAll(theirs.keySet());

        for (int inn : allInn) {
            boolean inMine   = mine.containsKey(inn);
            boolean inTheirs = theirs.containsKey(inn);

            if (inMine && inTheirs) {
                matchingCount++;
                weightDiffSum += Math.abs(mine.get(inn).weight - theirs.get(inn).weight);
            } else {
                if (inn <= threshold) {
                    disjoint++;
                } else {
                    excess++;
                }
            }
        }

        int N = Math.max(1, Math.max(this.connections.size(), other.connections.size()));
        double avgWeightDiff = matchingCount > 0 ? weightDiffSum / matchingCount : 0;

        return (c1 * excess + c2 * disjoint) / N + c3 * avgWeightDiff;
    }
}
