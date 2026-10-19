import java.util.HashMap;
import java.util.Map;

/**
 * Tracks global innovation numbers for NEAT connection genes.
 *
 * Within a single generation, if the same structural mutation (same from-node → to-node)
 * occurs in multiple genomes, they receive the SAME innovation number. This is what
 * allows NEAT crossover to align genes correctly.
 *
 * Reset between generations via {@link #resetGenerationCache()}.
 */
public class NEATInnovationTracker {

    /** Maps (fromNodeId, toNodeId) → innovation number */
    private final Map<Long, Integer> connectionInnovations = new HashMap<>();

    /** Maps (fromNodeId, toNodeId) of the split connection → new hidden node id */
    private final Map<Long, Integer> splitNodeIds = new HashMap<>();

    private int nextInnovation = 0;
    private int nextNodeId;

    /**
     * @param startingNodeId The first ID available for new hidden nodes
     *                       (should be set to inputCount + outputCount after genome init)
     */
    public NEATInnovationTracker(int startingNodeId) {
        this.nextNodeId = startingNodeId;
    }

    /** Combined key from two node IDs (fits in a long). */
    private long key(int from, int to) {
        return ((long) from << 32) | (to & 0xFFFFFFFFL);
    }

    /**
     * Returns the innovation number for a connection from→to,
     * creating a new one if it has never been seen this generation.
     */
    public int getInnovation(int from, int to) {
        long k = key(from, to);
        return connectionInnovations.computeIfAbsent(k, __ -> nextInnovation++);
    }

    /**
     * Returns the new node ID that was assigned when a connection from→to was split.
     * If this split has not happened yet this generation, allocates a fresh node ID.
     */
    public int getOrCreateSplitNode(int from, int to) {
        long k = key(from, to);
        return splitNodeIds.computeIfAbsent(k, __ -> nextNodeId++);
    }

    /** Peek at what the next innovation number would be (for diagnostics). */
    public int getNextInnovation() { return nextInnovation; }

    /** Peek at what the next node ID would be (for diagnostics). */
    public int getNextNodeId() { return nextNodeId; }

    /**
     * Call at the start of each generation so that identical mutations within
     * the same generation share innovation numbers, but new mutations in the
     * next generation get fresh numbers.
     */
    public void resetGenerationCache() {
        connectionInnovations.clear();
        splitNodeIds.clear();
    }
}
