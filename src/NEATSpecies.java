import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A NEAT species: a group of genomes that are structurally similar.
 * Protects innovation by giving new topological mutations time to optimize
 * before being out-competed by established networks.
 */
public class NEATSpecies {

    /** The genome used to test compatibility for membership. Updated each generation. */
    public NEATGenome representative;

    /** All genomes currently in this species. */
    public List<NEATGenome> members = new ArrayList<>();

    /** How many generations this species has not improved its best fitness. */
    public int stagnationCounter = 0;

    /** Best fitness ever seen in this species. */
    public double bestFitness = 0;

    /** Fitness scores parallel to {@code members}. Set externally after evaluation. */
    public List<Double> fitnesses = new ArrayList<>();

    public NEATSpecies(NEATGenome representative) {
        this.representative = representative;
    }

    /** Add a genome to this species. */
    public void add(NEATGenome genome) {
        members.add(genome);
    }

    /** Total adjusted fitness (raw fitness / species size) — used for offspring allocation. */
    public double totalAdjustedFitness() {
        double total = 0;
        int size = members.size();
        for (double f : fitnesses) {
            total += f / Math.max(1, size);
        }
        return total;
    }

    /**
     * Remove the bottom {@code fraction} of members by fitness.
     * Always keeps at least one survivor.
     */
    public void cull(double survivalFraction) {
        if (members.isEmpty()) return;

        // Sort members by fitness descending (parallel sort with fitnesses list)
        // Use a combined index sort
        Integer[] indices = new Integer[members.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        java.util.Arrays.sort(indices, (a, b) -> {
            double fa = fitnesses.size() > a ? fitnesses.get(a) : 0;
            double fb = fitnesses.size() > b ? fitnesses.get(b) : 0;
            return Double.compare(fb, fa); // descending
        });

        int keepCount = Math.max(1, (int) Math.ceil(members.size() * survivalFraction));

        List<NEATGenome> kept      = new ArrayList<>();
        List<Double>     keptFits  = new ArrayList<>();
        for (int i = 0; i < keepCount; i++) {
            int idx = indices[i];
            kept.add(members.get(idx));
            keptFits.add(fitnesses.size() > idx ? fitnesses.get(idx) : 0.0);
        }

        members  = kept;
        fitnesses = keptFits;
    }

    /**
     * Returns the best fitness seen by any current member.
     * Call after evaluation, before culling.
     */
    public double bestCurrentFitness() {
        double best = 0;
        for (double f : fitnesses) {
            if (f > best) best = f;
        }
        return best;
    }

    /**
     * Update stagnation counter and best fitness.
     * @return true if this species has stagnated past {@code maxStagnation} generations.
     */
    public boolean updateStagnation(int maxStagnation) {
        double current = bestCurrentFitness();
        if (current > bestFitness) {
            bestFitness = current;
            stagnationCounter = 0;
        } else {
            stagnationCounter++;
        }
        return stagnationCounter >= maxStagnation;
    }

    /**
     * Select a random genome weighted by fitness (tournament selection within species).
     */
    public NEATGenome selectParent(Random rand) {
        if (members.size() == 1) return members.get(0);

        // Tournament of size 3 within the species
        int size = members.size();
        int best = rand.nextInt(size);
        for (int i = 0; i < 2; i++) {
            int challenger = rand.nextInt(size);
            double bestFit = fitnesses.size() > best ? fitnesses.get(best) : 0;
            double chalFit = fitnesses.size() > challenger ? fitnesses.get(challenger) : 0;
            if (chalFit > bestFit) best = challenger;
        }
        return members.get(best);
    }

    /** Update the representative to a random surviving member. */
    public void updateRepresentative(Random rand) {
        if (!members.isEmpty()) {
            representative = members.get(rand.nextInt(members.size()));
        }
    }

    /** Clear members and fitnesses for the next generation. */
    public void clearMembers() {
        members.clear();
        fitnesses.clear();
    }
}
