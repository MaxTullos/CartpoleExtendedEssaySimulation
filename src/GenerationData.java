import java.io.Serializable;

/**
 * Stores performance metrics for a single generation.
 */
public class GenerationData implements Serializable {
    public final int generation;
    public final double bestFitness;
    public final double medianFitness;
    public final double meanFitness;
    public final double q1Fitness; // 25th percentile
    public final double q3Fitness; // 75th percentile

    public GenerationData(int generation, double bestFitness, double medianFitness, double meanFitness, double q1Fitness, double q3Fitness) {
        this.generation = generation;
        this.bestFitness = bestFitness;
        this.medianFitness = medianFitness;
        this.meanFitness = meanFitness;
        this.q1Fitness = q1Fitness;
        this.q3Fitness = q3Fitness;
    }
}
