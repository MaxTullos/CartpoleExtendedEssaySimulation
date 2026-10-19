import java.io.Serializable;

/**
 * Common interface for all brain types used by CartAgent.
 * Both the fixed-topology NeuralNetwork and NEATGenome implement this.
 */
public interface Brain extends Serializable {
    /**
     * Run a forward pass and return outputs.
     * @param inputs Normalized sensor values
     * @return Output activations (typically one value: force)
     */
    double[] feedForward(double[] inputs);

    /**
     * Deep-copy this brain.
     */
    Brain copy();
}
