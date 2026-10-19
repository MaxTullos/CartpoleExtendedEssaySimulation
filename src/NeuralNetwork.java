import java.io.Serializable;
import java.util.Random;

/**
 * A simple Feedforward Neural Network
 */
public class NeuralNetwork implements Brain {
    private static final long serialVersionUID = 1L;

    public int[] topology;
    public double[][][] weights; // [layer][neuron][incomingWeight]
    public double[][] biases;    // [layer][neuron]
    public double[][] lastActivations; // [layer][neuron] to store node values for visualization

    public NeuralNetwork(int[] topology) {
        this.topology = topology;
        weights = new double[topology.length - 1][][];
        biases = new double[topology.length - 1][];
        lastActivations = new double[topology.length][];

        Random rand = new Random();

        for (int i = 0; i < topology.length; i++) {
            lastActivations[i] = new double[topology[i]];
        }

        for (int i = 0; i < topology.length - 1; i++) {
            int neuronsCurrentLayer = topology[i + 1];
            int neuronsPreviousLayer = topology[i];
            
            weights[i] = new double[neuronsCurrentLayer][neuronsPreviousLayer];
            biases[i] = new double[neuronsCurrentLayer];

            for (int j = 0; j < neuronsCurrentLayer; j++) {
                biases[i][j] = rand.nextGaussian(); // Initialize bias
                for (int k = 0; k < neuronsPreviousLayer; k++) {
                    weights[i][j][k] = rand.nextGaussian(); // Initialize weight
                }
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, lastActivations[0], 0, inputs.length);
        double[] currentActivations = inputs;

        for (int i = 0; i < topology.length - 1; i++) {
            double[] nextActivations = new double[topology[i + 1]];
            for (int j = 0; j < nextActivations.length; j++) {
                double sum = biases[i][j];
                for (int k = 0; k < currentActivations.length; k++) {
                    sum += currentActivations[k] * weights[i][j][k];
                }
                // Tanh Activation function limits outputs between -1 and 1
                nextActivations[j] = Math.tanh(sum); 
            }
            System.arraycopy(nextActivations, 0, lastActivations[i + 1], 0, nextActivations.length);
            currentActivations = nextActivations;
        }
        return currentActivations;
    }

    @Override
    public Brain copy() {
        NeuralNetwork copy = new NeuralNetwork(topology);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                copy.biases[i][j] = this.biases[i][j];
                System.arraycopy(this.weights[i][j], 0, copy.weights[i][j], 0, weights[i][j].length);
            }
        }
        return copy;
    }

    public NeuralNetwork breed(NeuralNetwork partner, double mutationRate, double mutationMagnitude) {
        NeuralNetwork child = new NeuralNetwork(topology);
        Random rand = new Random();

        // Crossover and Mutation
        double actualMutationRate = mutationRate / 100.0; // User input %

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                
                // Breed Biases
                if (rand.nextBoolean()) child.biases[i][j] = this.biases[i][j];
                else child.biases[i][j] = partner.biases[i][j];
                
                if (rand.nextDouble() < actualMutationRate) {
                    child.biases[i][j] += rand.nextGaussian() * mutationMagnitude;
                }

                // Breed Weights
                for (int k = 0; k < weights[i][j].length; k++) {
                    if (rand.nextBoolean()) child.weights[i][j][k] = this.weights[i][j][k];
                    else child.weights[i][j][k] = partner.weights[i][j][k];
                    
                    if (rand.nextDouble() < actualMutationRate) {
                        child.weights[i][j][k] += rand.nextGaussian() * mutationMagnitude;
                    }
                }
            }
        }
        return child;
    }
}
