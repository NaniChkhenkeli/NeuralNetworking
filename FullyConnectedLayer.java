package Layer;

import java.util.List;
import java.util.Random;

public abstract class FullyConnectedLayer extends Layer {

    private long seed;
    private final double leak = 0.01;

    private double[][] weights;
    private int inputLength;
    private int outputLength;
    private double learningRate;

    private double[] lastZ;
    private double[] lastA;

    public FullyConnectedLayer(int inputLength, int outputLength, long seed, double learningRate) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.seed = seed;
        this.learningRate = learningRate;

        generateRandomWeights();
    }

    private void generateRandomWeights() {
        Random random = new Random(seed);
        weights = new double[inputLength][outputLength];

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] flattenedInput = flatten(input);
        lastZ = computeZ(flattenedInput);
        lastA = activate(lastZ);
        return lastA;
    }

    @Override
    public double[] getOutput(double[] input) {
        lastZ = computeZ(input);
        lastA = activate(lastZ);
        return lastA;
    }

    private double[] computeZ(double[] input) {
        double[] z = new double[outputLength];

        for (int j = 0; j < outputLength; j++) {
            double sum = 0.0;

            for (int i = 0; i < inputLength; i++) {
                sum += weights[i][j] * input[i];
            }

            z[j] = sum;
        }

        return z;
    }

    private double[] activate(double[] z) {
        double[] a = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            a[i] = Math.max(leak * z[i], z[i]);
        }

        return a;
    }

    @Override
    public void backPropagation(double[] dLdO) {
        double[] dLdZ = new double[dLdO.length];

        for (int i = 0; i < dLdO.length; i++) {
            dLdZ[i] = dLdO[i] * (lastZ[i] > 0 ? 1 : leak);
        }

        double[][] dLdW = new double[inputLength][outputLength];

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                dLdW[i][j] = dLdZ[j] * lastA[i];
            }
        }

        double[] dLdI = new double[inputLength];

        for (int i = 0; i < inputLength; i++) {
            double sum = 0.0;

            for (int j = 0; j < outputLength; j++) {
                sum += dLdZ[j] * weights[i][j];
            }

            dLdI[i] = sum;
        }

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                weights[i][j] -= learningRate * dLdW[i][j];
            }
        }

        if (getPreviousLayer() != null) {
            getPreviousLayer().backPropagation(dLdI);
        }
    }



    private double[] flatten(List<double[][]> input) {
        int totalElements = input.size() * input.get(0).length * input.get(0)[0].length;
        double[] flattened = new double[totalElements];
        int index = 0;

        for (double[][] matrix : input) {
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    flattened[index++] = matrix[i][j];
                }
            }
        }

        return flattened;
    }

    @Override
    public int getOutputLength() {
        return outputLength;
    }

    @Override
    public int getOutputRows() {
        return 1;
    }

    @Override
    public int getOutputCols() {
        return outputLength;
    }

    @Override
    public int getOutputElements() {
        return outputLength;
    }
}
