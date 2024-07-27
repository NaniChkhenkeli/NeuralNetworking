package Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class ConvolutionLayer extends Layer {

    private long seed;

    private List<double[][]> filters;
    private int filterSize;
    private int stepSize;

    private int inputLength;
    private int inputRows;
    private int inputCols;
    private double learningRate;

    private List<double[][]> lastInput;

    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, long seed, int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.seed = seed;
        this.learningRate = learningRate;

        generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters) {
        filters = new ArrayList<>();
        Random random = new Random(seed);

        for (int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[filterSize][filterSize];

            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);
        }
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> input) {
        lastInput = input;
        List<double[][]> output = new ArrayList<>();

        for (double[][] singleInput : input) {
            for (double[][] filter : filters) {
                output.add(convolve(singleInput, filter, stepSize));
            }
        }

        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;
        int inRows = input.length;
        int inCols = input[0].length;
        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i <= inRows - fRows; i += stepSize) {
            for (int j = 0; j <= inCols - fCols; j += stepSize) {
                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;
                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum += value;
                    }
                }

                output[i / stepSize][j / stepSize] = sum;
            }
        }

        return output;
    }

    public double[][] spaceArray(double[][] input) {
        if (stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * stepSize][j * stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);
        return getNextLayer().getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, inputLength, inputRows, inputCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, inputLength, inputRows, inputCols);
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for (int f = 0; f < filters.size(); f++) {
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for (int i = 0; i < lastInput.size(); i++) {
            double[][] errorForInput = new double[inputRows][inputCols];

            for (int f = 0; f < filters.size(); f++) {
                double[][] currFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);
                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, - learningRate);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currFilter, flippedError));
            }

            dLdOPreviousLayer.add(errorForInput);
        }

        for (int f = 0; f < filters.size(); f++) {
            double[][] modified = add(filtersDelta.get(f), filters.get(f));
            filters.set(f, modified);
        }

        if (getPreviousLayer() != null) {
            getPreviousLayer().backPropagation(dLdOPreviousLayer);
        }
    }

    public double[][] flipArrayHorizontal(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[rows - i - 1][j] = array[i][j];
            }
        }
        return output;
    }

    public double[][] flipArrayVertical(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }
        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outRows = input.length + filter.length - 1;
        int outCols = input[0].length + filter[0].length - 1;
        int inRows = input.length;
        int inCols = input[0].length;
        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        for (int i = -fRows + 1; i < inRows; i++) {
            for (int j = -fCols + 1; j < inCols; j++) {
                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols) {
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[i + fRows - 1][j + fCols - 1] = sum;
            }
        }

        return output;
    }

    @Override
    public int getOutputLength() {
        return filters.size() * inputLength;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }
}
