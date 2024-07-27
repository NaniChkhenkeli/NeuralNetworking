package Layer;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    private Layer nextLayer;
    private Layer previousLayer;

    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(double[] dLdO);

    public abstract void backPropagation(List<double[][]> dLdO);

    public abstract int getOutputLength();

    public abstract int getOutputRows();

    public abstract int getOutputCols();

    public abstract int getOutputElements();

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    protected List<double[][]> vectorToMatrix(double[] input, int inputLength, int inputRows, int inputCols) {
        List<double[][]> matrix = new ArrayList<>();

        int index = 0;
        for (int l = 0; l < inputLength; l++) {
            double[][] layer = new double[inputRows][inputCols];
            for (int i = 0; i < inputRows; i++) {
                for (int j = 0; j < inputCols; j++) {
                    layer[i][j] = input[index++];
                }
            }
            matrix.add(layer);
        }

        return matrix;
    }
}
