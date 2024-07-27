package network;

import Layer.ConvolutionLayer;
import Layer.FullyConnectedLayer;
import Layer.Layer;
import Layer.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private NeuralNetwork neuralNetwork;
    private int inputRows;
    private int inputCols;
    private double scaleFactor;
    private List<Layer> layers;

    public NetworkBuilder(int inputRows, int inputCols, double scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
        this.layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed) {
        Layer previousLayer = getPreviousLayer();
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(filterSize, stepSize, getOutputLength(previousLayer), getOutputRows(previousLayer), getOutputCols(previousLayer), seed, numFilters, learningRate);
        layers.add(convolutionLayer);
    }

    public void addMaxPoolLayer(int windowSize, int stepSize) {
        Layer previousLayer = getPreviousLayer();
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(stepSize, windowSize, getOutputLength(previousLayer), getOutputRows(previousLayer), getOutputCols(previousLayer));
        layers.add(maxPoolLayer);
    }

    public void addFullyConnectedLayer(int outputLength, double learningRate, long seed) {
        Layer previousLayer = getPreviousLayer();
        FullyConnectedLayer fullyConnectedLayer = new MyFullyConnectedLayer(getOutputElements(previousLayer), outputLength, seed, learningRate); // Instantiate your subclass here
        layers.add(fullyConnectedLayer);
    }

    private Layer getPreviousLayer() {
        return layers.isEmpty() ? null : layers.get(layers.size() - 1);
    }

    private int getOutputLength(Layer layer) {
        return layer == null ? 1 : layer.getOutputLength();
    }

    private int getOutputRows(Layer layer) {
        return layer == null ? inputRows : layer.getOutputRows();
    }

    private int getOutputCols(Layer layer) {
        return layer == null ? inputCols : layer.getOutputCols();
    }

    private int getOutputElements(Layer layer) {
        if (layer instanceof ConvolutionLayer) {
            return ((ConvolutionLayer) layer).getOutputElements();
        } else if (layer instanceof MaxPoolLayer) {
            return ((MaxPoolLayer) layer).getOutputElements();
        } else if (layer instanceof FullyConnectedLayer) {
            return ((FullyConnectedLayer) layer).getOutputElements();
        } else {
            throw new IllegalArgumentException("Unsupported layer type");
        }
    }

    public NeuralNetwork build() {
        neuralNetwork = new NeuralNetwork(layers, scaleFactor);
        return neuralNetwork;
    }

    // Example subclass of FullyConnectedLayer
    private static class MyFullyConnectedLayer extends FullyConnectedLayer {

        public MyFullyConnectedLayer(int inputLength, int outputLength, long seed, double learningRate) {
            super(inputLength, outputLength, seed, learningRate);
        }

        @Override
        public double[] getOutput(List<double[][]> input) {
            // Implement your custom logic here
            return null;
        }

        @Override
        public double[] getOutput(double[] input) {
            // Implement your custom logic here
            return null;
        }

        @Override
        public void backPropagation(double[] dLdO) {
            // Implement your custom logic here
        }

        @Override
        public void backPropagation(List<double[][]> dLdO) {
            // Implement your custom logic here
        }

        @Override
        public int getOutputLength() {
            // Implement your custom logic here
            return 0;
        }

        @Override
        public int getOutputRows() {
            // Implement your custom logic here
            return 0;
        }

        @Override
        public int getOutputCols() {
            // Implement your custom logic here
            return 0;
        }

        @Override
        public int getOutputElements() {
            // Implement your custom logic here
            return 0;
        }
    }
}
