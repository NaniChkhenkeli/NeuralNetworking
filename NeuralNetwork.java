package network;

import data.Image;
import Layer.Layer;

import java.util.List;

public class NeuralNetwork {

    private List<Layer> layers;
    private double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {
        if (layers.size() <= 1) {
            return;
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer currentLayer = layers.get(i);
            Layer previousLayer = i > 0 ? layers.get(i - 1) : null;
            Layer nextLayer = i < layers.size() - 1 ? layers.get(i + 1) : null;
            currentLayer.setPreviousLayer(previousLayer);
            currentLayer.setNextLayer(nextLayer);
        }
    }

    public double[] calculateErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;
        return data.MatrixUtility.add(networkOutput, data.MatrixUtility.multiply(expected, -1));
    }

    private int findMaxIndex(double[] input) {
        double max = 0;
        int index = 0;

        for (int i = 0; i < input.length; i++) {
            if (input[i] >= max) {
                max = input[i];
                index = i;
            }
        }

        return index;
    }

    public int makePrediction(Image image) {
        List<double[][]> inputList = List.of();
        double[] output = layers.get(0).getOutput(inputList);
        return findMaxIndex(output);
    }

    public float assessAccuracy(List<Image> images) {
        int correct = 0;

        for (Image img : images) {
            int prediction = makePrediction(img);

            if (prediction == img.getLabel()) {
                correct++;
            }
        }

        return (float) correct / images.size();
    }

    public void conductTraining(List<Image> images) {
        for (Image img : images) {
            List<double[][]> inputList = List.of();
            double[] output = layers.get(0).getOutput(inputList);
            double[] dldO = calculateErrors(output, img.getLabel());
            layers.get(layers.size() - 1).backPropagation(dldO);
        }
    }
}
