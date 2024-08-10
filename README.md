## Welcome to CNN framework ##

The network is designed to classify images from the MNIST dataset. The project includes custom implementations of convolutional layers, fully connected layers, max pooling, and backpropagation.

How It Works: 

Matrix Operations: The MatrixUtility class handles element-wise addition and scalar multiplication for matrices and vectors, crucial for calculations during forward and backward passes in the network.

NetworkBuilder: Simplifies neural network construction by allowing the addition of layers while ensuring they are properly linked, setting up the network's structure based on input dimensions and scale factors.

NeuralNetwork Class:
Forward Pass: Processes input data through each layer, passing the output of one layer as the input to the next.
Prediction: The makePrediction method identifies the predicted class by selecting the index of the highest output value.
Accuracy Assessment: The assessAccuracy method calculates accuracy by comparing predictions with true labels.
Backward Pass: Updates network weights during training by calculating gradients, managed by the conductTraining method.

Layers:
ConvolutionLayer: Detects patterns like edges and textures by applying filters to the input.
MaxPoolLayer: Reduces spatial dimensions while retaining key features, optimizing computational efficiency.
FullyConnectedLayer: Flattens input and processes it through neurons, typically for final classification.

