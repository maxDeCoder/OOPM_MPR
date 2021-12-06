package budget_TF;
import java.util.ArrayList;

// class that represents a layer of an ML model
public class Layer {
    private int units = 0;
    private Matrix weights = null;
    private Matrix bias = null;
    private Matrix input = null;
    private Matrix z = null;
    private Matrix activations = null;
    private Activation activation_function = new ReLU(); // default activation function is relu (linear regression)
    private ArrayList<Integer> input_shape;


    // constructor that takes in the number of units and an input shape, initializes the weights and baises to random values
    public Layer(int units, int[] inputShape, Activation activation_function) {
        this.units = units;
        this.weights = new Matrix(units, inputShape[0]);
        this.bias = new Matrix(units, 1);
        this.weights.randomize();
        this.bias.randomize();

        //set input shape
        input_shape = new ArrayList<Integer>();
        for(int i : inputShape) {
            this.input_shape.add(i);
        }

        //set activation function
        this.activation_function = activation_function;
    }

    // constructor that takes in the number of units and an input shape, initializes the weights and baises to random values
    public Layer(int units, int[] inputShape) {
        this.units = units;
        this.weights = new Matrix(units, inputShape[0]);
        this.bias = new Matrix(units, 1);
        this.weights.randomize();
        this.bias.randomize();
        this.input_shape = (ArrayList<Integer>) input_shape;
    }

    // default constructor that throws error because number of units must be specified
    public Layer() {
        throw new IllegalArgumentException("Number of units must be specified");
    }

    // constructor that takes in the number of units but throws error because input shape was not given
    public Layer(int units) {
        throw new IllegalArgumentException("Input shape must be specified");
    }

    // function that returns the weights of the layer
    public Matrix getWeights() {
        return this.weights;
    }

    // function that returns the bias of the layer
    public Matrix getBiases() {
        return this.bias;
    }

    // function that returns the number of units in the layer
    public int getUnits() {
        return this.units;
    }

    // function that returns the input of the layer
    public Matrix getInput() {
        return this.input;
    }

    // function that returns the z values of the layer
    public Matrix getZ() {
        return this.z;
    }

    // function that returns the activations of the layer
    public Matrix getActivations() {
        return this.activations;
    }

    // function that returns activation function
    public Activation getActivationFunction() {
        return this.activation_function;
    }

    // function that sets the weights of the layer
    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    // function that sets the bias of the layer
    public void setBiases(Matrix bias) {
        this.bias = bias;
    }

    // function to calculate the z values of the layer
    public void calculateZ(Matrix input)throws IncompatibleTensorException {
        this.input = input;
        this.z = Matrix.dot(weights, input);
        this.z.add(this.bias);
    }

    // function to calculate activations
    public void calculateActivations() {
        this.activations = this.activation_function.activation(this.z);
    }

    // function for forward propagation
    public void forwardPropagation(Matrix input) throws IncompatibleTensorException{
        this.calculateZ(input);
        this.calculateActivations();
    }

    // function to return input shape
    public ArrayList<Integer> getInputShape() {
        return this.input_shape;
    }

    // function to return output shape
    public ArrayList<Integer> getOutputShape() {
        ArrayList<Integer> output_shape = new ArrayList<Integer>();
        output_shape.add(this.units);
        output_shape.add(1);
        return output_shape;
    }
}
