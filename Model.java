package budget_TF;

import java.util.ArrayList;
import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;

public class Model {
    private ArrayList<Matrix> activations;
    private ArrayList<Matrix> z;
    private int num_layers;
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Loss loss_function;
    private double loss_value;

    /**
     * Constructor for the model
     * 
     * @param layers ArrayList<Layer>: the layers of the model
     */
    public Model(ArrayList<Layer> layers) {
        this.layers = layers;
        this.num_layers = layers.size();
        this.activations = new ArrayList<Matrix>();
        this.z = new ArrayList<Matrix>();
    }

    public Model() {
        this.layers = new ArrayList<Layer>();
        this.num_layers = 0;
        this.activations = new ArrayList<Matrix>();
        this.z = new ArrayList<Matrix>();
    }

    /**
     * Function to add a layer to the model
     * 
     * @param layer Layer: the layer to be added
     */
    public void addLayer(Layer layer) {
        this.layers.add(layer);
        this.num_layers++;
    }

    /**
     * Function to compile the model
     * 
     * @param optimizer     Optimizer: the optimizer to be used
     * @param loss_function Loss: the loss function to be used
     */
    public void compile(Optimizer optimizer, Loss loss_function) {
        this.loss_function = loss_function;
        this.optimizer = optimizer;
    }

    /**
     * Function to set weights
     * 
     * @param weights ArrayList<Matrix>: the weights to be set
     */
    public void setWeights(ArrayList<Matrix> weights) {
        // iterate over the layers and set the weights to appropriate layers
        for (int i = 0; i < num_layers; i++) {
            layers.get(i).setWeights(weights.get(i));
        }
    }

    /**
     * Function to set biases
     * 
     * @param biases ArrayList<Matrix>: the biases to be set
     */
    public void setBiases(ArrayList<Matrix> biases) {
        // iterate over the layers and set the biases to appropriate layers
        for (int i = 0; i < num_layers; i++) {
            layers.get(i).setBiases(biases.get(i));
        }
    }

    /**
     * Function to return weights
     * 
     * @return ArrayList<Matrix>: the weights
     */
    public ArrayList<Matrix> getWeights() {
        ArrayList<Matrix> weights = new ArrayList<Matrix>();
        for (int i = 0; i < num_layers; i++) {
            weights.add(layers.get(i).getWeights());
        }
        return weights;
    }

    /**
     * Function to return biases
     * 
     * @return ArrayList<Matrix>: the biases
     */
    public ArrayList<Matrix> getBiases() {
        ArrayList<Matrix> biases = new ArrayList<Matrix>();
        for (int i = 0; i < num_layers; i++) {
            biases.add(layers.get(i).getBiases());
        }
        return biases;
    }

    /**
     * Function to return activations
     * 
     * @return ArrayList<Matrix>: the activations
     */
    public ArrayList<Matrix> getActivations() {
        return activations;
    }

    /**
     * Function to return z
     * 
     * @return ArrayList<Matrix>: the z
     */
    public ArrayList<Matrix> getZ() {
        return z;
    }

    /**
     * Function to get num_layers
     * 
     * @return int: the number of layers
     */
    public int getNumLayers() {
        return num_layers;
    }

    /**
     * Function to get layers
     * 
     * @return ArrayList<Layer>: the layers
     */
    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public Layer getLayer(int i) {
        return layers.get(i);
    }

    /**
     * Function to forward propogate the model
     * @param inputs Matrix: the input to the model
     * @throws IncompatibleTensorException
     */
    private void forwardProp(Matrix inputs) throws IncompatibleTensorException {
        activations = new ArrayList<Matrix>();
        z = new ArrayList<Matrix>();
        activations.add(inputs);
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.forwardPropagation(inputs);
            inputs = layer.getActivations();
            activations.add(inputs);
            z.add(layer.getZ());
        }
    }

    /**
     * Function to back propogate the model
     * 
     * @param y Matrix: the expected output
     */
    private void optimize(Matrix y) throws IncompatibleTensorException {
        this.loss_value = optimizer.optimize(this, y, loss_function);
    }

    /**
     * Function to return loss value
     * 
     * @return double: the loss value
     */
    public double getLossValue() {
        return this.loss_value;
    }

    /**
     * Function to print model summary
     */
    public void printModelSummary() {
        if (optimizer != null) {
            System.out.println("Model Summary:");
            // iterate through layers and print the input shape and output shape in a table
            // format
            for (int i = 0; i < layers.size(); i++) {
                System.out.print("Layer " + i + ":");
                System.out.print("\tNum units: " + layers.get(i).getUnits());
                System.out.print("\tInput Shape: " + layers.get(i).getInputShape());
                System.out.print("\tOutput Shape: " + layers.get(i).getOutputShape());
                System.out.println("\tActivation Function: " + layers.get(i).getActivationFunction().getName());
            }
        } else {
            throw new IllegalArgumentException("Model has not been compiled");
        }
    }

    /**
     * Function to train the model on one Input and one Output
     * 
     * @param inputs Matrix: the input to the model
     * @param y      Matrix: the expected output
     */
    private void train(Matrix inputs, Matrix y) throws IncompatibleTensorException {
        this.forwardProp(inputs);
        this.optimize(y);
    }

    /**
     * Function to evaluate the model on one Input and one Output
     * 
     * @param inputs Matrix: the input to the model
     * @param y      Matrix: the expected output
     */
    private void evaluate(Matrix inputs, Matrix y) throws IncompatibleTensorException {
        this.forwardProp(inputs);
        this.loss_value = this.loss_function.calcLoss(inputs, y).get(0, 0);
    }

    /**
     * Function to train the model
     * @param x       Matrix: the input to the model
     * @param y       Matrix: the expected output
     * @param epochs  int: the number of epochs to train the model
     * @param verbose boolean: whether to print the loss value
     * @throws IncompatibleTensorException
     */
    public void train(ArrayList<Matrix> inputs, ArrayList<Matrix> y, int epochs, boolean verbose)
            throws IncompatibleTensorException {
        for (int j = 0; j < epochs; j++) {
            for (int i = 0; i < inputs.size(); i++) {
                this.train(inputs.get(i), y.get(i));
            }
            if (verbose) {
                System.out.println("Epoch " + j + ": Loss = " + this.getLossValue());
            }
        }
    }

    /**
     * Function that calculates the loss of the model
     * 
     * @param x       Matrix: the input to the model
     * @param y       Matrix: the expected output
     * @param verbose boolean: whether to print the loss value
     * @throws IncompatibleTensorException
     */
    public double evaluate(ArrayList<Matrix> inputs, ArrayList<Matrix> y, boolean verbose)
            throws IncompatibleTensorException {
        for (int i = 0; i < inputs.size(); i++) {
            this.evaluate(inputs.get(i), y.get(i));
        }
        if (verbose) {
            System.out.println("Loss = " + this.getLossValue());
        }
        return this.loss_value;
    }

    /**
     * Function to predict the output of the model
     * 
     * @param inputs Matrix: the input to the model
     * @return Matrix: the predicted output
     * @throws IncompatibleTensorException
     */
    public Matrix predict(Matrix inputs) throws IncompatibleTensorException {
        this.forwardProp(inputs);
        return this.activations.get(this.activations.size() - 1);
    }

    /**
     * Function to predict the output of the mode
     * @param inputs ArrayList<Matrix>: the input to the model
     * @return ArrayList<Matrix>: the predicted output
     * @throws IncompatibleTensorException
     */
    public ArrayList<Matrix> predict(ArrayList<Matrix> inputs) throws IncompatibleTensorException {
        ArrayList<Matrix> predictions = new ArrayList<Matrix>();
        for (int i = 0; i < inputs.size(); i++) {
            predictions.add(this.predict(inputs.get(i)));
        }
        return predictions;
    }

    /**
     * Function to save the model to a txt file
     * 
     * @param path String: the file to save the model to
     * @throws IOException
     */
    public void saveModel(String path) throws IOException {
        File layer_info_file = new File(path + "\\_layer_info.txt");
        File weights_file = new File(path + "\\_weights.txt");

        // write layer info to file
        FileWriter layer_info_writer = new FileWriter(layer_info_file);

        layer_info_writer.write(this.num_layers + "\n");

        for (int i = 0; i < this.layers.size(); i++) {
            layer_info_writer.write(this.layers.get(i).getUnits() + "\n");
            layer_info_writer.write(this.layers.get(i).getActivationFunction().getName() + "\n");
            layer_info_writer.write(
                    this.layers.get(i).getInputShape().get(0) + " " + this.layers.get(i).getInputShape().get(1) + "\n");
        }

        layer_info_writer.close();

        // write weights to file
        FileWriter weights_writer = new FileWriter(weights_file);

        // write the value of every weight on a new line. Add * as a seperator between
        // weights and bias. Add _ at the end of each layer
        for (int i = 0; i < this.layers.size(); i++) {
            System.out.println(i);
            Matrix weights = this.getLayer(i).getWeights();
            Matrix bias = this.getLayer(i).getBiases();
            weights = weights.flatten(1);
            weights = bias.flatten(1);

            for (int j = 0; j < weights.shape.get(0); j++) {
                weights_writer.write(weights.get(j, 0) + "\n");
            }

            for (int j = 0; j < bias.shape.get(0); j++) {
                weights_writer.write(bias.get(j, 0) + "\n");
            }
        }

        weights_writer.close();
    }

    // use the above function as template to load the model from a txt file
    /**
     * Function to load the model from a txt file
     * 
     * @param path String: directory where the layers info and weights are saved.
     *             Use the Model.save(path) function to save the model
     * @throws IOException
     * @throws InvalidActivationNameException
     */

    public void loadModel(String path) throws IOException, InvalidActivationNameException {
        File layer_info_file = new File(path + "\\_layer_info.txt");
        File weights_file = new File(path + "\\_weights.txt");

        // read layer info from file
        FileReader layer_info_reader = new FileReader(layer_info_file);
        BufferedReader layer_info_buffer = new BufferedReader(layer_info_reader);

        String line = layer_info_buffer.readLine();
        int temp_num_layers = Integer.parseInt(line);

        // initialize all layers
        for (int i = 0; i < temp_num_layers; i++) {

            int units = 0;
            String activation_function = "";
            int[] shape = new int[2];

            for (int j = 0; j < 3; j++) {
                line = layer_info_buffer.readLine();
                switch (j) {
                    case 0:
                        units = Integer.parseInt(line);
                        break;
                    case 1:
                        activation_function = line;
                        break;
                    case 2:
                        String[] shape_string = line.split(" ");
                        shape[0] = Integer.parseInt(shape_string[0]);
                        shape[1] = Integer.parseInt(shape_string[1]);
                        break;
                }
            }
            Activation activation = Activation.getActivation(activation_function);

            Layer temp = new Layer(units, shape, activation);

            this.addLayer(temp);
        }

        layer_info_buffer.close();
        // initialize weights and bias
        FileReader weights_reader = new FileReader(weights_file);
        BufferedReader weights_buffer = new BufferedReader(weights_reader);

        for (int l = 0; l < this.layers.size(); l++) {
            Matrix weights = new Matrix(this.layers.get(l).getUnits(), this.layers.get(l).getInputShape().get(0));
            Matrix bias = new Matrix(this.layers.get(l).getUnits(), 1);

            for(int m = 0;m < weights.shape.get(0);m++){
                for(int n = 0;n < weights.shape.get(1);n++){
                    line = weights_buffer.readLine();
                    double value = Double.parseDouble(line);

                    weights.set(m, n, value);
                }
            }

            for (int m = 0; m < bias.shape.get(0); m++) {
                line = weights_buffer.readLine();
                bias.set(m, 0, Double.parseDouble(line));
            }

            this.layers.get(l).setWeights(weights);
            this.layers.get(l).setBiases(bias);
        }

        weights_buffer.close();
    }
}