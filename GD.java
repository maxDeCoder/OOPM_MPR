package budget_TF;
import java.util.ArrayList;

public class GD extends Optimizer {
    private double learning_rate;

    public GD(){
        learning_rate = 0.01;
    }

    /**
     * Constructor for the GD optimizer
     * @param lr double: learning rate
     */
    public GD(double lr){
        learning_rate = lr;
    }

    /**
     * Function to set the learning rate
     * @param lr
     */
    public void setLearningRate(double lr){
        learning_rate = lr;
    }


    /**
     * Function to return the learning rate
     */
    public double getLearningRate(){
        return learning_rate;
    }


    // Function to calculate gradients and update weights
    /**
     * Function to calculate gradients and update weights
     * @param m Model: the model to be trained
     * @param y Matrix: the target values
     * @param loss_function LossFunction: the loss function to be used
     */
    public double optimize(Model m, Matrix y, Loss loss_function) throws IncompatibleTensorException{
        ArrayList<Matrix> w = m.getWeights();
        ArrayList<Matrix> a = m.getActivations();
        ArrayList<Matrix> z = m.getZ();
        ArrayList<Layer> layers = m.getLayers();

        ArrayList<ArrayList<Matrix>> grads = loss_function.calcGrad(w, z, a, y, layers.get(m.getNumLayers()-1).getActivationFunction());
        ArrayList<Matrix> dws = grads.get(0);
        ArrayList<Matrix> dbs = grads.get(1);

        // multiply learning rate by all matrices in dws and dbs
        for(int i = 0; i < dws.size(); i++){
            dws.set(i, dws.get(i).multiply(learning_rate));
        }
        for(int i = 0; i < dbs.size(); i++){
            dbs.set(i, dbs.get(i).multiply(learning_rate));
        }
        int num_layers = m.getNumLayers();
        for(int i = 0; i < num_layers; i++){
            // w.get(i).print();
            // // System.out.println();
            // dws.get(num_layers-i-1).print();
            w.set(i, w.get(i).subtract(dws.get(num_layers-i-1)));
        }

        // subtract dbs from b
        ArrayList<Matrix> old_biases = m.getBiases();

        for (int i = 0; i < num_layers; i++){
            old_biases.set(i, old_biases.get(i).subtract(dbs.get(num_layers-i-1)));
        }

        // set model parameters to new values
        m.setWeights(w);
        m.setBiases(old_biases);

        double loss = loss_function.calcLoss(a.get(m.getNumLayers()), y).get(0,0);

        return loss;
    }
    
}