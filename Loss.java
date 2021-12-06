package budget_TF;
import java.util.ArrayList;

public abstract class Loss {
    /**
     * Function that will calculate the loss for one output and ground truth value.
     * @param y Matrix: output values
     * @param yHat Matrix: ground truth values
     * @return Matrix: loss values
     * @throws IncompatibleTensorException
     */
    public abstract Matrix calcLoss(Matrix y, Matrix yHat) throws IncompatibleTensorException;

    /**
     * Function that will calculate the gradients for one output and ground truth value. Make sure to return the for value in the as delta w and second as delta b.
     * @param w Matrix: weights
     * @param z Matrix: input values
     * @param a Matrix: activation values
     * @param y Matrix: output values
     * @param activation_function ActivationFunction: activation function
     * @return ArrayList<ArrayList<Matrix>>: gradients of length 2. gradients[0] -> delta w and gradients[1] -> delta b
     * @throws IncompatibleTensorException
     */
    public abstract ArrayList<ArrayList<Matrix>> calcGrad(ArrayList<Matrix> w, ArrayList<Matrix> z, ArrayList<Matrix> a, Matrix y, Activation activation_function) throws IncompatibleTensorException;
}
