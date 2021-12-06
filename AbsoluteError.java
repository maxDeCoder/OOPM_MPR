package budget_TF;

import java.util.ArrayList;

public class AbsoluteError extends Loss {
    public Matrix calcLoss(Matrix y, Matrix yHat) throws IncompatibleTensorException {
        if (y.getRows() != yHat.getRows() || y.getCols() != yHat.getCols()) {
            throw new IncompatibleTensorException("y and yHat must have the same dimensions");
        }
        Matrix loss = new Matrix(y.getRows(), y.getCols());
        for (int i = 0; i < y.getRows(); i++) {
            for (int j = 0; j < y.getCols(); j++) {
                loss.set(i, j, Math.abs(y.get(i, j) - yHat.get(i, j)));
            }
        }
        return loss;
    }

    private double diffAbs(double input) {
        return input >= 0 ? input : -input;
    }

    private Matrix diffAbs(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                output.set(i, j, diffAbs(input.get(i, j)));
            }
        }
        return output;
    }

    /**
     * Function that will calculate the gradients for one output and ground truth
     * value. Make sure to return the for value in the as delta w and second as
     * delta b.
     * @param w                   Matrix: weights
     * @param z                   Matrix: input values
     * @param a                   Matrix: activation values
     * @param y                   Matrix: output values
     * @param activation_function ActivationFunction: activation function
     * @return ArrayList<ArrayList<Matrix>>: gradients of length 2. gradients[0] ->
     *         delta w and gradients[1] -> delta b
     * @throws IncompatibleTensorException
     */
    public ArrayList<ArrayList<Matrix>> calcGrad(ArrayList<Matrix> w, ArrayList<Matrix> z, ArrayList<Matrix> a,
            Matrix y, Activation activation_function) throws IncompatibleTensorException {
        ArrayList<ArrayList<Matrix>> gradients = new ArrayList<ArrayList<Matrix>>();
        ArrayList<Matrix> delta_w = new ArrayList<Matrix>();
        ArrayList<Matrix> delta_b = new ArrayList<Matrix>();

        int num_layers = w.size();

        // calculate dz for the last layer
        // remember that a.get(0) is the input vector to the matrix, so adjust the inedx
        // value for a accordingly
        Matrix output = a.get(num_layers);
        Matrix error = Matrix.subtract(output, y);
        Matrix dw = new Matrix(0, 0), db = new Matrix(0, 0);
        for (int i = num_layers - 1; i >= 0; i--) {
            if (i == num_layers - 1) {
                error = Matrix.multiply(error, activation_function.derivative(a.get(i + 1)));
            } else {
                error = Matrix.multiply(error.transpose(), w.get(i + 1));
                error = Matrix.multiply(activation_function.derivative(a.get(i + 1)).transpose(), error);
            }
            // error.print();
            dw = Matrix.multiply(error, a.get(num_layers - 1).transpose());
            dw.print();
            db = error.copy();
            delta_w.add(dw);
            delta_b.add(db);
        }

        for( Matrix m : delta_w){
        m.print();
        }

        // System exit
        System.exit(0);

        gradients.add(delta_w);
        gradients.add(delta_b);

        return gradients;
    }
}
