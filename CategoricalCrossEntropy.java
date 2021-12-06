package budget_TF;
import java.util.ArrayList;

public class CategoricalCrossEntropy extends Loss{
    public Matrix calcLoss(Matrix y, Matrix y_hat) throws IncompatibleTensorException{
        y_hat.log(); 
        y = Matrix.multiply(y, y_hat);
        y.sumColumns();

        return y;
    }

    public ArrayList<ArrayList<Matrix>> calcGrad(ArrayList<Matrix> w, ArrayList<Matrix> z, ArrayList<Matrix> a, Matrix y, Activation activation_function) throws IncompatibleTensorException{
        // TODO: implement this function please

        return null;
    }
}