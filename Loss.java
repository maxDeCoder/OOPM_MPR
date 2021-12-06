package budget_TF;
import java.util.ArrayList;

public abstract class Loss {
    public abstract Matrix calcLoss(Matrix y, Matrix yHat) throws IncompatibleTensorException;
    public abstract ArrayList<ArrayList<Matrix>> calcGrad(ArrayList<Matrix> w, ArrayList<Matrix> z, ArrayList<Matrix> a, Matrix y, Activation activation_function) throws IncompatibleTensorException;
}
