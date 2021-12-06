package budget_TF;
import java.util.ArrayList;

public class MeanSquaredError extends Loss{

    private String functionName = "Mean Squared Error";

    public Matrix calcLoss(Matrix y, Matrix y_hat) throws IncompatibleTensorException{
        int n = y.shape.get(0);
        Matrix loss = new Matrix(y.shape);

        loss = Matrix.subtract(y, y_hat);
        loss.pow(2);
        loss = loss.sumColumns();
        loss = Matrix.divide(loss, n);
        return loss;
    }

    public ArrayList<ArrayList<Matrix>> calcGrad(ArrayList<Matrix> w, ArrayList<Matrix> z, ArrayList<Matrix> a, Matrix y, Activation activation_function) throws IncompatibleTensorException{
        // int n = number of layers in the model
        int n = w.size();

        ArrayList<Matrix> dw = new ArrayList<Matrix>();
        ArrayList<Matrix> db = new ArrayList<Matrix>();
        Matrix y_hat = a.get(n);
        
        Matrix dz = Matrix.subtract(y_hat, y);

        dz = Matrix.multiply(dz, activation_function.derivative(z.get(n-1)));
        dz = Matrix.multiply(dz, 2/y.shape.get(0));
        Matrix db_temp = dz.copy();
        // convert dz from [1, 1] to the same dimensiona as the last weights matrix
        
        Matrix dw_temp = Matrix.multiply(dz, a.get(n-2).transpose()).expandNew(1, w.get(n-1).shape.get(1)-1);

        dw.add(dw_temp);
        db.add(db_temp);

        dz.expand(1, w.get(n-1).shape.get(1)-1);
        Matrix dz_new = dz.copy();
        for(int i = n-2;i >= 0;i--){
            dz_new = Matrix.multiply(w.get(i+1), dz_new);
            dz_new = Matrix.multiply(dz_new, activation_function.derivative(z.get(i)).transpose());

            dw_temp = Matrix.multiply(dz_new, a.get(i+1).transpose());
            db_temp = dz_new;

            dw.add(dw_temp.transpose());
            db.add(db_temp.transpose());
            dz = dz_new;
        }

        // create arraylist that contains the gradients dw and db
        ArrayList<ArrayList<Matrix>> grads = new ArrayList<ArrayList<Matrix>>();
        grads.add(dw);
        grads.add(db);

        return grads;
    }

    public String getFunctionName(){
        return functionName;
    }
}