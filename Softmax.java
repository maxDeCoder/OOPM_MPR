package budget_TF;
public class Softmax extends Activation {

    private String name = "softmax";

    private double sumExp(Matrix values){
        double sum = 0;
        for(int i = 0; i < values.shape.get(0); i++){
            for(int j = 0; j < values.shape.get(1); j++){
                sum += Math.exp(values.get(i, j));
            }
        }
        return sum;
    }
    
    public Matrix activation(Matrix x) {
        double expSums = sumExp(x);
        Matrix result = new Matrix(x.shape);

        for(int i = 0; i < x.shape.get(0); i++){
            for(int j = 0; j < x.shape.get(1); j++){
                result.set(i, j, Math.exp(x.get(i, j)) / expSums);
            }
        }

        return result;
    }

    public Matrix derivative(Matrix x) {
        Matrix result = new Matrix(x.shape);

        for(int i = 0; i < x.shape.get(0); i++){
            for(int j = 0; j < x.shape.get(1); j++){
                result.set(i, j, x.get(i, j) * (1 - x.get(i, j)));
            }
        }

        return result;
    }

    public String getName() {
        return name;
    }
}
