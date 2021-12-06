package budget_TF;

public class Tanh extends Activation {

    private static String name = "tanh";    

    private double activation(double x) {
        return Math.tanh(x);
    }

    private double derivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }

    public String getName() {
        return name;
    }

    public Matrix activation(Matrix x) {
        Matrix result = new Matrix(x.shape);
        for (int i = 0; i < x.shape.get(0); i++) {
            for (int j = 0; j < x.shape.get(1); j++) {
                result.set(i, j, activation(x.get(i, j)));
            }
        }
        return result;
    }

    public Matrix derivative(Matrix x) {
        Matrix result = new Matrix(x.shape);
        for (int i = 0; i < x.shape.get(0); i++) {
            for (int j = 0; j < x.shape.get(1); j++) {
                result.set(i, j, derivative(x.get(i, j)));
            }
        }
        return result;
    }
    
}
