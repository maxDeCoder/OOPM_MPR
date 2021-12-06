package budget_TF;
public class ReLU extends Activation {

    private String name = "relu";

    private double activate(double x) {
        return Math.max(0, x);
    }

    private double derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public Matrix activation(Matrix x) {
        Matrix activated = new Matrix(x.shape);
        for (int i = 0;i < x.shape.get(0);i++){
            for(int j = 0;j < x.shape.get(1);j++){
                activated.set(i, j, activate(x.get(i, j)));
            }
        }

        return activated;
    }

    public Matrix derivative(Matrix x) {
        Matrix activated = new Matrix(x.shape);
        for (int i = 0;i < x.shape.get(0);i++){
            for(int j = 0;j < x.shape.get(1);j++){
                activated.set(i, j, derivative(x.get(i, j)));
            }
        }

        return activated;
    }

    public String getName() {
        return name;
    }
}
