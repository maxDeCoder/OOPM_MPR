package budget_TF;
public abstract class Activation {
    public abstract Matrix activation(Matrix x);
    public abstract Matrix derivative(Matrix x);
    public abstract String getName();

    public static Activation getActivation(String name) throws InvalidActivationNameException {
        switch(name) {
            case "sigmoid":
                return new Sigmoid();
            case "relu":
                return new ReLU();
            case "tanh":
                return new Tanh();
            case "softmax":
                return new Softmax();
            default:
                throw new InvalidActivationNameException("Invalid activation name: " + name);
        }
    }
}