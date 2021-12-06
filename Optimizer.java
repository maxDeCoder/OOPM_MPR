package budget_TF;
public abstract class Optimizer {
    public abstract double optimize(Model m, Matrix y, Loss loss_function) throws IncompatibleTensorException;
}
