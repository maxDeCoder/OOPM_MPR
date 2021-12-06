package budget_TF;
public abstract class Optimizer {
    /**
     * Function to optimize. Must return a double which is the loss value.
     * @param m Model: Model to optimizer
     * @param y double: Target value
     * @param loss_function LossFunction: Loss function to use
     * @return double: Loss value
     * @throws IncompatibleTensorException
     */
    public abstract double optimize(Model m, Matrix y, Loss loss_function) throws IncompatibleTensorException;
}
