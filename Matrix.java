package budget_TF;
import java.util.ArrayList;

public class Matrix{
    public ArrayList<Integer> shape = new ArrayList<Integer>();
    
    // create 2 dimentional arraylist for storing values of the matrix
    // this is a property of the class
    private ArrayList<ArrayList<Double>> values = new ArrayList<ArrayList<Double>>();

    /**
     * Contructor that takes shape of matrix as arraylist
     * @param shape ArrayList<Integer>: shape of matrix
     */
    public Matrix(ArrayList<Integer> shape){
        this.shape = shape;
        for(int i = 0; i < shape.get(0); i++){
            values.add(new ArrayList<Double>());
            for(int j = 0; j < shape.get(1); j++){
                values.get(i).add(0.0);
            }
        }
    }

    //constructor that takes shape of matrix as int array
    /**
     * Contructor that takes shape of matrix as int array
     * @param shape int[]: shape of matrix
     */
    public Matrix(int[] shape){
        for(int i = 0; i < shape.length; i++){
            this.shape.add(shape[i]);
        }
        for(int i = 0; i < shape.length; i++){
            ArrayList<Double> temp = new ArrayList<Double>(shape[i]);
            for(int j = 0; j < shape[i]; j++){
                temp.add(0.0);
            }
            values.add(temp);
        }
    }

    // construtor that takes in n and m of 2d array as input
    /**
     * Contructor that takes in n and m of 2d array as input
     * @param n int: number of rows
     * @param m int: number of columns
     */
    public Matrix(int n, int m){
        this.shape.add(n);
        this.shape.add(m);
        for(int i = 0; i < n; i++){
            ArrayList<Double> row = new ArrayList<Double>(0);
            for(int j = 0; j < m; j++){
                row.add(0.0);
            }
            values.add(row);
        }
    }

    // constructor that takes in 2d double array as input
    /**
     * Contructor that takes in 2d double array as input
     * @param values double[][]: 2d double array
     */
    public Matrix(double[][] values){
        this.shape.add(values.length);
        this.shape.add(values[0].length);
        for(int i = 0; i < values.length; i++){
            ArrayList<Double> row = new ArrayList<Double>(0);
            for(int j = 0; j < values[0].length; j++){
                row.add(values[i][j]);
            }
            this.values.add(row);
        }
    }

    // initialize values to random numbers
    /**
     * initialize values to random numbers
     */
    public void randomize(){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                values.get(i).set(j, Math.random());
            }
        }
    }

    // static function to add two matrices
    /**
     * static function to add two matrices
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of addition
     * @throws IncompatibleTensorException
     */
    public static Matrix add(Matrix a, Matrix b) throws IncompatibleTensorException{
        if(a.shape.get(0) != b.shape.get(0) || a.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Error: cannot add matrix with shape " + a.shape + " and " + b.shape);
        }
        Matrix c = new Matrix(a.shape.get(0), a.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < a.shape.get(1); j++){
                c.values.get(i).set(j, a.values.get(i).get(j) + b.values.get(i).get(j));
            }
        }
        return c;
    }

    // static function to subtract two matrices
    /**
     * static function to subtract two matrices
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of subtraction
     */
    public static Matrix subtract(Matrix a, Matrix b) throws IncompatibleTensorException{
        if(a.shape.get(0) != b.shape.get(0) || a.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Error: cannot subtract matrix with shape " + a.shape + " and " + b.shape);
        }
        Matrix c = new Matrix(a.shape.get(0), a.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < a.shape.get(1); j++){
                c.values.get(i).set(j, a.values.get(i).get(j) - b.values.get(i).get(j));
            }
        }
        return c;
    }

    // static function to multiply two matrices
    /**
     * static function to multiply two matrices
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of multiplication
     */
    public static Matrix dot(Matrix a, Matrix b) throws IncompatibleTensorException{
        if(a.shape.get(1) != b.shape.get(0)){
            throw new IncompatibleTensorException("Error: cannot dot matrix with shape " + a.shape + " and " + b.shape);
        }
        Matrix c = new Matrix(a.shape.get(0), b.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < b.shape.get(1); j++){
                double sum = 0;
                for(int k = 0; k < a.shape.get(1); k++){
                    sum += a.values.get(i).get(k) * b.values.get(k).get(j);
                }
                c.values.get(i).set(j, sum);
            }
        }
        return c;
    }

    // static function for element wise multiplication
    /**
     * static function for element wise multiplication
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of element wise multiplication
     */
    public static Matrix multiply(Matrix a, Matrix b) throws IncompatibleTensorException{
        if(a.shape.get(0) != b.shape.get(0) || a.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Error: cannot element-wise multiply matrix with shape " + a.shape + " and " + b.shape);
        }
        Matrix c = new Matrix(a.shape.get(0), a.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < a.shape.get(1); j++){
                c.values.get(i).set(j, a.values.get(i).get(j) * b.values.get(i).get(j));
            }
        }
        return c;
    }

    // static function to multiply matrix with scalar
    /**
     * static function to multiply matrix with scalar
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of multiplication
     */
    public static Matrix multiply(Matrix a, double scalar){
        Matrix c = new Matrix(a.shape.get(0), a.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < a.shape.get(1); j++){
                c.values.get(i).set(j, a.values.get(i).get(j) * scalar);
            }
        }
        return c;
    }

    // static function to divide by a scalar
    /**
     * static function to divide by a scalar
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of division
     */
    public static Matrix divide(Matrix a, double scalar){
        Matrix c = new Matrix(a.shape.get(0), a.shape.get(1));
        for(int i = 0; i < a.shape.get(0); i++){
            for(int j = 0; j < a.shape.get(1); j++){
                c.values.get(i).set(j, a.values.get(i).get(j) / scalar);
            }
        }
        return c;
    }

    // static function to return a zero matrix
    /**
     * static function to return a zero matrix
     * @param a Matrix: first matrix
     * @param b Matrix: second matrix
     * @return Matrix: result of division
     */
    public static Matrix zeroes(ArrayList<Integer> shape){
        Matrix c = new Matrix(shape);
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                c.values.get(i).set(j, 0.0);
            }
        }
        return c;
    }

    //static function to return a matrix of zeroes
    /**
     * static function to return a matrix of zeroes
     * @param n int: number of rows
     * @param m int: number of columns
     * @return Matrix: result of division
     */
    public static Matrix zeroes(int n, int m){
        Matrix c = new Matrix(n, m);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                c.values.get(i).set(j, 0.0);
            }
        }
        return c;
    }

    // function that add a matrix to this matrix
    /**
     * function that add a matrix to this matrix
     * @param m Matrix: matrix to be added
     * @return Matrix: result of addition
     */
    public Matrix add(Matrix b) throws IncompatibleTensorException{
        if(this.shape.get(0) != b.shape.get(0) || this.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Error: cannot add matrix with shape " + this.shape + " and " + b.shape);
        }
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < this.shape.get(1); j++){
                this.values.get(i).set(j, this.values.get(i).get(j) + b.values.get(i).get(j));
            }
        }
        return this;
    }

    // function that subtracts a matrix from this matrix
    /**
     * function that subtracts a matrix from this matrix
     * @param m Matrix: matrix to be subtracted
     * @return Matrix: result of subtraction
     */
    public Matrix subtract(Matrix b) throws IncompatibleTensorException{
        if(this.shape.get(0) != b.shape.get(0) || this.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Error: cannot subtract matrix with shape " + this.shape + " and " + b.shape);
        }
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < this.shape.get(1); j++){
                this.values.get(i).set(j, this.values.get(i).get(j) - b.values.get(i).get(j));
            }
        }
        return this;
    }

    // function that multiplies this matrix by a scalar
    /**
     * function that multiplies this matrix by a scalar
     * @param scalar double: scalar to be multiplied by
     * @return Matrix: result of multiplication
     */
    public Matrix multiply(double scalar){
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < this.shape.get(1); j++){
                this.values.get(i).set(j, this.values.get(i).get(j) * scalar);
            }
        }
        return this;
    }

    // function that divides this matrix by a scalar
    /**
     * function that divides this matrix by a scalar
     * @param scalar double: scalar to be divided by
     * @return Matrix: result of division
     */
    public Matrix divide(double scalar){
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < this.shape.get(1); j++){
                this.values.get(i).set(j, this.values.get(i).get(j) / scalar);
            }
        }
        return this;
    }

    // function that element wise multiplies this matrix by another matrix
    /**
     * function that element wise multiplies this matrix by another matrix
     * @param m Matrix: matrix to be multiplied by
     * @return Matrix: result of element wise multiplication
     */
    public Matrix multiply(Matrix b) throws IncompatibleTensorException{
        if(this.shape.get(0) != b.shape.get(0) || this.shape.get(1) != b.shape.get(1)){
            throw new IncompatibleTensorException("Incompatible Tensors of shape: " + this.shape.get(0) + "x" + this.shape.get(1) + " and " + b.shape.get(0) + "x" + b.shape.get(1));
        }
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < this.shape.get(1); j++){
                this.values.get(i).set(j, this.values.get(i).get(j) * b.values.get(i).get(j));
            }
        }
        return this;
    }

    // function that dots this matrix with another matrix
    /**
     * function that dots this matrix with another matrix
     * @param m Matrix: matrix to be dotted with
     * @return Matrix: result of dot product
     */
    public Matrix dot(Matrix b) throws IncompatibleTensorException{
        if(this.shape.get(1) != b.shape.get(0)){
            // raise incompatilbility exception
            throw new IncompatibleTensorException("Incompatible Tensors of shape: " + this.shape.get(0) + "x" + this.shape.get(1) + " and " + b.shape.get(0) + "x" + b.shape.get(1));
        }
        Matrix c = new Matrix(this.shape.get(0), b.shape.get(1));
        for(int i = 0; i < this.shape.get(0); i++){
            for(int j = 0; j < b.shape.get(1); j++){
                double sum = 0;
                for(int k = 0; k < this.shape.get(1); k++){
                    sum += this.values.get(i).get(k) * b.values.get(k).get(j);
                }
                c.values.get(i).set(j, sum);
            }
        }
        return c;
    }

    // transpose of the matrix
    /**
     * transpose of the matrix
     * @return Matrix: transpose of the matrix
     */
    public Matrix transpose(){
        Matrix c = new Matrix(shape.get(1), shape.get(0));
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                c.values.get(j).set(i, values.get(i).get(j));
            }
        }
        return c;
    }

    // function to get rows 
    /**
     * function to get rows 
     * @param i int: row index
     * @return Vector: row
     */
    public ArrayList<ArrayList<Double>> getRows(int... rows){
        ArrayList<ArrayList<Double>> c = new ArrayList<ArrayList<Double>>(0);
        for(int i = 0; i < rows.length; i++){
            c.add(values.get(rows[i]));
        }
        return c;
    }
    
    // function to get columns
    /**
     * function to get columns
     * @param i int: column index
     * @return Vector: column
     */
    public ArrayList<ArrayList<Double>> getColumns(int... columns){
        ArrayList<ArrayList<Double>> c = new ArrayList<ArrayList<Double>>(0);
        for(int i = 0; i < columns.length; i++){
            ArrayList<Double> column = new ArrayList<Double>(0);
            for(int j = 0; j < shape.get(0); j++){
                column.add(values.get(j).get(columns[i]));
            }
            c.add(column);
        }
        return c;
    }
    
    // function to get a specific value
    /**
     * function to get a specific value
     * @param i int: row index
     * @param j int: column index
     * @return double: value at the specified index
     */
    public double get(int row, int column){
        return values.get(row).get(column);
    }

    // function to set a spicific value
    /**
     * function to set a spicific value
     * @param i int: row index
     * @param j int: column index
     * @param value double: value to be set
     */
    public void set(int row, int column, double value){
        values.get(row).set(column, value);
    }

    public void addRow(Matrix row){
        this.values.add(row.values.get(0));
        this.shape.set(0, this.shape.get(0) + 1);
    }

    public Matrix getRow(int row){
        Matrix c = new Matrix(1, this.shape.get(1));
        for(int i = 0; i < this.shape.get(1); i++){
            c.values.get(0).set(i, this.values.get(row).get(i));
        }
        return c;
    }

    public void setAll(double value){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                set(i, j, value);
            }
        }
    }

    // function to sum all values in a column but keeping the dims the same
    /**
     * function to sum all values in a column but keeping the dims the same
     * @param column int: column index
     * @return Matrix: matrix with summed values
     */
    public Matrix sumColumns(){
        Matrix c = new Matrix(shape.get(0), 1);
        for(int i = 0; i < shape.get(0); i++){
            double sum = 0;
            for(int j = 0; j < shape.get(1); j++){
                sum += values.get(i).get(j);
            }
            c.values.get(i).set(0, sum);
        }
        return c;
    }

    /**
     * function to raise all values to a power
     * @param power int: power to raise to
     */
    public void pow(int power){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                values.get(i).set(j, Math.pow(values.get(i).get(j), power));
            }
        }
    }

    // function to copy a matrix
    /**
     * function to copy a matrix
     * @return Matrix: copy of the matrix
     */
    public Matrix copy(){
        Matrix c = new Matrix(shape.get(0), shape.get(1));
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                c.set(i, j, get(i,j));
            }
        }
        return c;
    }

    // function to return a column as a new matrix
    /**
     * function to return a column as a new matrix
     * @param column int: column index
     * @return Matrix: column as a matrix
     */
    public Matrix getColumn(int column){
        Matrix c = new Matrix(shape.get(0), 1);
        for(int i = 0; i < shape.get(0); i++){
            c.set(i, 0, get(i, column));
        }
        return c;
    }

    // function to set a column to vector
    /**
     * function to set a column to vector
     * @param column int: column index
     * @param vector Vector: vector to set column to
     */
    public void setColumn(int column, Matrix vector){
        for(int i = 0; i < shape.get(0); i++){
            set(i, column, vector.get(i, 0));
        }
    }

    // function to add a new column to the matrix
    /**
     * function to add a new column to the matrix
     * @param column int: column index
     * @param vector Vector: vector to add to the matrix
     */
    public void addColumn(Matrix vector){
        for(int i = 0; i < shape.get(0); i++){
            values.get(i).add(vector.get(i, 0));
        }
        shape.set(1, shape.get(1) + 1);
    }

    // function to take natural log of all elements in matrix
    /**
     * function to take natural log of all elements in matrix
     */
    public void log(){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                set(i, j, Math.log(get(i, j)));
            }
        }
    }

    // function to take exp of all elements in matrix
    /**
     * function to take exp of all elements in matrix
     */
    public void exp(){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                set(i, j, Math.exp(get(i, j)));
            }
        }
    }

    // function to print matrix
    /**
     * function to print matrix
     */
    public void print(){
        for(int i = 0; i < shape.get(0); i++){
            for(int j = 0; j < shape.get(1); j++){
                String value = Double.toString(get(i, j));
                if(value.length() > 5){
                    value = value.substring(0, 5);
                }
                System.out.print(value + "\t");
            }
            System.out.println();
        }
    }

    // function to flatten a matrix and return as type Matrix and take axis as input to determine which axis to flatten
    /**
     * function to flatten a matrix and return as type Matrix and take axis as input to determine which axis to flatten
     * @param axis int: axis to flatten
     * @return Matrix: flattened matrix
     */
    public Matrix flatten(int axis){
        if(axis == 0){
            Matrix c = new Matrix(1, shape.get(0) * shape.get(1));
            for(int i = 0; i < shape.get(0); i++){
                for(int j = 0; j < shape.get(1); j++){
                    c.set(0, i * shape.get(1) + j, get(i, j));
                }
            }
            return c;
        }
        else{
            Matrix c = new Matrix(shape.get(0) * shape.get(1), 1);
            for(int i = 0; i < shape.get(0); i++){
                for(int j = 0; j < shape.get(1); j++){
                    c.set(i * shape.get(1) + j, 0, get(i, j));
                }
            }
            return c;
        }
    }

    /**
     * Function to expend a matrix such that edge value get repeated
     * @param axis int: axis to expend
     * @param times int: number of times to repeat
     */
    public void expand(int axis, int times){
        Matrix last_vec = axis == 0 ? getRow(shape.get(0) - 1) : getColumn(shape.get(1) - 1);
        for(int i = 0; i < times; i++){
            if(axis == 0){
                addRow(last_vec);
            }
            else{
                addColumn(last_vec);
            }
        }
    }

    /**
     * Function to expend a matrix such that edge value get repeated
     * @param axis int: axis to expend
     * @param times int: number of times to repeat
     * @return Matrix: expended matrix
     */
    public Matrix expandNew(int axis, int times){

        Matrix new_matrix = copy();

        Matrix last_vec = axis == 0 ? new_matrix.getRow(new_matrix.shape.get(0) - 1) : new_matrix.getColumn(new_matrix.shape.get(1) - 1);
        for(int i = 0; i < times; i++){
            if(axis == 0){
                new_matrix.addRow(last_vec);
            }
            else{
                new_matrix.addColumn(last_vec);
            }
        }

        return new_matrix;
    }
}
