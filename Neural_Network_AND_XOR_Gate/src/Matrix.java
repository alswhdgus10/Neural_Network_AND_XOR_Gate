public class Matrix {

    double[][] A;

    /*
    A[i][j]
        i = row index
        j = column index
    A.length = number of rows
    A[0].length = number of columns
     */


    public Matrix(double[][] a) {
        A = a;
    }


    public double[][] getMatrix() {
        return A;
    }

    public Matrix multiplication(Matrix matrix) throws Exception {

        double[][] B = matrix.getMatrix();

        if(A[0].length != B.length)
            throw new Exception();

        double[][] C = new double[A.length][B[0].length];

        int A_rows = A.length;
        int A_cols = A[0].length;
        int B_cols = B[0].length;


        for (int i = 0; i < A_rows; i++)
            for (int j = 0; j < B_cols; j++)
                for (int k = 0; k < A_cols; k++)
                    C[i][j] += A[i][k] * B[k][j];


        // C = A x B
        return new Matrix(C);
    }

    public Matrix transpose() {
        int rows = A.length;
        int cols = A[0].length;

        double[][] A_T = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A_T[j][i] = A[i][j];
            }
        }

        return new Matrix(A_T);
    }

    public Matrix to_softmax() {

        int rows = A.length;
        int cols = A[0].length;

        double[][] c = new double[rows][cols];

        for (int i = 0; i < rows; i++) {

            double sum = 0;

            for (int j = 0; j < cols; j++) {
                c[i][j] = Math.exp(A[i][j]);
                sum += c[i][j];
            }

            for (int j = 0; j < cols; j++)
                c[i][j]/=sum;


        }


        return new Matrix(c);
    }

    public Matrix to_log() {

        int rows = A.length;
        int cols = A[0].length;

        double[][] c = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = Math.log(A[i][j]);
            }
        }

        return new Matrix(c);
    }

    public Matrix unit_prodoct(Matrix B) throws Exception {

        if(A.length != B.A.length)
            throw new Exception();
        if(A[0].length != B.A[0].length)
            throw new Exception();

        int rows = A.length;
        int cols = A[0].length;

        double[][] c = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = A[i][j] * B.A[i][j];
            }
        }

        return new Matrix(c);
    }

    public Matrix aggregate_cols() {

        int rows = A.length;
        int cols = A[0].length;

        double[][] res = new double[rows][1];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res[i][0] += A[i][j];
            }

        }
        return new Matrix(res);
    }

    public Matrix unit_prodoct(double scala) {
        int rows = A.length;
        int cols = A[0].length;

        double[][] c = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = A[i][j] * scala;
            }
        }
        return new Matrix(c);
    }

    public double get(int i, int j) {
        return A[i][j];
    }

    public Matrix unit_plus(Matrix B) throws Exception {

        if(A.length != B.A.length)
            throw new Exception();
        if(A[0].length != B.A[0].length)
            throw new Exception();

        int rows = A.length;
        int cols = A[0].length;

        double[][] c = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c[i][j] = A[i][j] + B.A[i][j];
            }
        }

        return new Matrix(c);
    }
}