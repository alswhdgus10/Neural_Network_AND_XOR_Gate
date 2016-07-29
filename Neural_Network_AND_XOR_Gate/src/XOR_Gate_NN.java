import java.util.ArrayList;
import java.util.List;

/**
 * Created by wschoi on 2015-12-22.
 */
public class XOR_Gate_NN {

    private static double learning_rate = 0.2;

    public static void main(String[] args) {

        double[] c_1 = {0, 0, 1};
        double[] c_2 = {0, 1, 1};
        double[] c_3 = {1, 0, 1};
        double[] c_4 = {1, 1, 1};

        double[] t   = {0,1,1,0};

        List<double[]> list = new ArrayList<double[]>(4);
        list.add(c_1); list.add(c_2); list.add(c_3); list.add(c_4);

        double[] w0= {-0.54, -0.61 , -0.15};
        double[] w1= {-0.24, -0.24, 0.14};
        double[] w2= {+0.72, -0.24, 0.15};
        double[][] W = {w0, w1, w2};


        for (int i = 0; i < 6000; i++) {
            W = training(W, list, t);
            print_err(W, list, t);
        }
        print_err(W, list, t);
        print_out(W, list);
    }

    private static void print_out(double[][] w, List<double[]> list) {

        double[] out = new double[list.size()];

        for (int i = 0; i < list.size(); i++)
        {
            double[] cases = list.get(i);

            double z1 = product(cases, w[0]);
            double sigz1 = sigm(z1);

            double z2 = product(cases, w[1]);
            double sigz2 = sigm(z2);

            double[] z3_input = {sigz1, sigz2, 1};
            double z3 = product(z3_input, w[2]);
            double sigz3 = sigm(z3);

            out[i] = sigz3;
        }

        print_all(out);
    }

    private static double print_err(double[][] w, double[] cases, double target) {

        double z1 = product(cases, w[0]);
        double sigz1 = sigm(z1);

        double z2 = product(cases, w[1]);
        double sigz2 = sigm(z2);

        double[] z3_input = {sigz1, sigz2, 1};
        double z3 = product(z3_input, w[2]);
        double sigz3 = sigm(z3);

        double target_minus_sigz3= target-sigz3;

        return target_minus_sigz3;

    }
    private static double print_err(double[][] w, List<double[]> list, double[] t) {

        double res=0;

        for (int i = 0; i < list.size(); i++) {
            double[] cases = list.get(i);
            double target = t[i];

            double err = print_err(w, cases, target) ;


            res += err*err;

        }

        System.out.println("error: " + res);
        return res;

    }

    private static double[][] training(double[][] w, List<double[]> list, double[] t) {

        double[][] new_w = w.clone();

        for (int i = 0; i < list.size(); i++) {
            double[] z12_input = list.get(i);
            double target = t[i];

            double z1 = product(z12_input, new_w[0]);
            double sigz1 = sigm(z1);

            double z2 = product(z12_input, new_w[1]);
            double sigz2 = sigm(z2);

            double[] z3_input = {sigz1, sigz2, 1};
            double z3 = product(z3_input, new_w[2]);
            double sigz3 = sigm(z3);

            double target_minus_sigz3= target-sigz3;


            for (int j = 0; j < new_w[2].length; j++)
                    new_w[2][j] -= -1 * learning_rate *  z3_input[j]                                     * target_minus_sigz3 *  sigz3 * (1-sigz3);

            for (int j = 0; j < new_w[1].length; j++)
                    new_w[1][j] -= -1 * learning_rate * z12_input[j] * (sigz2 * (1-sigz2) * new_w[2][1]) * target_minus_sigz3 *  sigz3 * (1-sigz3) ;

            for (int j = 0; j < new_w[0].length; j++)
                    new_w[0][j] -= -1 * learning_rate * z12_input[j] * (sigz1 * (1-sigz1) * new_w[2][0]) * target_minus_sigz3 *  sigz3 * (1-sigz3) ;


        }


        return new_w;
    }

    private static double sigm(double z) {
        return (1/(1+ Math.exp(-1 * z)));
    }

    private static void print_all(double[] new_w) {
        for (int i = 0; i < new_w.length; i++) {
            System.out.print(new_w[i]+", ");
        }
        System.out.println();
    }

    private static double product(double[] cases, double[] w) {

        double res = 0;
        for (int i = 0; i < cases.length; i++) {
            res += cases[i] * w[i];
        }

        return res;
    }
}
