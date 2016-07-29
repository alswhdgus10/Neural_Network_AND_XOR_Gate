import java.util.ArrayList;
import java.util.List;

/**
 * Created by wschoi on 2015-12-22.
 */
public class AND_Gate_by_NN {

    public static void main(String[] args) {

        double[] c_1 = {0, 0, 1};
        double[] c_2 = {0, 1, 1};
        double[] c_3 = {1, 0, 1};
        double[] c_4 = {1, 1, 1};

        double[] t   = {0,0,0,1};

        List<double[]> list = new ArrayList<double[]>(4);
        list.add(c_1); list.add(c_2); list.add(c_3); list.add(c_4);

        double[] w= {+0.54, -0.61 , -0.15};

        print_all(w);
        
        print_err(w, list, t);


        for (int i = 0; i < 10; i++) {
            w = training(w, list, t);
            print_err(w, list, t);
//            print_out(w, list);

        }
        print_err(w, list, t);
        print_out(w, list);
    }

    private static void print_out(double[] w, List<double[]> list) {

        double[] out = new double[list.size()];

        for (int i = 0; i < list.size(); i++)
            out[i] = sigm(product(list.get(i), w));

        print_all(out);
    }

    private static double print_err(double[] w, List<double[]> list, double[] t) {

        double res=0;

        for (int i = 0; i < list.size(); i++) {
            double[] cases = list.get(i);
            double target = t[i];
            double z = product(cases, w);
            double sigz = sigm(z);
            double target_minus_z= target-sigz;

            res += target_minus_z*target_minus_z;

        }

        System.out.println("error: " + res);
        return res;

    }

    private static double[] training(double[] w, List<double[]> list, double[] t) {

//        double[] new_w = new double[w.length];
        double[] new_w = w.clone();

        for (int i = 0; i < list.size(); i++) {
            double[] cases = list.get(i);

            double target = t[i];
            double z = product(cases, new_w);
            double sigz = sigm(z);
            double target_minus_z= target-sigz;

//            System.out.println("targer: " + target + ", estimate: " + sigz);

            for (int j = 0; j < new_w.length; j++)
                    new_w[j] -= - 0.4 *target_minus_z *  sigz * (1-sigz) * cases[j];


            z= product(cases, new_w);
            sigz = sigm(z);
//            System.out.println("targer: " + target + ", estimate: " + sigz);


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
