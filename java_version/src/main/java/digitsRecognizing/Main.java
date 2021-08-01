package digitsRecognizing;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.distribution.NormalDistribution;
import javafx.animation.Animation;
import javafx.application.Application;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import scala.Tuple2;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

public class Main {

    public static void main(String[] args) throws Exception {



//        digitsRecognizing.Network network = new digitsRecognizing.Network(784, 100, 10);
//        double bestRes = 0.0;
//        for (int i=0; i<10; i++){
//            digitsRecognizing.Network net = new digitsRecognizing.Network(784, 100, 10);
//            net.gradientDescent(10, 30, 0.5, 5.5);
//            double res = net.test();
//            if (res > bestRes){
//                bestRes = res;
//                net.saveNetwork("test.dat");
//            }
//        }

//        digitsRecognizing.Network network = new digitsRecognizing.Network(784, 100, 10);
//        network.gradientDescent(10, 30, 0.5, 5.5);
//        System.out.println("Test result: " + network.test());

        Network network1 = Network.loadNetwork("network.dat");
        System.out.println("Test result is: " + network1.test());

        MnistReader mnistReader = new MnistReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", MnistReaderType.TESTING);

        for (int i=0; i<10; i++){
            System.out.println("----------------------------------------");
            Tuple2<Integer, ArrayRealVector> sample = mnistReader.getTestSample(i);
            System.out.println("Label: " + sample._1);
            System.out.println("digitsRecognizing.Network answer: " + network1.feedforward(sample._2));
        }

    }

}
