package digitsRecognizing;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import scala.Tuple2;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Network {
    // feedforward fully-connected neural network
    // cost function - cross entropy
    // regularisation for weights is also used

    private List<Layer> layers;

    class Layer{
        private ArrayRealVector bias;
        private Array2DRowRealMatrix weights;
        private ArrayRealVector activations;
        private ArrayRealVector error;
        private ArrayRealVector biasDerivative;
        private Array2DRowRealMatrix weightsDerivative;

    }

    public Network(int... layersSizes){
        layers = new ArrayList<>();

        NormalDistribution distribution = new NormalDistribution(0, 1);

        for (int i=0; i<layersSizes.length; i++){
            Layer layer = new Layer();
            layer.activations = new ArrayRealVector(new double[layersSizes[i]]);

            if (i != 0){
                layer.bias = new ArrayRealVector(distribution.sample(layersSizes[i]));
                layer.biasDerivative = new ArrayRealVector(new double[layersSizes[i]]);
                layer.error = new ArrayRealVector(new double[layersSizes[i]]);

                // in order to avoid sampling of too huge weights
                // they are taken from normal distribution with mean 0 and variance 1/(num. of neurons in previous layer)
                double[][] tmp = Arrays.stream(new double[layersSizes[i]][layersSizes[i-1]])
                        .map(s -> distribution.sample(s.length)).toArray(double[][]::new);
                layer.weights = new Array2DRowRealMatrix(tmp);
                layer.weights = (Array2DRowRealMatrix) layer.weights.scalarMultiply((double) 1/Math.pow(layer.weights.getColumnDimension(), 0.5));
                layer.weightsDerivative = new Array2DRowRealMatrix(new double[layersSizes[i]][layersSizes[i-1]]);
            }

            layers.add(layer);

        }
    }

    public ArrayRealVector feedforward(ArrayRealVector input){
        layers.get(0).activations = input;
        for(int i=1; i<layers.size(); i++){
            ArrayRealVector z = (ArrayRealVector) layers.get(i).weights.operate(layers.get(i-1).activations).add(layers.get(i).bias);
            layers.get(i).activations = z.map(new Sigmoid());
        }

        return layers.get(layers.size()-1).activations;
    }


    private void backprop(ArrayRealVector input, ArrayRealVector output){
        ArrayRealVector res = this.feedforward(input);
        layers.get(layers.size() - 1).error = res.subtract(output);

        for(int i=layers.size()-1; i>0; i--){
            layers.get(i).biasDerivative.add(layers.get(i).error);

            for(int j=0; j<layers.get(i).error.getDimension(); j++){
                ArrayRealVector vector = (ArrayRealVector) layers.get(i-1).activations.mapMultiply(layers.get(i).error.getEntry(j));
                vector = vector.add(layers.get(i).weightsDerivative.getRowVector(j));
                layers.get(i).weightsDerivative.setRow(j, vector.toArray());

            }

            if (i == 1){
                break;
            }

            layers.get(i-1).error = (ArrayRealVector) layers.get(i).weights.transpose().operate(layers.get(i).error);
            ArrayRealVector sigmoidDerivative = layers.get(i-1).activations.ebeMultiply(layers.get(i-1).activations.mapSubtract(1));
            sigmoidDerivative = (ArrayRealVector) sigmoidDerivative.mapMultiply(-1);
            layers.get(i-1).error = layers.get(i-1).error.ebeMultiply(sigmoidDerivative);
        }

    }

    public void gradientDescent(int miniBatchSize, int epochs, double eta, double lambda) throws Exception {
        // implements stochastic gradient descent algorithm
        // eta - step size
        // lambda - regularisation parameter

        int currentMiniBatch = 0;
        int miniBatchesPerEpoch = 60000/(miniBatchSize * epochs);
        MnistReader mnistReader = new MnistReader("train-images.idx3-ubyte", "train-labels.idx1-ubyte", MnistReaderType.LEARNING);

        for (int epoch=0; epoch<epochs; epoch++){
            for (int batch=0; batch<miniBatchesPerEpoch; batch++){

                ArrayList<Tuple2<Integer, ArrayRealVector>> miniBatch = mnistReader.getMiniBatch(currentMiniBatch, miniBatchSize);
                currentMiniBatch += miniBatchSize;

                for (Tuple2<Integer, ArrayRealVector> entry : miniBatch){
                    double[] answer = new double[10];
                    answer[entry._1] = 1;
                    backprop(entry._2, new ArrayRealVector(answer));
                }

                for (int i=1; i<layers.size(); i++){
                    layers.get(i).biasDerivative.mapDivideToSelf(miniBatchSize);
                    layers.get(i).weightsDerivative = (Array2DRowRealMatrix) layers.get(i).weightsDerivative.scalarMultiply((double) 1/miniBatchSize);

                    layers.get(i).bias = layers.get(i).bias.subtract(layers.get(i).biasDerivative.mapMultiply(eta));
                    layers.get(i).weights = (Array2DRowRealMatrix) layers.get(i).weights.add(layers.get(i).weights.scalarMultiply(-eta * lambda/60000));
                    layers.get(i).weights = (Array2DRowRealMatrix) layers.get(i).weights.add(layers.get(i).weightsDerivative.scalarMultiply(-eta));

                    layers.get(i).biasDerivative.mapMultiplyToSelf(0.0);
                    layers.get(i).weightsDerivative = (Array2DRowRealMatrix) layers.get(i).weightsDerivative.scalarMultiply(0.0);

                }
            }
        }

    }

    public double test() throws Exception {
        int rate = 0;
        MnistReader mnistReader = new MnistReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", MnistReaderType.TESTING);

        for(int i=0; i<10000; i++){
            Tuple2<Integer, ArrayRealVector> sample = mnistReader.getTestSample(i);
             double[] output = feedforward(sample._2).toArray();

             int maxElementNumber = IntStream.range(0, output.length)
                     .reduce((k, j) -> output[k] > output[j] ? k : j)
                     .getAsInt();

             if (sample._1 == maxElementNumber){
                 rate += 1;
             }

        }

        return (double) 100 * rate/10000;
    }

    public void saveNetwork(String saveFilePath){
        // saves weights and biases to file - saveFilePath

        List<double[]> biasList = layers.stream()
                .skip(1)
                .map(s -> s.bias)
                .map(ArrayRealVector::toArray)
                .collect(Collectors.toList());

        List<double[][]> weightsList = layers.stream()
                .skip(1)
                .map(s -> s.weights)
                .map(Array2DRowRealMatrix::getData)
                .collect(Collectors.toList());

        HashMap<String, Object> networkMap = new HashMap<>();
        networkMap.put("biases", biasList);
        networkMap.put("weights", weightsList);

        try {
            FileOutputStream fos = new FileOutputStream(saveFilePath);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(networkMap);
        } catch (IOException i) {
            i.printStackTrace();
        }

    }

    static public Network loadNetwork(String loadFilePath){
        // loads weights and biases from file - loadFilePath

        try {
            FileInputStream fis = new FileInputStream(loadFilePath);
            ObjectInputStream iis = new ObjectInputStream(fis);
            HashMap<String, Object> map = (HashMap<String, Object>) iis.readObject();

            List<double[]> biasList = (List<double[]>) map.get("biases");
            List<double[][]> weightsList = (List<double[][]>) map.get("weights");

            int[] layersSizes = new int[biasList.size()+1];
            layersSizes[0] = 784;
            for (int i=1; i<layersSizes.length; i++){
                layersSizes[i] = biasList.get(i-1).length;
            }

            Network network = new Network(layersSizes);

            for (int i=1; i<network.layers.size(); i++){
                network.layers.get(i).bias = new ArrayRealVector(biasList.get(i-1));
                network.layers.get(i).weights = new Array2DRowRealMatrix(weightsList.get(i-1));
            }

            return network;

        } catch (Exception ex) {
            System.err.println("failed to read " + loadFilePath + ", "+ ex);
        }

        return null;

    }

}
