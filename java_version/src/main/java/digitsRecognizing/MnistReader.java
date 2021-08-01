package digitsRecognizing;

import org.apache.commons.math3.linear.ArrayRealVector;
import scala.Tuple2;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;


public class MnistReader {

    // class for reading MNIST dataset

    private byte[] images;
    private byte[] labels;
    private MnistReaderType type;

    public MnistReader(String imagesPath, String labelsPath, MnistReaderType type) throws IOException {
        this.type = type;
        File imagesFile = new File(imagesPath);
        File labelsFile = new File(labelsPath);

        long size = imagesFile.length();
        images = new byte[(int)size];
        FileInputStream in = new FileInputStream(imagesFile);
        in.read(images);
        in.close();

        long size1 = labelsFile.length();
        labels = new byte[(int)size1];
        FileInputStream in1 = new FileInputStream(labelsFile);
        in1.read(labels);
        in1.close();
    }

    public ArrayList<Tuple2<Integer, ArrayRealVector>> getMiniBatch(int firstElementNumber, int miniBatchSize) throws Exception {
        if (type != MnistReaderType.LEARNING){
            throw new Exception("getMiniBatch can be used only for LEARNING type readers");
        }

        ArrayList<Tuple2<Integer, ArrayRealVector>> miniBatch = new ArrayList<>();

        for (int k=firstElementNumber; k < miniBatchSize + firstElementNumber; k++){
            double[] pixels = new double[784];

            for(int j = 0; j < 28; j++){
                for(int i = 0; i < 28; i++){
                    int x = images[16 + j*28 + i + 784*k];
                    if(x < 0) // java reads bites with signs
                    {
                        x = (x * (-1)) + 128;
                    }

                    pixels[28*j + i] = (double) x/256; // scale to [0, 1] interval

                }
            }
            miniBatch.add(new Tuple2<>((int) labels[8+k], new ArrayRealVector(pixels)));
        }

        return miniBatch;
    }

    public Tuple2<Integer, ArrayRealVector> getTestSample(int testSampleNumber) throws Exception {
        if (type != MnistReaderType.TESTING){
            throw new Exception("getTestSample can be used only for TESTING type readers");
        }

        double[] pixels = new double[784];
        for(int j = 0; j < 28; j++) {
            for (int i = 0; i < 28; i++) {
                int x = images[16 + j*28 + i + 784*testSampleNumber];
                if (x < 0) {
                    x = (x * (-1)) + 128;
                }

                pixels[28*j + i] = (double) x/256;

            }
        }

        return new Tuple2<>((int) labels[8+testSampleNumber], new ArrayRealVector(pixels));

    }

    public MnistReaderType getMnistReaderType(){
        return type;
    }



}
