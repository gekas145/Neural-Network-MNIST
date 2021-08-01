package digitsRecognizing;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelReader;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import org.apache.commons.math3.linear.ArrayRealVector;
import java.util.ArrayList;
import java.util.List;


public class DrawApp extends Application {

    // application for creating own examples for network recognition tests

    // remember that network was trained on centered images

    @Override
    public void start(Stage primaryStage) {

        Canvas canvas = new Canvas(280, 280);
        final GraphicsContext graphicsContext = canvas.getGraphicsContext2D();
        Group root = new Group();

        Text text = new Text();
        text.setText("Network prediction:");
        text.setX(320);
        text.setY(20);
        root.getChildren().add(text);
        ArrayList<Rectangle> rectangles = new ArrayList<>();

        for (int i=0; i<10; i++){
            Text label = new Text();
            label.setText(i + ":");
            label.setX(300);
            label.setY(12 + 25*(i+1));
            root.getChildren().add(label);

            Rectangle rectangle = new Rectangle();
            rectangle.setX(310);
            rectangle.setY(6 + 25*(i+1));
            rectangle.setHeight(7);
            rectangle.setWidth(0);
            rectangle.setFill(Color.RED);
            rectangles.add(rectangle);
            root.getChildren().add(rectangle);
        }
        Network network1 = Network.loadNetwork("network.dat"); // pretrained network loading

        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED,
                new EventHandler<MouseEvent>(){
                    @Override
                    public void handle(MouseEvent event) {
                        if (event.getButton() == MouseButton.PRIMARY) {
                            // right mouse button - draw mode
                            graphicsContext.setStroke(Color.BLACK);
                            graphicsContext.setLineWidth(15);
                            graphicsContext.beginPath();
                            graphicsContext.moveTo(event.getX(), event.getY());
                            graphicsContext.stroke();
                        }
                        if (event.getButton() == MouseButton.SECONDARY){
                            // left mouse button - erase mode
                                graphicsContext.setStroke(Color.WHITE);
                                graphicsContext.setLineWidth(20);
                                graphicsContext.beginPath();
                                graphicsContext.moveTo(event.getX(), event.getY());
                                graphicsContext.stroke();
                        }
                        if (event.getButton() == MouseButton.MIDDLE){
                            // middle mouse button - delete all contents from screen
                            graphicsContext.setFill(Color.WHITE);
                            graphicsContext.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
                            graphicsContext.beginPath();
                            graphicsContext.moveTo(event.getX(), event.getY());
                            graphicsContext.stroke();
                        }
                    }
                });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED,
                new EventHandler<MouseEvent>(){
                    @Override
                    public void handle(MouseEvent event) {
                        graphicsContext.lineTo(event.getX(), event.getY());
                        graphicsContext.stroke();
                        Image snapshot = canvas.snapshot(null, null);

                        double[] number = parseDrawing(snapshot);
                        double[] output = network1.feedforward(new ArrayRealVector(number)).toArray();

                        updateRectangles(rectangles, output);
                    }
                });

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED,
                new EventHandler<MouseEvent>(){

                    @Override
                    public void handle(MouseEvent event) {

                    }
                });

        canvas.setFocusTraversable(true);

        root.getChildren().add(canvas);

        Scene scene = new Scene(root, 450, 280);
        scene.setFill(Color.GRAY);

        graphicsContext.setFill(Color.WHITE);
        graphicsContext.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        primaryStage.setTitle("Draw app");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }


    private double[] parseDrawing(Image snapshot){
        ImageView imageView  = new ImageView(snapshot);
        imageView.setFitHeight(28);
        imageView.setFitWidth(28);
        imageView.setSmooth(true);
        snapshot = imageView.snapshot(null, null);
        PixelReader pr = snapshot.getPixelReader();

        double[] parsedDrawing = new double[784];
        for (int i=0; i<28; i++){
            for (int j=0; j<28; j++){
                double grayscale = 0.11 * pr.getColor(i, j).getBlue();
                grayscale += 0.3 * pr.getColor(i, j).getRed();
                grayscale += 0.59 * pr.getColor(i, j).getGreen();
                parsedDrawing[28*j + i] = 1 - grayscale;
            }
        }

        return parsedDrawing;
    }

    private void updateRectangles(List<Rectangle> rectangles, double[] networkOutput){
        for (int i=0; i<10; i++){
            rectangles.get(i).setWidth(networkOutput[i] * 60);
            rectangles.get(i).setFill(Color.rgb(100, (int) (networkOutput[i] * 200), 0));
        }
    }

}