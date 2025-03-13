package com.example.facialattributetest;

import static java.security.AccessController.getContext;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
//import java.nio.HeapFloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class MediaPipeFaceDetectionModel {
    private Context context;
    private MappedByteBuffer tfliteFile;
    private Map<Integer, Object> rawOutputs;
    private FloatBuffer[] boxCoords = new FloatBuffer[896];

    // Get Context from MainActivity
    public MediaPipeFaceDetectionModel(Context context) throws IOException {
        this.context = context;
        this.loadModelFile();
    }

    // Open image from assets and return its bitmap
    public Bitmap openSampleImageAsBitMap() throws IOException {

        Bitmap bMap = BitmapFactory.decodeStream(context.getAssets().open("sample_image.jpg"));

        // Resize to 256 by 256
        bMap = resizeAndPadMaintainAspectRatio(bMap, 256, 256, 0);

        return bMap;
    }

    // Load the tflite file for mediapipe face detection model
    public void loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor=context.getAssets().openFd("mediapipe_face-mediapipefacedetector.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        this.tfliteFile =  fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }

    // Turn bitmap to bytebuffer
    public static ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int size = width * height * 4; // 4 bytes per pixel for ARGB_8888

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(size);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int pixel : pixels) {
            byteBuffer.put((byte) ((pixel >> 16) & 0xFF)); // Red
            byteBuffer.put((byte) ((pixel >> 8) & 0xFF));  // Green
            byteBuffer.put((byte) (pixel & 0xFF));       // Blue
            byteBuffer.put((byte) ((pixel >> 24) & 0xFF)); // Alpha
        }

        byteBuffer.rewind();
        return byteBuffer;
    }

    // Resize bitmap
    public static Bitmap resizeAndPadMaintainAspectRatio(
            Bitmap image,
            int outputBitmapWidth,
            int outputBitmapHeight,
            int paddingValue) {
        int width = image.getWidth();
        int height = image.getHeight();
        float ratioBitmap = (float) width / (float) height;
        float ratioMax = (float) outputBitmapWidth / (float) outputBitmapHeight;

        int finalWidth = outputBitmapWidth;
        int finalHeight = outputBitmapHeight;
        if (ratioMax > ratioBitmap) {
            finalWidth = (int) ((float)outputBitmapHeight * ratioBitmap);
        } else {
            finalHeight = (int) ((float)outputBitmapWidth / ratioBitmap);
        }

        Bitmap outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ARGB_8888);
        Canvas can = new Canvas(outputImage);
        can.drawARGB(0xFF, paddingValue, paddingValue, paddingValue);
        int left = (outputBitmapWidth - finalWidth) / 2;
        int top = (outputBitmapHeight - finalHeight) / 2;
        can.drawBitmap(image, null, new RectF(left, top, finalWidth + left, finalHeight + top), null);
        return outputImage;
    }

    // Preprocess image, resize turn into bitmap then bytebuffer
    private ByteBuffer preprocessSampleImage() throws IOException {
        Bitmap resizedSampleImage = this.openSampleImageAsBitMap();
//        ByteBuffer input = ByteBuffer.allocateDirect(256 * 256 * 3 * 4);
//        input = bitmapToByteBuffer(resizedSampleImage);
//        input.order(ByteOrder.nativeOrder());
//        return input;

        // For FLOAT 32 use image processor
        ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build(); // 0.0f, 255.0f
        TensorImage tImg = new TensorImage(DataType.FLOAT32);
        //TensorImage tImg = new TensorImage();
        tImg.load(resizedSampleImage);
        TensorImage processedImage = imageProcessor.process(tImg);
        ByteBuffer inputImage = processedImage.getBuffer();
        ByteBuffer inputImage2 = tImg.getBuffer();
        return inputImage;


//        TensorImage inputImage = new TensorImage(DataType.FLOAT32);
//        inputImage.load(resizedSampleImage);
//        return inputImage.getBuffer();
    }

    // Set up input buffers for interpreter should be Object[] containing bytebuffer
    private Object[] setUpInputBuffer(ByteBuffer image)
    {
        Object[] inputs = new Object[1];
        inputs[0] = image;
        return inputs;
    }
    // Set up output buffers
    private Map<Integer, Object> setUpOutputBuffers()
    {
        Map<Integer, Object> outputs = new HashMap<>();
        FloatBuffer[] boxCoords = new FloatBuffer[896];
        for (int i = 0; i < 896; i++)
        {
            boxCoords[i] = FloatBuffer.allocate(16);
        }
        FloatBuffer[] boxScores = new FloatBuffer[896];
        for (int i = 0; i < 896; i++)
        {
            boxScores[i] = FloatBuffer.allocate(1);
        }
        outputs.put(0, boxCoords);  // box_coords
        outputs.put(1, boxScores);  // box_scores

        FloatBuffer boxCoords2 = FloatBuffer.allocate(896 * 16);
        FloatBuffer boxScores2 = FloatBuffer.allocate(896);
        Map<Integer, Object> outputs2 = new HashMap<>();
        outputs2.put(0, boxCoords2);
        outputs2.put(1, boxScores2);
        return outputs2;
    }

    // Reorganize outputs so each box coord has 16 elements
    private void organizeOutputs(Map<Integer, Object> outputs)
    {
        FloatBuffer rawBoxCoords = (FloatBuffer) outputs.get(0);
        rawBoxCoords.flip();
        int index = 0;
        for (int i = 0; i < 56; i++)
        {
            FloatBuffer b = FloatBuffer.allocate(16);
            for (int j = 0; j < 16; j++)
            {
                b.put(rawBoxCoords.get());
            }
            this.boxCoords[index] = b;
            index++;
        }
    }

    // Create the interpreter and run inference
    public void runInterpreter() throws IOException {
        Log.d("Interpreter", "Setting up interpreter");
        Interpreter i = new Interpreter(this.tfliteFile);
        i.allocateTensors();
        Log.d("Interpreter", "Input tensor count = " + Integer.toString(i.getInputTensorCount()));   // 1
        Log.d("Interpreter", "Output tensor count = " + Integer.toString(i.getOutputTensorCount())); // 2
        Tensor inputTensor = i.getInputTensor(0);
        Tensor outputTensor0 = i.getOutputTensor(0);
        Tensor outputTensor1 = i.getOutputTensor(1);

        // tensor information
        int [] inputShape = inputTensor.shape(); // 1, 256, 256, 3
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        DataType type = inputTensor.dataType(); //FLOAT32
        //output tensor information
        int [] outputShape0 = outputTensor0.shape(); // [1, 896, 16] id_feature
        DataType type0 = outputTensor0.dataType(); // FLOAT32
        int [] outputShape1 = outputTensor1.shape(); // [1, 896, 1] liveness
        DataType type1 = outputTensor1.dataType(); // FLOAT32

        ByteBuffer sampleInput = this.preprocessSampleImage();
        Object[] inputs = this.setUpInputBuffer(sampleInput);
        Map<Integer, Object> outputs = this.setUpOutputBuffers();
        i.runForMultipleInputsOutputs(inputs, outputs);
        this.rawOutputs = outputs;
        this.organizeOutputs(outputs);
        i.close();
    }


}
