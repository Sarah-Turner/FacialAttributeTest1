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

public class FaceAttributeModel {
    private Context context;
    private MappedByteBuffer tfliteFile;
    private Map<Integer, Object> rawOutputs;
    private FloatBuffer originalLivenessEmbedding;
    private FloatBuffer currentLivenessEmbedding;
    public double probEyeClosenessL;
    public double probEyeClosenessR;
    public double probSunglasses;
    public boolean eyeClosenessL;
    public boolean eyeClosenessR;
    public boolean sunglasses;
    public double livenessLoss;
    public boolean liveness;




    // Get Context from MainActivity
    public FaceAttributeModel(Context context) throws IOException {
        this.context = context;
        this.loadModelFile();
    }

    // Open image from assets and return its bitmap
    public Bitmap openSampleImageAsBitMap() throws IOException {
        Bitmap bMap = BitmapFactory.decodeStream(context.getAssets().open("sample_image.jpg"));

        // Resize to 128 by 128
        bMap = resizeAndPadMaintainAspectRatio(bMap, 128, 128, 0);

        return bMap;
    }

    // Load the tflite file for facial attribute model
    public void loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor=context.getAssets().openFd("face_attrib_net.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        this.tfliteFile =  fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }

//    // Convert bitmap to grayscale
//    public Bitmap toGrayscale(Bitmap bmpOriginal)
//    {
//        int width, height;
//        height = bmpOriginal.getHeight();
//        width = bmpOriginal.getWidth();
//
//        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//        Canvas c = new Canvas(bmpGrayscale);
//        Paint paint = new Paint();
//        ColorMatrix cm = new ColorMatrix();
//        cm.setSaturation(0);
//        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
//        paint.setColorFilter(f);
//        c.drawBitmap(bmpOriginal, 0, 0, paint);
//        return bmpGrayscale;
//    }
//
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
    // Bitmap to greyscale bytebuffer
//    public ByteBuffer getGrayScaleByteBuffer(Bitmap bitmap){
//        int width = bitmap.getWidth();
//        int height = bitmap.getHeight();
//        ByteBuffer mImgData = ByteBuffer
//                .allocateDirect(4 * width * height);
//        mImgData.order(ByteOrder.nativeOrder());
//        int[] pixels = new int[width*height];
//        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
//        for (int pixel : pixels) {
//            mImgData.putFloat((float) Color.red(pixel));
//        }
//        return mImgData;
//    }

    // Create the interpreter and run inference
    public void runInterpreter() throws IOException {
        Log.d("Interpreter", "Setting up interpreter");
        Interpreter i = new Interpreter(this.tfliteFile);
        i.allocateTensors();
//        Log.d("Interpreter", "Input tensor count = " + Integer.toString(i.getInputTensorCount()));
//        Log.d("Interpreter", "Output tensor count = " + Integer.toString(i.getOutputTensorCount()));
//        Log.d("Interpreter", "Open and resize image as bitmap");
        //Bitmap resizedSampleImage = this.openSampleImageAsBitMap();
        //Log.d("Interpreter", "Turn bitmap into greyscale bytebuffer");


//        Tensor inputTensor = i.getInputTensor(0);
//        Tensor outputTensor0 = i.getOutputTensor(0);
//        Tensor outputTensor1 = i.getOutputTensor(1);
//        Tensor outputTensor2 = i.getOutputTensor(2);
//        Tensor outputTensor3 = i.getOutputTensor(3);
//        Tensor outputTensor4 = i.getOutputTensor(4);
//        Tensor outputTensor5 = i.getOutputTensor(5);
//        // input tensor information
//        int [] inputShape = inputTensor.shape(); // 1, 2, 128, 128
//        int inputHeight = inputShape[2];
//        int inputWidth = inputShape[3];
//        DataType type = inputTensor.dataType(); //FLOAT32
//        //output tensor information
//        int [] outputShape0 = outputTensor0.shape(); // [1, 512] id_feature
//        DataType type0 = outputTensor0.dataType(); // FLOAT32
//        int [] outputShape1 = outputTensor1.shape(); // [1, 32] liveness
//        DataType type1 = outputTensor1.dataType(); // FLOAT32
//        int [] outputShape2 = outputTensor2.shape(); // [1, 2] eye_closeness
//        DataType type2 = outputTensor2.dataType(); // FLOAT32
//        int [] outputShape3 = outputTensor3.shape(); // [1, 2] glasses
//        DataType type3 = outputTensor3.dataType(); // FLOAT32
//        int [] outputShape4 = outputTensor4.shape(); // [1, 2] mask
//        DataType type4 = outputTensor4.dataType(); // FLOAT32
//        int [] outputShape5 = outputTensor5.shape(); // [1, 2] sunglasses
//        DataType type5 = outputTensor5.dataType(); // FLOAT32


//        ByteBuffer input = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4);
//        input = bitmapToByteBuffer(resizedSampleImage);
//        //input = this.getGrayScaleByteBuffer(resizedSampleImage);
//        input.order(ByteOrder.nativeOrder());
//
//        // For FLOAT 32 use image processor
//        ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
//        TensorImage tImg = TensorImage.fromBitmap(resizedSampleImage);
//        input = imageProcessor.process(tImg).getBuffer();
//
//        Log.d("Interpreter", "Run interpreter");
//        Object[] inputs = new Object[1];
//        inputs[0] = input;


        //input shape incorrect when using greyscale


        // set up output buffers using output tensor shapes
//        Map<Integer, Object> outputs = new HashMap<>();
//        FloatBuffer fbuffer = FloatBuffer.allocate(512);
//        outputs.put(0, fbuffer);
//        fbuffer = FloatBuffer.allocate(32);
//        outputs.put(1, fbuffer);
//        fbuffer = FloatBuffer.allocate(2);
//        outputs.put(2, fbuffer);
//        fbuffer = FloatBuffer.allocate(2);
//        outputs.put(3, fbuffer);
//        fbuffer = FloatBuffer.allocate(2);
//        outputs.put(4, fbuffer);
//        fbuffer = FloatBuffer.allocate(2);
//        outputs.put(5, fbuffer);

        ByteBuffer sampleInput = this.preprocessSampleImage();
        Object[] inputs = this.setUpInputBuffer(sampleInput);
        Map<Integer, Object> outputs = this.setUpOutputBuffers();
        i.runForMultipleInputsOutputs(inputs, outputs);
        this.rawOutputs = outputs;
        this.setOriginalEmbedding(); // liveness embeddings
        this.setCurrentEmbedding();
        i.close();
    }

    // Set original embedding for liveness
    private void setOriginalEmbedding()
    {
        Object sunglassesResults = this.rawOutputs.get(1);
        FloatBuffer buffer = (FloatBuffer) sunglassesResults;
        this.originalLivenessEmbedding = buffer;
    }

    // Set current embedding for liveness
    private void setCurrentEmbedding()
    {
        Object sunglassesResults = this.rawOutputs.get(1);
        FloatBuffer buffer = (FloatBuffer) sunglassesResults;
        this.currentLivenessEmbedding = buffer;
    }

    // Calculate softmax for a given element in a FloatBuffer
    // input = element in FloatBuffer
    // buffer = Entire FloatBuffer
    public static double softmax(double input, FloatBuffer buffer)
    {
        // Calculates e^element / sum(e^i element) for all i in buffer where i = 0 to sizeOfBuffer
        double result = 0.0; // Return value
        double total = 0.0; // denominator = sum of e^i element of buffer for all elements

        buffer.flip(); // For reading the buffer set pos to 0
        while(buffer.hasRemaining())
        {
            double element = buffer.get();
            total += Math.exp(element); // e^(element)
        }

        result = Math.exp(input) / total; // softmax formula for single element
        return result;
    }

    // L2 loss function for liveness
    // L2 = sum((y_true - y_pred)^2) where y = element from embedding
    // Loss closer to 0 means the liveness check passes
    private double L2LossFunc()
    {
        double loss = 0.0;

        FloatBuffer original = this.originalLivenessEmbedding;
        FloatBuffer current = this.currentLivenessEmbedding;

        original.flip(); // Set pos to 0
        current.flip();  // Set pos to 0

        // For each element of both embeddings
        while (original.hasRemaining())
        {
            // Find their difference squared
            double y_true = original.get();
            double y_pred = current.get();
            double diff = y_true - y_pred;
            double square = Math.pow(diff, 2);
            // add to final result
            loss += square;
        }

        return loss;
    }

    // Preprocess image, resize turn into bitmap then bytebuffer
    private ByteBuffer preprocessSampleImage() throws IOException {
        Bitmap resizedSampleImage = this.openSampleImageAsBitMap();
        ByteBuffer input = ByteBuffer.allocateDirect(128 * 128 * 3 * 4);
        input = bitmapToByteBuffer(resizedSampleImage);
        //input = this.getGrayScaleByteBuffer(resizedSampleImage);
        input.order(ByteOrder.nativeOrder());

        // For FLOAT 32 use image processor
        ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        TensorImage tImg = TensorImage.fromBitmap(resizedSampleImage);
        return imageProcessor.process(tImg).getBuffer();
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
        FloatBuffer fbuffer = FloatBuffer.allocate(512);
        outputs.put(0, fbuffer);  // id_feature
        fbuffer = FloatBuffer.allocate(32);
        outputs.put(1, fbuffer);  // liveness
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(2, fbuffer);  // eye_closeness
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(3, fbuffer);  // glasses
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(4, fbuffer);  // mask
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(5, fbuffer);  // sunglasses
        return outputs;
    }

    // Postprocessing computations

    // Compute liveness
    public void computeLiveness()
    {
        this.computeLivenessLoss();
        this.computeLivenessBoolean();
    }

    // Compute eyeCloseness
    public void computeEyeCloseness()
    {
        this.computeEyeClosenessProbLR();
        this.computeEyeClosenessBoolean();
    }

    // Compute sunglasses
    public void computeSunglasses()
    {
        this.computeSunglassesProb();
        this.computeSunglassesBoolean();
    }

    // Compute eyeCloseness probability and boolean for left and right eyes
    private void computeEyeClosenessProbLR()
    {
        Object eyeClosenessResults = this.rawOutputs.get(2);
        //Class<?> classtype = eyeClosenessResults.getClass();
        FloatBuffer buffer = (FloatBuffer) eyeClosenessResults;
        double leftEyeClosenessProbability = softmax(buffer.get(0), buffer);
        double rightEyeClosenessProbability = softmax(buffer.get(1), buffer);
        this.probEyeClosenessL = leftEyeClosenessProbability;
        this.probEyeClosenessR = rightEyeClosenessProbability;
    }

    // Compute sunglasses probability
    private void computeSunglassesProb()
    {
        Object sunglassesResults = this.rawOutputs.get(5);
        FloatBuffer buffer = (FloatBuffer) sunglassesResults;
        this.probSunglasses = buffer.get(0); // Probability sunglasses true
    }

    // Compute liveness loss
    private void computeLivenessLoss()
    {
        double loss = this.L2LossFunc();
        this.livenessLoss = loss;
    }

    // Compute eyecloseness booleans
    private void computeEyeClosenessBoolean()
    {
        if (this.getEyeClosenessProbL() > 0.90)
        {
            this.eyeClosenessL = true;
        }
        else
        {
            this.eyeClosenessL = false;
        }
        if (this.getEyeClosenessProbR() > 0.90)
        {
            this.eyeClosenessR = true;
        }
        else
        {
            this.eyeClosenessR = false;
        }
    }

    // Compute sunglasses booleans
    private void computeSunglassesBoolean()
    {
        if (this.getSunglassesProb() > 0.90)
        {
            this.sunglasses = true;
        }
        else
        {
            this.sunglasses = false;
        }
    }

    // Compute liveness boolean
    private void computeLivenessBoolean()
    {
        // If loss is close to 0
        if (this.getLivenessLoss() < 10.0)
        {
            this.liveness = true;
        }
        else
        {
            this.liveness = false;
        }
    }

    // Accessors

    // Get eyeClosenessProbL
    public double getEyeClosenessProbL()
    {
        return this.probEyeClosenessL;
    }

    // Get eyeClosenessProbR
    public double getEyeClosenessProbR()
    {
        return this.probEyeClosenessR;
    }

    // Get sunglassesProb
    public double getSunglassesProb()
    {
        return this.probSunglasses;
    }

    // Get liveness loss
    public double getLivenessLoss()
    {
        return this.livenessLoss;
    }

    // Get eyeClosenessL boolean
    public boolean getEyeClosenessL()
    {
        return this.eyeClosenessL;
    }

    // Get eyeClosenessR boolean
    public boolean getEyeClosenessR()
    {
        return this.eyeClosenessR;
    }

    // Get sunglasses boolean
    public boolean getSunglasses()
    {
        return this.sunglasses;
    }

    // Get liveness boolean
    public Boolean getLiveness()
    {
        return this.liveness;
    }






}
