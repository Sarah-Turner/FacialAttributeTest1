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
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class FaceAttributeModel {
    private Context context;

    // Get Context from MainActivity
    public FaceAttributeModel(Context context)
    {
        this.context = context;
    }

    // Open image from assets and return its bitmap
    public Bitmap openSampleImageAsBitMap() throws IOException {
        Bitmap bMap = BitmapFactory.decodeStream(context.getAssets().open("sample_image.jpg"));

        // Resize to 128 by 128
        bMap = resizeAndPadMaintainAspectRatio(bMap, 128, 128, 0);

        return bMap;
    }

    // Load the tflite file for facial attribute model
    public MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor=context.getAssets().openFd("face_attrib_net.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
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
    public ByteBuffer getGrayScaleByteBuffer(Bitmap bitmap){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        ByteBuffer mImgData = ByteBuffer
                .allocateDirect(4 * width * height);
        mImgData.order(ByteOrder.nativeOrder());
        int[] pixels = new int[width*height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int pixel : pixels) {
            mImgData.putFloat((float) Color.red(pixel));
        }
        return mImgData;
    }

    // Create the interpreter and run inference
    public void setUpInterpreter(MappedByteBuffer tflitemodel) throws IOException {
        Log.d("Interpreter", "Setting up interpreter");
        Interpreter i = new Interpreter(tflitemodel);
        i.allocateTensors();
        Log.d("Interpreter", "Input tensor count = " + Integer.toString(i.getInputTensorCount()));
        Log.d("Interpreter", "Output tensor count = " + Integer.toString(i.getOutputTensorCount()));
        Log.d("Interpreter", "Open and resize image as bitmap");
        Bitmap resizedSampleImage = this.openSampleImageAsBitMap();
        //Log.d("Interpreter", "Turn bitmap into greyscale bytebuffer");


        Tensor inputTensor = i.getInputTensor(0);
        Tensor outputTensor0 = i.getOutputTensor(0);
        Tensor outputTensor1 = i.getOutputTensor(1);
        Tensor outputTensor2 = i.getOutputTensor(2);
        Tensor outputTensor3 = i.getOutputTensor(3);
        Tensor outputTensor4 = i.getOutputTensor(4);
        Tensor outputTensor5 = i.getOutputTensor(5);
        // input tensor information
        int [] inputShape = inputTensor.shape(); // 1, 2, 128, 128
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];
        DataType type = inputTensor.dataType(); //FLOAT32
        //output tensor information
        int [] outputShape0 = outputTensor0.shape(); // [1, 512] id_feature
        DataType type0 = outputTensor0.dataType(); // FLOAT32
        int [] outputShape1 = outputTensor1.shape(); // [1, 32] liveness
        DataType type1 = outputTensor1.dataType(); // FLOAT32
        int [] outputShape2 = outputTensor2.shape(); // [1, 2] eye_closeness
        DataType type2 = outputTensor2.dataType(); // FLOAT32
        int [] outputShape3 = outputTensor3.shape(); // [1, 2] glasses
        DataType type3 = outputTensor3.dataType(); // FLOAT32
        int [] outputShape4 = outputTensor4.shape(); // [1, 2] mask
        DataType type4 = outputTensor4.dataType(); // FLOAT32
        int [] outputShape5 = outputTensor5.shape(); // [1, 2] sunglasses
        DataType type5 = outputTensor5.dataType(); // FLOAT32


        ByteBuffer input = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4);
        input = bitmapToByteBuffer(resizedSampleImage);
        //input = this.getGrayScaleByteBuffer(resizedSampleImage);
        input.order(ByteOrder.nativeOrder());

        // For FLOAT 32 use image processor
        ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        TensorImage tImg = TensorImage.fromBitmap(resizedSampleImage);
        input = imageProcessor.process(tImg).getBuffer();

        Log.d("Interpreter", "Run interpreter");
        Object[] inputs = new Object[1];
        inputs[0] = input;


        //input shape incorrect when using greyscale


        // set up output buffers using output tensor shapes
        Map<Integer, Object> outputs = new HashMap<>();
        FloatBuffer fbuffer = FloatBuffer.allocate(512);
        outputs.put(0, fbuffer);
        fbuffer = FloatBuffer.allocate(32);
        outputs.put(1, fbuffer);
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(2, fbuffer);
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(3, fbuffer);
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(4, fbuffer);
        fbuffer = FloatBuffer.allocate(2);
        outputs.put(5, fbuffer);

        i.runForMultipleInputsOutputs(inputs, outputs);

        i.close();
    }
}
