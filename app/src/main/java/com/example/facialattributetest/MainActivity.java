package com.example.facialattributetest;

import static java.security.AccessController.getContext;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private Button button;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        FaceAttributeModel model1 = null;
        MediaPipeFaceDetectionModel model2 = null;
        try {
            model1 = new FaceAttributeModel(this);
            model2 = new MediaPipeFaceDetectionModel(this);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        button = (Button)findViewById(R.id.button);
        FaceAttributeModel finalFacialAttributeModel = model1;
        MediaPipeFaceDetectionModel finalMediaPipeDetectionModel = model2;
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                // click handling code
                Log.d("BUTTONS", "User tapped the button");
                try {

                    //finalFacialAttributeModel.runInterpreter();
//                    finalFacialAttributeModel.computeEyeCloseness();
//                    finalFacialAttributeModel.computeSunglasses();
//                    finalFacialAttributeModel.computeLiveness();
                    finalMediaPipeDetectionModel.runInterpreter();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
//                Log.d("RESULTS", "EyeClosenessProbL = " + Double.toString(finalFacialAttributeModel.getEyeClosenessProbL()));
//                Log.d("RESULTS", "EyeClosenessProbR = " + Double.toString(finalFacialAttributeModel.getEyeClosenessProbR()));
//                Log.d("RESULTS", "SunglassesProb = " + Double.toString(finalFacialAttributeModel.getSunglassesProb()));
//                Log.d("RESULTS", "Liveness loss = " + Double.toString(finalFacialAttributeModel.getLivenessLoss()));
//                Log.d("RESULTS", "EyeClosenessL = " + Boolean.toString(finalFacialAttributeModel.getEyeClosenessL()));
//                Log.d("RESULTS", "EyeClosenessR = " + Boolean.toString(finalFacialAttributeModel.getEyeClosenessR()));
//                Log.d("RESULTS", "Sunglasses = " + Boolean.toString(finalFacialAttributeModel.getSunglasses()));
//                Log.d("RESULTS", "Liveness = " + Boolean.toString(finalFacialAttributeModel.getLiveness()));
            }
        });
    }


}