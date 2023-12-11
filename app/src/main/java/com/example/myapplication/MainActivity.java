package com.example.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private ImageView imageView;
    private Button selectImageButton;
    private Button predictButton;

    private Bitmap selectedImageBitmap;
    private Interpreter interpreter; // Initialize your TensorFlow Lite interpreter

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        selectImageButton = findViewById(R.id.selectImageButton);
        predictButton = findViewById(R.id.predictButton);

        // Initialize your TensorFlow Lite interpreter with the model file
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("YourTag", "Interpreter error"+e);

        }

    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("Bone_classification_final.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void selectImage(View view) {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);
    }

    public void predict(View view) {
        try {
            if (selectedImageBitmap != null) {
                // Preprocess the image
                ByteBuffer inputBuffer = preprocessImage(selectedImageBitmap);

                // Run inference
                float[][] outputArray = new float[1][1]; // Modify based on your model output shape
                interpreter.run(inputBuffer, outputArray);

                // Process the outputArray as needed for your specific model

                // Example: Get the predicted class and confidence
                float confidence = outputArray[0][0];
                String predictedClass = (confidence < 0.5) ? "Fracture" : "No Fracture";

                // Display the result
                Toast.makeText(this, "Predicted class: " + predictedClass + "\nConfidence: " + (confidence * 100) + "%", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error during prediction", Toast.LENGTH_SHORT).show();
            Log.e("YourTag", "This is an error message"+e);

        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();

            try {
                selectedImageBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                imageView.setImageBitmap(selectedImageBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        bitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int channels = 3; // Assuming RGB image

        // Ensure the expected input shape matches the model's requirements
        int[] expectedInputShape = {128, 128, 3};  // Replace with your model's expected input shape

        // Calculate the expected capacity based on the input shape
        int expectedCapacity = expectedInputShape[0] * expectedInputShape[1] * expectedInputShape[2] * 4;  // Assuming 4 bytes per float

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(expectedCapacity);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        for (int value : intValues) {
            byteBuffer.putFloat(((value >> 16) & 0xFF) / 255.0f);
            byteBuffer.putFloat(((value >> 8) & 0xFF) / 255.0f);
            byteBuffer.putFloat((value & 0xFF) / 255.0f);
        }

        byteBuffer.rewind();  // Ensure the buffer position is reset to 0

        return byteBuffer;
    }
}
