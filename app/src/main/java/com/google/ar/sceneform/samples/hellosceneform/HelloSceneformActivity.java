/*
 * Copyright 2018 Google LLC. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ar.sceneform.samples.hellosceneform;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import com.google.ar.core.Anchor;
import com.google.ar.core.Pose;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ViewRenderable;
import com.google.ar.sceneform.samples.TensorFlow.Classifier;
import com.google.ar.sceneform.samples.TensorFlow.TensorFlowMultiBoxDetector;
import com.google.ar.sceneform.samples.TensorFlow.TensorFlowObjectDetectionAPIModel;
import com.google.ar.sceneform.samples.TensorFlow.TensorFlowYoloDetector;
import com.google.ar.sceneform.samples.TensorFlow.tracking.MultiBoxTracker;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * This is an example activity that uses the Sceneform UX package to make common AR tasks easier.
 */
public class HelloSceneformActivity extends AppCompatActivity {
    private static final String TAG = HelloSceneformActivity.class.getSimpleName();
    private static final double MIN_OPENGL_VERSION = 3.0;

    private ArFragment arFragment;
    private ModelRenderable lapTopRenderable;
    private ModelRenderable teapotRenderable;
    private ModelRenderable remoteRenderable;


    public ArrayList<String > itemsDisplayed = new ArrayList<>(Arrays.asList("tv","keyboard",
            "cup"));

    private MultiBoxTracker tracker;
    private Classifier detector;
    private static final float TEXT_SIZE_DIP = 24;

    private static final int MB_INPUT_SIZE = 300;
    private static final int MB_IMAGE_MEAN = 128;
    private static final float MB_IMAGE_STD = 128;
    private static final String MB_INPUT_NAME = "ResizeBilinear";
    private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
    private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
    private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
    private static final String MB_LOCATION_FILE =
            "file:///android_asset/multibox_location_priors.txt";

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE =
            "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list" +
            ".txt";

    private Boolean loop = true;

    static int cropSize = TF_OD_API_INPUT_SIZE;

    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 300;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;
    final HandlerThread handlerThread = new HandlerThread("PixelCopier");

    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.5f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.5f;


    @Override
    protected void onDestroy() {
        super.onDestroy();
        loop = false;
    }

    @Override
    @SuppressWarnings({"AndroidApiChecker", "FutureReturnValueIgnored"})
    // CompletableFuture requires api level 24
    // FutureReturnValueIgnored is not valid
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!checkIsSupportedDeviceOrFinish(this)) {
            return;
        }

        setContentView(R.layout.activity_ux);
        arFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);

        // When you build a Renderable, Sceneform loads its resources in the background while
        // returning
        // a CompletableFuture. Call thenAccept(), handle(), or check isDone() before calling get().
        ModelRenderable.builder()
                .setSource(this, R.raw.laptop)
                .build()
                .thenAccept(renderable -> lapTopRenderable = renderable)
                .exceptionally(
                        throwable -> {
                            Toast toast =
                                    Toast.makeText(this, "Unable to load lapTopRenderable " +
                                            "renderable", Toast.LENGTH_LONG);
                            toast.setGravity(Gravity.CENTER, 0, 0);
                            toast.show();
                            return null;
                        });

        ModelRenderable.builder()
                .setSource(this, R.raw.teapot)
                .build()
                .thenAccept(renderable -> teapotRenderable = renderable)
                .exceptionally(
                        throwable -> {
                            Toast toast =
                                    Toast.makeText(this, "Unable to load lapTopRenderable " +
                                            "renderable", Toast.LENGTH_LONG);
                            toast.setGravity(Gravity.CENTER, 0, 0);
                            toast.show();
                            return null;
                        });

        ModelRenderable.builder()
                .setSource(this, R.raw.remote)
                .build()
                .thenAccept(renderable -> remoteRenderable = renderable)
                .exceptionally(
                        throwable -> {
                            Toast toast =
                                    Toast.makeText(this, "Unable to load lapTopRenderable " +
                                            "renderable", Toast.LENGTH_LONG);
                            toast.setGravity(Gravity.CENTER, 0, 0);
                            toast.show();
                            return null;
                        });


        Thread r = new Thread() {
            public void run() {
                handlerThread.start();

                while (loop) {
                    try {


                        ArSceneView view = arFragment.getArSceneView();

                        // Create a bitmap the size of the scene view.

                        final Bitmap bitmap = Bitmap.createBitmap(view.getWidth(), view.getHeight(),
                                Bitmap.Config.ARGB_8888);

                        PixelCopy.request(view, bitmap, (copyResult) -> {

                            if (copyResult == PixelCopy.SUCCESS) {


                                processImage(getResizedBitmap(bitmap, cropSize, cropSize));
                            } else {

                            }

                        }, new Handler(handlerThread.getLooper()));


                        if (detector == null) {
                            DisplayMetrics displayMetrics = new DisplayMetrics();
                            getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
                            int height = displayMetrics.heightPixels;
                            int width = displayMetrics.widthPixels;

                            HelloSceneformActivity.this.onPreviewSizeChosen(new Size(width, height),
                                    90);
                        }
                        try {
                            sleep(1000);
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                    } catch (Exception e) {
                        Log.d("testing Exception", e.toString());

                        try {
                            sleep(500);
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                    }
                }
            }
        };


        r.start();
    }

    public static Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    private static byte[] YUV_420_888toNV21(Image image) {
        byte[] nv21;
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        nv21 = new byte[ySize + uSize + vSize];

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        return nv21;
    }


    /**
     * Returns false and displays an error message if Sceneform can not run, true if Sceneform
     * can run
     * on this device.
     * <p>
     * <p>Sceneform requires Android N on the device as well as OpenGL 3.0 capabilities.
     * <p>
     * <p>Finishes the activity if Sceneform can not run
     */
    public static boolean checkIsSupportedDeviceOrFinish(final Activity activity) {
        if (Build.VERSION.SDK_INT < VERSION_CODES.N) {
            Log.e(TAG, "Sceneform requires Android N or later");
            Toast.makeText(activity, "Sceneform requires Android N or later", Toast.LENGTH_LONG)
                    .show();
            activity.finish();
            return false;
        }
        String openGlVersionString =
                ((ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE))
                        .getDeviceConfigurationInfo()
                        .getGlEsVersion();
        if (Double.parseDouble(openGlVersionString) < MIN_OPENGL_VERSION) {
            Log.e(TAG, "Sceneform requires OpenGL ES 3.0 later");
            Toast.makeText(activity, "Sceneform requires OpenGL ES 3.0 or later", Toast.LENGTH_LONG)
                    .show();
            activity.finish();
            return false;
        }
        return true;
    }


    public void processImage(Bitmap croppedBitmap) {

        final long startTime = SystemClock.uptimeMillis();
        if (croppedBitmap == null) {

        }

        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
        final long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;


        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        switch (MODE) {
            case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
            case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
        }

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= minimumConfidence) {


                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        try {


                            if (itemsDisplayed.contains("cup") && result.getTitle().trim()
                                    .toLowerCase().equals("cup")){

                                Log.d("testing", "run: Cup");

                                ViewRenderable.builder()
                                        .setView(getApplicationContext(), R.layout.imageview)
                                        .build()
                                        .thenAccept(renderable -> {

                                            ImageView imageView = (ImageView) renderable.getView();

                                            imageView.setImageResource(R.drawable.teapot);
                                            imageView.setScaleType(ImageView.ScaleType.FIT_XY);


                                            result.setLocation(location);
                                            Pose temp = Pose.makeRotation(0, 0, 0, 0);

                                            final Pose air = Pose.makeTranslation((location.centerX() -
                                                            (cropSize
                                                                    / 2))
                                                            / cropSize,
                                                    (location
                                                            .centerY() - (cropSize / 2)) / cropSize,
                                                    -0.75f);
                                            final Pose newPose = air.compose(temp);


                                            Anchor anchor = arFragment.getArSceneView().getSession()
                                                    .createAnchor
                                                            (newPose);
                                            AnchorNode anchorNode = new AnchorNode(anchor);
                                            anchorNode.setParent(arFragment.getArSceneView().getScene
                                                    ());

                                            // Create the transformable andy and add it to the anchor.
                                            TransformableNode tNode = new TransformableNode(arFragment
                                                    .getTransformationSystem());
                                            tNode.getScaleController().setMinScale(0.1f);
                                            tNode.getScaleController().setMaxScale(0.5f);
                                            tNode.setOnTapListener(new Node.OnTapListener() {
                                                @Override
                                                public void onTap(HitTestResult hitTestResult,
                                                                  MotionEvent
                                                                          motionEvent) {
                                                        anchor.detach();


                                                }
                                            });

                                            tNode.setLocalScale(new Vector3(0.25f, 0.25f, 0.25f));
                                            tNode.setParent(anchorNode);
                                            tNode.setRenderable(renderable);
                                            tNode.select();
                                            itemsDisplayed.remove(result.getTitle().trim()
                                                    .toLowerCase());

                                        });



                            }

                            if (itemsDisplayed.contains(result.getTitle().trim().toLowerCase())){


                                ViewRenderable.builder()
                                        .setView(getApplicationContext(), R.layout.view)
                                        .build()
                                        .thenAccept(renderable -> {

                                            TextView textView = (TextView) renderable.getView();

                                            textView.setText("Buy a new " + result.getTitle().trim()
                                                    .toUpperCase()+ " Today!");

                                            Log.d("testing", "run: " + location.centerX() +
                                                    "|||" +
                                                    location.centerY());


                                            result.setLocation(location);
                                            Pose temp = Pose.makeRotation(0, 0, 0, 0);

                                            final Pose air = Pose.makeTranslation((location.centerX() -
                                                            (cropSize
                                                                    / 2))
                                                            / cropSize,
                                                    (location
                                                            .centerY() - (cropSize / 2)) / cropSize,
                                                    -0.75f);
                                            final Pose newPose = air.compose(temp);


                                            Anchor anchor = arFragment.getArSceneView().getSession()
                                                    .createAnchor
                                                            (newPose);
                                            AnchorNode anchorNode = new AnchorNode(anchor);
                                            anchorNode.setParent(arFragment.getArSceneView().getScene
                                                    ());

                                            // Create the transformable andy and add it to the anchor.
                                            TransformableNode tNode = new TransformableNode(arFragment
                                                    .getTransformationSystem());
                                            tNode.getScaleController().setMinScale(0.1f);
                                            tNode.getScaleController().setMaxScale(0.5f);
                                            tNode.setOnTapListener(new Node.OnTapListener() {
                                                @Override
                                                public void onTap(HitTestResult hitTestResult,
                                                                  MotionEvent
                                                                          motionEvent) {
                                                    Log.d("testing ", "onTap: destroy ");
                                                    anchor.detach();
                                                }
                                            });

                                            tNode.setLocalScale(new Vector3(0.25f, 0.25f, 0.25f));
                                            tNode.setParent(anchorNode);
                                            tNode.setRenderable(renderable);
                                            tNode.select();
                                            itemsDisplayed.remove(result.getTitle().trim()
                                                    .toLowerCase());

                                        });


                                return;
                            }



                            switch (result.getTitle().trim().toLowerCase()) {


                                case "keyboard":
                                    if (lapTopRenderable != null) {
                                        result.setLocation(location);
                                        Pose temp = Pose.makeRotation(0, 0, 0, 0);

                                        final Pose air = Pose.makeTranslation((location.centerX() -
                                                        (cropSize
                                                                / 2))
                                                        / cropSize,
                                                (location
                                                        .centerY() - (cropSize / 2)) / cropSize,
                                                -0.75f);
                                        final Pose newPose = air.compose(temp);


                                        Anchor anchor = arFragment.getArSceneView().getSession()
                                                .createAnchor
                                                        (newPose);
                                        AnchorNode anchorNode = new AnchorNode(anchor);
                                        anchorNode.setParent(arFragment.getArSceneView().getScene
                                                ());


                                        // Create the transformable andy and add it to the anchor.
                                        TransformableNode tNode = new TransformableNode(arFragment
                                                .getTransformationSystem());
                                        tNode.getScaleController().setMinScale(0.1f);
                                        tNode.getScaleController().setMaxScale(0.5f);
                                        tNode.setOnTapListener(new Node.OnTapListener() {
                                            @Override
                                            public void onTap(HitTestResult hitTestResult,
                                                              MotionEvent
                                                    motionEvent) {
                                                Log.d("testing ", "onTap: destroy ");
                                                anchor.detach();
                                            }
                                        });
                                        Log.d("testing detection image", result.getTitle() + 4);

                                        tNode.setLocalScale(new Vector3(0.25f, 0.25f, 0.25f));
                                        tNode.setParent(anchorNode);
                                        tNode.setRenderable(lapTopRenderable);
                                        tNode.select();
                                        Log.d("testing detection image", result.getTitle() + 5);

                                        lapTopRenderable = null;

                                    }

                                    break;


                                case "book":
                                case "tv":
                                    if (remoteRenderable != null) {
                                        result.setLocation(location);
                                        Pose temp = Pose.makeRotation(0, 0, 0, 0);

                                        final Pose air = Pose.makeTranslation((location.centerX() -
                                                        (cropSize
                                                                / 2))
                                                        / cropSize,
                                                (location
                                                        .centerY() - (cropSize / 2)) / cropSize,
                                                -0.75f);
                                        final Pose newPose = air.compose(temp);


                                        Anchor anchor = arFragment.getArSceneView().getSession()
                                                .createAnchor
                                                        (newPose);
                                        AnchorNode anchorNode = new AnchorNode(anchor);
                                        anchorNode.setParent(arFragment.getArSceneView().getScene());


                                        // Create the transformable andy and add it to the anchor.
                                        TransformableNode tNode = new TransformableNode(arFragment
                                                .getTransformationSystem());
                                        tNode.getScaleController().setMinScale(0.1f);
                                        tNode.getScaleController().setMaxScale(0.5f);
                                        tNode.setOnTapListener(new Node.OnTapListener() {
                                            @Override
                                            public void onTap(HitTestResult hitTestResult, MotionEvent
                                                    motionEvent) {
                                                Log.d("testing ", "onTap: destroy ");
                                                anchor.detach();
                                            }
                                        });
                                        tNode.setLocalScale(new Vector3(0.5f, 0.5f, 0.5f));
                                        tNode.setParent(anchorNode);
                                        tNode.setRenderable(remoteRenderable);
                                        tNode.select();
                                        remoteRenderable = null;

                                    }
                                    break;

                            }

                        } catch (Exception e) {

                            Log.d(TAG, "Error" + e.toString());
                        }
                    }
                });


                mappedRecognitions.add(result);
            }
        }
    }

    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources()
                                .getDisplayMetrics());

        tracker = new MultiBoxTracker(getApplicationContext());

        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getApplicationContext().getAssets(),
                            YOLO_MODEL_FILE,
                            YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
        } else if (MODE == DetectorMode.MULTIBOX) {
            detector =
                    TensorFlowMultiBoxDetector.create(
                            getApplicationContext().getAssets(),
                            MB_MODEL_FILE,
                            MB_LOCATION_FILE,
                            MB_IMAGE_MEAN,
                            MB_IMAGE_STD,
                            MB_INPUT_NAME,
                            MB_OUTPUT_LOCATIONS_NAME,
                            MB_OUTPUT_SCORES_NAME);
            cropSize = MB_INPUT_SIZE;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getApplicationContext().getAssets(), TF_OD_API_MODEL_FILE,
                        TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                cropSize = TF_OD_API_INPUT_SIZE;
            } catch (final IOException e) {
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized",
                                Toast.LENGTH_SHORT);
                toast.show();
            }
        }
    }
}
