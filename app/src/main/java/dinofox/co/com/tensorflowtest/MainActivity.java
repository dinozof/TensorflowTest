package dinofox.co.com.tensorflowtest;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.ajithvgiri.canvaslibrary.CanvasView;
import com.simplify.ink.InkView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt4;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static android.graphics.Bitmap.Config.ALPHA_8;
import static android.graphics.Bitmap.Config.ARGB_8888;

public class MainActivity extends AppCompatActivity {


    private static final String TAG="textscan";
    private CanvasView canvasView;
    //private InkView ink;
    private TextView mResultText;
    private RelativeLayout parentView;
    private Button buttonclear;
    private Switch switchB;
    private Boolean developerEnabled=false;

    private  Button buttonSubmit;
    private  Button buttonSave;
    private Bitmap scaledImage;
    private static final int PIXEL_WIDTH = 28;
    private boolean invertImageColor=true;


    private static final int INPUT_SIZE = 28;
    private  int IMAGE_MEAN = 0;
    private  float IMAGE_STD = 0;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";


    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar myToolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(myToolbar);
        myToolbar.inflateMenu(R.menu.menu);

        parentView = findViewById(R.id.parentView);
        canvasView = new CanvasView(this);




        parentView.addView(canvasView);


        switchB = findViewById(R.id.switch1);
        switchB.setChecked(true);



        buttonSave = findViewById(R.id.buttonSave);
        buttonclear = findViewById(R.id.buttonClear);
        buttonSubmit = findViewById(R.id.buttonDetect);
        mResultText = findViewById(R.id.resultDisplay);
        buttonclear.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                canvasView.clearCanvas();

                mResultText.setText(R.string.waiting);
                // Code here executes on main thread after user presses button
            }
        });

        switchB.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                // do something, the isChecked will be
                // true if the switch is in the On position
                if(isChecked){
                    invertImageColor=true;
                    LinearLayout layout = findViewById(R.id.toplinearl);
                    layout.setBackgroundColor(Color.BLACK);
                    switchB.setBackgroundColor(Color.BLACK);
                    switchB.setTextColor(Color.WHITE);
                    Toast.makeText(MainActivity.this, R.string.reverseColorBlack, Toast.LENGTH_SHORT).show();
                }
                else{
                    invertImageColor=false;
                    //switchB.setBackgroundColor(Color.WHITE);
                    LinearLayout layout = findViewById(R.id.toplinearl);
                    layout.setBackgroundColor(Color.WHITE);
                    switchB.setBackgroundColor(Color.WHITE);
                    switchB.setTextColor(Color.BLACK);
                    Toast.makeText(MainActivity.this, R.string.reverseColorWhite, Toast.LENGTH_SHORT).show();
                }
            }
        });

        buttonSave.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (!checkWriteExternalPermission()){
                    Log.d(TAG, "No Permission");
                    ActivityCompat.requestPermissions(MainActivity.this,
                            new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                            1);
                }else{
                    parentView.setDrawingCacheEnabled(true);
                    Bitmap image=parentView.getDrawingCache();
                    Bitmap toBeSaved = scaleAndBinarize(image,invertImageColor);
                    parentView.setDrawingCacheEnabled(false);
                    saveImage(toBeSaved);

                }
                // Code here executes on main thread after user presses button
            }
        });

        buttonSubmit.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                parentView.setDrawingCacheEnabled(true);
                Bitmap image=parentView.getDrawingCache();
                image.setConfig(ARGB_8888);
                Log.d(TAG,"invert: "+invertImageColor);
                scaledImage = scaleAndBinarize(image,invertImageColor);
                parentView.setDrawingCacheEnabled(false);


                float[] pixels;
                if(invertImageColor){
                    pixels=retrivePixelData(scaledImage);
                }else{
                    pixels = getPixelData(scaledImage);
                }



                final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

                if (results.size() > 0) {
                    String value = " Number is : " +results.get(0).getTitle();
                    mResultText.setText(value);
                }

                // Code here executes on main thread after user presses button
            }
        });

        initTensorFlowAndLoadModel();
        buttonSave.setEnabled(false);
        buttonSubmit.setEnabled(false);
        LinearLayout layout = findViewById(R.id.toplinearl);
        layout.setBackgroundColor(Color.BLACK);
        switchB.setBackgroundColor(Color.BLACK);
        switchB.setTextColor(Color.WHITE);
        layout.setVisibility(View.INVISIBLE);
    }

    @Override
    public void onResume(){
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0,this,
                mLoaderCallback);
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private boolean checkWriteExternalPermission()
    {
        return checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)==PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {

                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    saveImage(scaledImage);
                    Toast.makeText(MainActivity.this, R.string.imageSaved, Toast.LENGTH_SHORT).show();

                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(MainActivity.this, R.string.permissionDenied, Toast.LENGTH_SHORT).show();
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    private void saveImage(Bitmap finalBitmap) {

        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/Download");
        Random generator = new Random();
        int n = 10000;
        n = generator.nextInt(n);
        String fname = "Image-"+ n +".jpg";
        File file = new File (myDir, fname);
        if (file.exists ())
            file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
            Toast.makeText(MainActivity.this, getString(R.string.imageSavedLocationMessage)+myDir, Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private Bitmap scaleAndBinarize(Bitmap image, boolean invert){

        Mat tmp = new Mat (image.getHeight(), image.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(image,tmp);

        //reduce image to correct size for tensorflow model
        Mat resized = new Mat(PIXEL_WIDTH,PIXEL_WIDTH,tmp.type());
        Imgproc.resize(tmp,resized,resized.size(),0,0,Imgproc.INTER_AREA);

        resized = blur(resized);

        //threshold to binarize result
        Mat binarized =new Mat(PIXEL_WIDTH,PIXEL_WIDTH,tmp.type());

        Imgproc.threshold(resized, binarized , 250, 255, Imgproc.THRESH_BINARY);



        Bitmap end = Bitmap.createBitmap(resized.cols(), resized.rows(),ARGB_8888 );


        if (invert){ //need black background and white number?
            Log.d(TAG,"inverting colors...");
            Mat inverted =new Mat(PIXEL_WIDTH,PIXEL_WIDTH,tmp.type());
            Mat greyImage =new Mat();
            //NON PASSAVO A SCALA DI GRIGI
            Imgproc.cvtColor(binarized,greyImage, Imgproc.COLOR_RGB2GRAY);
            Core.bitwise_not(greyImage,inverted);

            MatOfDouble mean = new MatOfDouble();
            MatOfDouble std = new MatOfDouble();


            Core.meanStdDev(inverted, mean, std);
            IMAGE_MEAN= (int) mean.toArray()[0];
            IMAGE_STD = (float) std.toArray()[0];
            Utils.matToBitmap(inverted, end);
        }
        else {Utils.matToBitmap(binarized, end);}


        return end;
    }



    public float [] retrivePixelData(Bitmap scaledImage){
        float[] pixels;
        Log.d(TAG,"trying with original inverted image");
        int width=scaledImage.getWidth();
        int height = scaledImage.getHeight();
        int[]pix = new int[width*height];
        scaledImage.getPixels(pix, 0, width, 0, 0, width, height);
        pixels = new float[pix.length];
        for (int i=0; i<pix.length; i++){
            int c =pix[i];
            int b = c & 0xff; //255
            pixels[i]=b;
            Log.d(TAG, pix.length+" pixel inverted: "+String.valueOf(b)+" "+String.valueOf(Color.alpha(pix [i])));
        }

        return pixels;
    }


    /**
     * Return pixel data inverted.
     *
     * @deprecated use {@link #retrivePixelData(Bitmap)} instead.
     */
    @Deprecated
    public float[] getPixelData(Bitmap mOffscreenBitmap) {
        if (mOffscreenBitmap == null) {
            return null;
        }

        int width = mOffscreenBitmap.getWidth();
        int height = mOffscreenBitmap.getHeight();

        // Get 28x28 pixel data from bitmap
        int[] pixels = new int[width * height];
        mOffscreenBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] retPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff; //255
            Log.d(TAG,"pixel:"+String.valueOf(b));
            retPixels[i] = 0xff - b;


        }
        return retPixels;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    // Questo oggetto callback è usato quando inizializzaimo la libreria OpenCV
    // in modo asincrono
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        // Una volta che OpenCV manager è connesso viene chiamato questo metodo di
        public void onManagerConnected(int status) {
            switch (status) {
                // Una volta che OpenCV manager si è connesso con successo
                // possiamo abilitare l'interazione con la tlc
                case LoaderCallbackInterface.SUCCESS:
                    buttonSave.setEnabled(true);
                    buttonSubmit.setEnabled(true);
                    Log.i(TAG, "OpenCV loaded successfully");
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            } } };


    public Mat blur(Mat input){ //blur out the image
        Mat sourceImage = new Mat();
        Mat destImage= new Mat();

        sourceImage =input.clone();
        Mat  gaussianImage = new Mat();
        Imgproc.blur(sourceImage, gaussianImage, new Size(3,3));
        Imgproc.medianBlur(gaussianImage,destImage,5);

        return destImage;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_settings:
                // User chose the "Settings" item, show the app settings UI...
                return true;

            case R.id.action_developer:
                // User chose the "Developer" action, mark the current item
                // as a favorite...
                developerEnabled=!developerEnabled;
                if(developerEnabled){
                    LinearLayout layout = findViewById(R.id.toplinearl);
                    layout.setVisibility(View.VISIBLE);
                    item.setIcon(R.drawable.ic_cancel_black_24dp);
                    Toast.makeText(MainActivity.this, R.string.nothingUseful, Toast.LENGTH_SHORT).show();
                }else{
                    LinearLayout layout = findViewById(R.id.toplinearl);
                    layout.setVisibility(View.INVISIBLE);
                    item.setIcon(R.drawable.ic_build_black_24dp);
                }


                return true;

            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);

        }
    }

}
