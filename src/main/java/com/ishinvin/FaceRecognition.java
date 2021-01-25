package com.ishinvin;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.*;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Author: Mak Sophea
 * Date: 01/25/2021
 */
public class FaceRecognition {

    public static final String FACE_NET_MODEL_PATH = "/pretrained/face_detector/res10_300x300_ssd_iter_140000.caffemodel";
    public static final String FACE_NET_CONFIG_PATH = "/pretrained/face_detector/deploy.prototxt";
    public static final String MASK_NET_MODEL_PATH = "/pretrained/face_mask/mask_detector_optmized.pb";
    public static final String MASK_NET_CONFIG_PATH = "/pretrained/face_mask/mask_detector_optmized.pbtxt";

    public static String getFilePath(String fileName) {

        final InputStream in = FaceRecognition.class.getResourceAsStream(fileName);
        final String tmpdir = System.getProperty("java.io.tmpdir");
        final String filename = fileName.substring(fileName.lastIndexOf("/") + 1);
        final File destFile = new File(tmpdir, filename);
        try {
            Files.copy(in, destFile.toPath());
        } catch (IOException e) {

        }
        // return file.getAbsolutePath();
        return destFile.getAbsolutePath();
    }

    public static String wearingMask(String imageFilename) {


        // required for OpenCV library
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // load model
        Net faceNet = Dnn.readNet(getFilePath(FaceRecognition.FACE_NET_MODEL_PATH), getFilePath(FaceRecognition.FACE_NET_CONFIG_PATH));
        Net maskNet = Dnn.readNet(getFilePath(FaceRecognition.MASK_NET_MODEL_PATH), getFilePath(FaceRecognition.MASK_NET_CONFIG_PATH));

        // reading the image from the file
        //Mat image = Imgcodecs.imread(args.length < 1 ? DEMO_IMAGE : args[0]);
        //Mat image = Imgcodecs.imread("D:\\tmp\\images_vision\\mask\\mask1.jpg");
        //Mat image = Imgcodecs.imread("D:\\tmp\\images_vision\\mask\\mask2.jpg");
        Mat image = Imgcodecs.imread(imageFilename);

        // original image size
        int width = image.cols();
        int height = image.rows();

        // construct a blob from the image
        Size frameSize = new Size(300, 300);
        Scalar mean = new Scalar(104.0f, 177.0f, 123.0f);
        Mat blob = Dnn.blobFromImage(image, 1.0f, frameSize, mean);

        // pass the blob through the network and obtain the face detections
        faceNet.setInput(blob);
        Mat detections = faceNet.forward();
        detections = detections.reshape(1, (int) detections.total() / 7);

        // confidence threshold, default 0.5
        float confThreshold = 0.5f;

        // loop for the detected faces in a frame, in order to classify the mask or no mask
        for (int i = 0; i < detections.rows(); ++i) {
            // filter out weak detections by ensuring the confidence is greater than the minimum confidence
            double confidence = detections.get(i, 2)[0];
            if (confidence < confThreshold) {
                continue;
            }

            // get the predicted bounding box points
            int startX = (int) (detections.get(i, 3)[0] * width);
            int startY = (int) (detections.get(i, 4)[0] * height);
            int endX = (int) (detections.get(i, 5)[0] * width);
            int endY = (int) (detections.get(i, 6)[0] * height);

            // ensure the bounding boxes fall within the dimensions of the frame
            startX = Math.max(0, startX);
            startY = Math.max(0, startY);
            endX = Math.min(width - 1, endX);
            endY = Math.min(height - 1, endY);

            Mat face = image.submat(startY, endY, startX, endX);
            face = Dnn.blobFromImage(face, 1 / 127.5, new Size(224, 224), new Scalar(1.0f), true);
            face.convertTo(face, CvType.CV_32FC3);

            // face mask prediction
            maskNet.setInput(face);
            Mat preds = maskNet.forward();

            // get the classification result
            double mask = preds.get(0, 0)[0];
            double withoutMask = preds.get(0, 1)[0];

            // prepare text and color to draw
            final String label = mask > withoutMask ? "Mask" : "No Mask";
            System.out.println("Label :" + label);
//            Scalar color = label.equals("Mask")? new Scalar(0, 255, 0) : new Scalar(0, 0, 255);
//            label = String.format("%s: %.2f%%", label, Math.max(mask, withoutMask) * 100);
            return label;
        }
        return "";
    }
}
