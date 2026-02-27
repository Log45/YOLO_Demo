#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(){
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }
    int fps = (int) cap.get(CAP_PROP_FPS);
    int width = (int) cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int) cap.get(CAP_PROP_FRAME_HEIGHT);
    
    cv::VideoWriter out("output-cpp.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    try{
        while(true){
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Could not read frame" << endl;
                break;
            }
            CascadeClassifier face_cascade;
            cv::CascadeClassifier(cv.data::haarcascades + 'haarcascade_frontalface_default.xml') >> face_cascade; 
            
            Mat gray_frame;
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY); // Convert to grayscale
            // Optional: Equalize histogram for better performance
            // equalizeHist(gray_frame, gray_frame); 

            vector<Rect> faces;
            // Detect faces: detectMultiScale parameters control detection sensitivity
            face_cascade.detectMultiScale(gray_frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            // Draw rectangles around the detected faces
            for (const auto& face : faces) {
                rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

            out.write(frame);
            cv::imshow("Frame", frame);
            if (cv::waitKey(1) == 27) { // Press 'ESC' to exit
                break;
            }
            break;
        }
    }
    catch (...){
    }
    // if there is an error, close the camera and the output file properly
    cap.release();
    out.release();
    cv::destroyAllWindows();
    
}