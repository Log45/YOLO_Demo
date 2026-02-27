#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

const char *class_names[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

using Array = std::vector<float>;
using Shape = std::vector<long>;

// std::string model_path = "/Users/log/Documents/code/YOLO_Demo/cpp/yolov11n-pose.onnx";
std::string model_path = "/Users/log/Documents/code/YOLO_Demo/cpp/yolov8n.onnx";

// Read image and convert to ONNX input format (NCHW, normalized to [0, 1])
std::tuple<Array, Shape, cv::Mat> read_image(cv::Mat image, int size)
{
    assert(!image.empty() && image.channels() == 3);
    cv::resize(image, image, {size, size});
    Shape shape = {1, image.channels(), image.rows, image.cols};
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0, {}, {}, true) / 255.f;
    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return {array, shape, image};
}

// Run the model and get the output
std::pair<Array, Shape> process_image(Ort::Session &session, Array &array, Shape shape)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(
        memory_info, (float *)array.data(), array.size(), shape.data(), shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"output"};
    auto output = session.Run({}, input_names, &input, 1, output_names, 1);
    shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    auto ptr = output[0].GetTensorData<float>();
    return {Array(ptr, ptr + shape[0] * shape[1]), shape};
}

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

    // ONNX Runtime BS
    bool use_cuda = false;
    int image_size = 640;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO");
    Ort::SessionOptions options;
    if (use_cuda) Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
    Ort::Session session(env, model_path.c_str(), options);

    try{
        while(true){
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Could not read frame" << endl;
                break;
            }

            std::tuple<Array, Shape, cv::Mat> input = read_image(frame, image_size);
            std::pair<Array, Shape> output = process_image(session, std::get<0>(input), std::get<1>(input));

            // print the output
            cout << "Output shape: [" << output.second[0] << ", " << output.second[1] << "]" << endl;
            // print all values in the output
            for (size_t i = 0; i < output.first.size(); i++) {
                cout << output.first[i] << " ";
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