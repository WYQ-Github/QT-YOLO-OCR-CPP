#pragma once
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <variant>   
#include <optional>  
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdint>   
#include "3rdparty/clipper2/clipper.h"
#include "3rdparty/mtools.hpp"
#include <opencv2/dnn.hpp> 
#include <limits>         
#include <cmath>           
#include <sstream>         
#include <iomanip>       
#include <stdexcept>       

namespace Ort {
    struct Env;
    class Session;
    struct SessionOptions;
    struct MemoryInfo;
    struct Value;
}

namespace yo {
    struct Node {
        char* name = nullptr;
        std::vector<int64_t> dim; 
        Node() = default;
    };
}


class PaddleOCR { 

public:
    struct ParamsOCR {
        bool repeat = false;        // 是否去除重复字符
        int min_area = 64;           // 文字区域最小面积
        float text = 0.5f;           // 文字检测阈值, 0<=text<=1 (Added 'f' for float literal)
        float thresh = 0.5f;	        // 文字区域识别阈值, 0<=thresh<=1 (Added 'f')
        float unclip_ratio = 2.0f;   // 区域扩展强度，1<=unclip_ratio
        const char* dictionary = nullptr;  	// 字典文件路径(dictionary.txt)
    };
    struct Polygon {
        float score;
        std::vector<cv::Point2f> points;
    };
    struct YoloDetectionBox {
        float left, top, right, bottom;
        float score;
    };
private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    MT::OCRDictionary dictionary;

    ParamsOCR params;
    std::vector<Polygon> polygons;
    std::vector<cv::Mat> input_images;
    std::vector<yo::Node> input_nodes_det, input_nodes_rec, input_nodes_cls;
    std::vector<yo::Node> output_nodes_det, output_nodes_rec, output_nodes_cls;

    Ort::Env* env_det = nullptr; 
    Ort::Env* env_cls = nullptr;
    Ort::Env* env_rec = nullptr;

    Ort::Session* session_det = nullptr;
    Ort::Session* session_cls = nullptr;
    Ort::Session* session_rec = nullptr;

    Ort::SessionOptions* options_det = nullptr; 
    Ort::SessionOptions* options_cls = nullptr;
    Ort::SessionOptions* options_rec = nullptr;
    
    Ort::MemoryInfo* memory_info = nullptr; 

    void clear_nodes_vector(std::vector<yo::Node>& nodes);
    void load_onnx_info(Ort::Session* session, std::vector<yo::Node>& input, std::vector<yo::Node>& output, const std::string& onnx_name);


protected:
    void preprocess(cv::Mat &image); 
    //void postprocess(std::vector<Ort::Value>& output_tensors); 
    std::vector<std::string> postprocess(std::vector<Ort::Value>& output_tensors);
    std::vector<cv::Point2f> unclip(const std::vector<cv::Point>& points); 
    float box_score_slow(cv::Mat&pred, const std::vector<cv::Point>& approx); 
    std::vector<Polygon> poly_from_bitmap(cv::Mat &pred, cv::Mat& bitmap);

    std::optional<std::vector<cv::Mat>> infer_det();                                // 文本区域识别,返回文本区域的分割
    std::optional<std::vector<cv::Mat>> infer_cls(std::vector<cv::Mat>& images);    // 文本方向识别
    std::optional<std::vector<Ort::Value>> infer_rec(std::vector<cv::Mat>& images); // 文本内容识别
public:
    PaddleOCR();
    ~PaddleOCR();

    PaddleOCR(const PaddleOCR&) = delete;
    PaddleOCR& operator=(const PaddleOCR&) = delete;

    int setparms(ParamsOCR parms);
    std::variant<bool, std::string> initialize(const std::vector<std::string>& onnx_paths, bool is_cuda);
    std::variant<bool, std::string> inference(cv::Mat &image, std::vector<std::string>& texts); 
    std::variant<bool, std::string> inference_from_custom_boxes(cv::Mat &image, const std::vector<YoloDetectionBox>& custom_boxes, std::vector<std::string>& texts);
};
