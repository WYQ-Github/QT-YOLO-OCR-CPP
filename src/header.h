#pragma once
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <iostream> 
#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <algorithm>
#include "3rdparty/SimpleIni.hpp"
#include "3rdparty/udp/UdpTool.h"
#include "3rdparty/spdlog/spdlog.h"
#include "3rdparty/spdlog/LogManager.h"
#include "yolo/model.hpp"
#include "yolo/option.hpp"
#include "yolo/result.hpp"
#include "filetools.h"
#include "algorithm/CRHTrainTypeAlg.h"
#include "algorithm/TrainNumberDetector.h"
#include "algorithm/MetroTypeAlg.h"
#include "ocr/paddleocr.h"
#include <future>
struct GlobalParam {
    std::string modelPath, imagePath, YOLOPath;  // 模型路径和侧部图片根目录
    std::string OCRRecPath, OCRDetPath, OCRClsPath;
    std::string dictPath, savePath;
    int recMode;
    int resizeWidth, reiszeHeight;     // Resize大小
    double factor;                     // 缩放因子
    bool isSave;
};

struct AlgorithmParam {
    int max_empty_frames;
    int min_length;
    int trian_type;
};


