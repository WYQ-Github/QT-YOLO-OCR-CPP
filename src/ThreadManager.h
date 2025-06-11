#pragma once
#include <thread>
#include <vector>
#include <memory>
#include <atomic> // Added for std::atomic
#include <QObject>
#include <optional>
#include "configread.h"
class ThreadManager : public QObject
{
    Q_OBJECT
public:
    ThreadManager(QObject *parent = nullptr);
    ~ThreadManager();


    void startThreads();
    void stopThreads();

private:

    bool UdpToolRecvMessage();
    bool UdpProcessMessage();
    bool PicProcessThread();

    void visualize(cv::Mat& image, const deploy::DetectRes& result, const std::vector<std::string>& labels);
    std::string getCurrentNum(const deploy::DetectRes& result, const std::vector<std::string>& labels, int image_width, int image_height, float margin);
    cv::Mat getStitchImageOptimized( const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3,
                                                    double height_reduction_factor, const cv::Size& final_target_size) ;
    deploy::DetectRes preprocess_detection_result(const deploy::DetectRes& yolo_detection_result,int image_width, int image_height);
public:

    struct StitchedImageData {
        int flag = 0; 
        cv::Mat image;
        std::string imageSequenceNumber;
        std::string timestamp;
    };

    std::queue<StitchedImageData> m_queue_picProcess;
    std::shared_ptr<std::mutex> m_mtx_picProcess;

private:
    std::atomic<bool> threadStop;

    std::queue<std::string> m_queue_udpTool;
    std::shared_ptr<std::mutex> m_mtx_udpTool;

    std::queue<std::string> m_queue_udpProcess;
    std::shared_ptr<std::mutex> m_mtx_udpProcess;

    std::unique_ptr<UdpTool> m_udpTool;
    std::unique_ptr<ConfigRead> m_ConfigRead;

    std::unique_ptr<PaddleOCR> m_paddleOcr;

    UdpToolParam m_udpToolParam;
    GlobalParam m_GlobalParam;
    AlgorithmParam m_AlgParam;

    PaddleOCR::ParamsOCR m_ParamsOCR;

    std::vector<std::shared_ptr<std::thread>> m_threads;
    std::unique_ptr<deploy::DetectModel> m_detector;
    
    // 算法处理相关
    std::unique_ptr<TrainParser> m_trainParser;
    std::unique_ptr<TrainNumberDetector> m_trainNumberDetector;
    std::unique_ptr<MetroTrainParser> m_MetrotrainParser;
    int trainNumCount;
    std::string trianString;
    std::vector<std::string> trianNums;

private:

    struct TaskRecord {
        std::string taskId; 
        std::chrono::system_clock::time_point timestamp;  
    }; 
    std::optional<TaskRecord> lastTask; 
    static constexpr int64_t TASK_WINDOW = 10; 

    std::shared_ptr<LogManager> logManager = LogManager::getInstance("Logs/log.txt", spdlog::level::info);
    std::shared_ptr<Logger> m_logger = logManager->getLogger();
    // std::vector<std::string> m_labels = {"1"} ;
    std::vector<std::string> m_labels = {
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","P","Q","R","S","T","U",
        "V","W","X","Y","Z","0","1","2","3","4",
        "5","6","7","8","9"
    };
signals:
    void m_Logs(const QString& log);
    void m_ShowString(const std::string& str);
    void m_UpdateProgress(int current, int total);
    void m_UpdateCurrentGroup(const QString& groupInfo);

};