#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <utility>

class TrainNumberDetector {
private:
    struct NumberRecord {
        std::vector<std::string> fragments;  // 存储号码片段
        bool isReported;                     // 是否已报告
        std::string completeNumber;          // 最可能的完整车号
        int emptyFrameCount;                 // 连续空帧计数
    };

    NumberRecord currentRecord;              // 当前正在处理的记录
    std::mutex mtx;                          // 互斥锁

    void resetCurrentRecord();
    void handleEmptyFrame(std::string& triannum);
    void handleNumberFrame(const std::string& number);
    void finalizeCurrentTrain(std::string& finalNumber);
    std::string combineTrainNumber(const std::vector<std::string>& fragments);
    void updateCompleteNumber();
    void reportNumber(std::string& trainNum);
    void correctTrainNumber(std::string& input);
    static int levenshtein(const std::string& s1, const std::string& s2);
    static std::pair<std::string, std::string> splitPrefixNumber(const std::string& s); 

public:
    int MAX_EMPTY_FRAMES;                    // 最大连续空帧数用来判断车厢通过
    int MIN_LENGTH;                          // 最小识别长度（调整为0以允许所有非空片段）
    int TRAIN_TYPE;                          // 过车类型 
    std::string lastReportedNumber;          // 存储上一次报告的车号 
    bool isProcessing;                       // 是否正在处理一列车

public:
    TrainNumberDetector();
    void processFrame(const std::string& currentNumber, std::string& triannum);
};