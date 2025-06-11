#include "TrainNumberDetector.h"
#include <iostream>
#include <algorithm> 
#include <limits>    

TrainNumberDetector::TrainNumberDetector() : isProcessing(false), lastReportedNumber("") {
    MAX_EMPTY_FRAMES = 5;
    MIN_LENGTH = 3;
    TRAIN_TYPE = 2;
    resetCurrentRecord();
}

void TrainNumberDetector::processFrame(const std::string& currentNumber, std::string& triannum) {
    try {
        std::lock_guard<std::mutex> lock(mtx);

        if (currentNumber.empty()) {
            handleEmptyFrame(triannum);
            return;
        }

        handleNumberFrame(currentNumber);
    }
    catch (const std::exception& e) {
        std::cerr << "处理过程中发生错误: " << e.what() << std::endl;
    }
}

void TrainNumberDetector::resetCurrentRecord() {
    currentRecord.fragments.clear();
    currentRecord.isReported = false;
    currentRecord.completeNumber.clear();
    currentRecord.emptyFrameCount = 0;
}

void TrainNumberDetector::handleEmptyFrame(std::string& triannum) {
    if (!isProcessing) return;

    currentRecord.emptyFrameCount++;

    if (currentRecord.emptyFrameCount >= MAX_EMPTY_FRAMES) {
        finalizeCurrentTrain(triannum);
    }
}

void TrainNumberDetector::handleNumberFrame(const std::string& number) {
    if (!isProcessing) {
        isProcessing = true;
        resetCurrentRecord();
    }

    if (number.length() < static_cast<size_t>(MIN_LENGTH)) return; 

    currentRecord.emptyFrameCount = 0;

    if (std::find(currentRecord.fragments.begin(), currentRecord.fragments.end(), number)
        == currentRecord.fragments.end()) {
        currentRecord.fragments.push_back(number);
    }
}

void TrainNumberDetector::finalizeCurrentTrain(std::string& finalNumber) {
    if (isProcessing) {
        if (!currentRecord.fragments.empty()) {
            updateCompleteNumber();
            if (currentRecord.completeNumber != lastReportedNumber) { 
                // TODO 可以加入长度检查
                reportNumber(finalNumber);
                lastReportedNumber = currentRecord.completeNumber;
            }
        }
        isProcessing = false;
        resetCurrentRecord();
    }
}

std::string TrainNumberDetector::combineTrainNumber(const std::vector<std::string>& fragments) {
    if (fragments.empty()) return "";

    std::string combinedNumber;
    size_t max_length = 0;
    const std::vector<std::string> validPrefixes = { "CRH", "JC", "SW", "SY", "ZE", "ZY", "CR" };

    // 优先选择以有效前缀开头的最长片段
    for (const auto& prefix : validPrefixes) {
        for (const auto& fragment : fragments) {
            if (fragment.rfind(prefix, 0) == 0 && fragment.length() > max_length) {
                max_length = fragment.length();
                combinedNumber = fragment;
            }
        }
    }

    // 如果没有找到以有效前缀开头的片段，再选择所有片段中最长的
    if (combinedNumber.empty()) {
        max_length = 0;
        for (const auto& fragment : fragments) {
            if (fragment.length() > max_length) {
                max_length = fragment.length();
                combinedNumber = fragment;
            }
        }
    }

    bool hasChanges = true;
    while (hasChanges) {
        hasChanges = false;

        for (const auto& fragment : fragments) {
            if (combinedNumber == fragment || combinedNumber.find(fragment) != std::string::npos) {
                continue;
            }

            // 正向重叠检查
            size_t max_overlap_fwd = std::min(fragment.length(), combinedNumber.length());
            for (size_t i = max_overlap_fwd; i >= 1; --i) {
                if (combinedNumber.size() >= i && fragment.size() >= i && // Boundary check
                    combinedNumber.substr(combinedNumber.size() - i) == fragment.substr(0, i)) {
                    combinedNumber += fragment.substr(i);
                    hasChanges = true;
                    break;
                }
            }

            if (hasChanges) continue;
            
            // 反向重叠检查
            size_t max_overlap_bwd = std::min(combinedNumber.length(), fragment.length());
            for (size_t i = max_overlap_bwd; i >= 1; --i) {
                 if (fragment.size() >= i && combinedNumber.size() >= i && // Boundary check
                    fragment.substr(fragment.size() - i) == combinedNumber.substr(0, i)) {
                    combinedNumber = fragment.substr(0, fragment.size() - i) + combinedNumber;
                    hasChanges = true;
                    break;
                }
            }
        }
    }

    if (TRAIN_TYPE == 2)
    {
        bool isValid = false;
        for (const auto& prefix : validPrefixes) {
            if (combinedNumber.size() >= prefix.size() &&
                combinedNumber.compare(0, prefix.size(), prefix) == 0) {
                isValid = true;
                break;
            }
        }
        if (!isValid) {
            combinedNumber.clear();
        }
        else {
            
            if (combinedNumber.find("CR") != std::string::npos) {
                correctTrainNumber(combinedNumber);
            }

            size_t len = combinedNumber.length();

            if ((len < 8 || len > 9) && 
                (combinedNumber.find("SW") != std::string::npos || combinedNumber.find("SY") != std::string::npos ||
                 combinedNumber.find("ZE") != std::string::npos || combinedNumber.find("ZY") != std::string::npos)) {
                combinedNumber.clear();
            }
        }
    }

    return combinedNumber;
}

void TrainNumberDetector::updateCompleteNumber() {
    if (currentRecord.fragments.empty()) return;
    currentRecord.completeNumber = combineTrainNumber(currentRecord.fragments);
}

void TrainNumberDetector::reportNumber(std::string& trainNum) {
    trainNum = currentRecord.completeNumber;
    std::cout << "实时检测到车号: " << currentRecord.completeNumber << std::endl;
    std::cout << "识别过程片段: ";
    for (size_t i = 0; i < currentRecord.fragments.size(); ++i) {
        std::cout << currentRecord.fragments[i] << (i == currentRecord.fragments.size() - 1 ? "" : " -> ");
    }
    std::cout << std::endl;
}

void TrainNumberDetector::correctTrainNumber(std::string& input) {
    std::vector<std::string> validPrefixes = {
        "CRH5A", "CRH5G", "CRH380BL", "CRH380BG", "CRH380B" ,"CRH380BJA", "CRH380CL", "CR400BF", "CR400BFA", 
        "CR400BFG","CR400BFB","CR400BFZ","CR400BFJ","CR400BFBS","CR400BFGS", 
        "CR300BF", "CRH5J","CRH5E", "CRH3AA", "CRH3A"
    };
    
    auto [prefix, number] = splitPrefixNumber(input); 
    
    // 如果前缀已合法，不修改 input
    for (const auto& valid : validPrefixes) {
        if (prefix == valid) return;
    }
    
    // 找最相似合法前缀
    int minDist = std::numeric_limits<int>::max();
    std::string bestMatch;
    for (const auto& valid : validPrefixes) {
        int dist = levenshtein(prefix, valid);
        if (dist < minDist) {
            minDist = dist;
            bestMatch = valid;
        }
    }
    
    // 替换 input 内容
    if (!bestMatch.empty()) { // Ensure a best match was found
       input = bestMatch + number;
    } else if (!prefix.empty() && !number.empty()){ 
       input = prefix + number; 
    } else if (!prefix.empty()){
        input = prefix;
    } else {
        input = number; 
    }
}

// Levenshtein 距离算法
int TrainNumberDetector::levenshtein(const std::string& s1, const std::string& s2) {
    size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

    for (size_t i = 0; i <= len1; i++) dp[i][0] = static_cast<int>(i);
    for (size_t j = 0; j <= len2; j++) dp[0][j] = static_cast<int>(j);

    for (size_t i = 1; i <= len1; i++) {
        for (size_t j = 1; j <= len2; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({ 
                dp[i - 1][j] + 1,     // 删除
                dp[i][j - 1] + 1,     // 插入
                dp[i - 1][j - 1] + cost // 替换
            });
        }
    }
    return dp[len1][len2];
}


std::pair<std::string, std::string> TrainNumberDetector::splitPrefixNumber(const std::string& s) {
    size_t pos = s.find_last_not_of("0123456789");
    if (pos == std::string::npos) {
        // 整个字符串是数字
        if (!s.empty() && s.find_first_not_of("0123456789") == std::string::npos) { 
            return { "", s };
        }
        // 整个字符串不是数字，但也没有找到最后一个非数字字符
        return { s, "" }; 
    } else {
        std::string prefix = s.substr(0, pos + 1);
        std::string number = s.substr(pos + 1);
        return { prefix, number };
    }
}