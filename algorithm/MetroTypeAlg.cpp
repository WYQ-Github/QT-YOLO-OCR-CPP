#include "MetroTypeAlg.h"
#include <sstream>      // 用于高效构建字符串
#include <stdexcept>    // 用于 std::stoi 的异常处理

void MetroTrainParser::parse(const std::string& input)
{
    direction_.clear();
    trainNumber_.clear();
    trainNumbers_.clear();
    correctedInput_.clear();

    // 1. 按 '#' 分割字符串，获取各个车厢的信息
    std::vector<std::string> parts = splitString(input, '#');

    // 使用 map 按车厢序号存储车号，可以自动排序并处理乱序输入
    std::map<int, std::string> indexedTrainNumbers;
    for (const auto& part : parts) {
        if (part.empty()) continue;

        std::vector<std::string> pair = splitString(part, '&');
        if (pair.size() == 2) {
            try {
                int carriageIndex = std::stoi(pair[0]); // 车厢序号
                const std::string& fullNumber = pair[1]; // 完整车号
                indexedTrainNumbers[carriageIndex] = fullNumber;
            } catch (const std::invalid_argument&) {
                // 如果车厢序号不是有效数字，则忽略
            } catch (const std::out_of_range&) {
                // 如果车厢序号超出范围，则忽略
            }
        }
    }

    // 将 map 中排序后的车号存入 vector
    for(auto const& [key, val] : indexedTrainNumbers) {
        trainNumbers_.push_back(val);
    }
    
    // 如果没有解析到任何有效的车号，则直接返回
    if (trainNumbers_.empty()) {
        return;
    }

    // 2. 判断方向：比较相邻车号的最后一位数字
    int increasing_count = 0;
    int decreasing_count = 0;
    for (size_t i = 0; i + 1 < trainNumbers_.size(); ++i) {
        if (trainNumbers_[i].empty() || trainNumbers_[i+1].empty()) continue;

        int lastDigit1 = trainNumbers_[i].back() - '0'; // 获取最后一个字符并转为数字
        int lastDigit2 = trainNumbers_[i+1].back() - '0';

        if (lastDigit2 > lastDigit1) {
            increasing_count++;
        } else if (lastDigit2 < lastDigit1) {
            decreasing_count++;
        }
    }
    // 根据递增和递减的计数值判断最终方向
    direction_ = (increasing_count > decreasing_count) ? "1" : "2";

    // 3. 获取主车号：找到出现次数最多的车号前缀
    std::unordered_map<std::string, int> baseNumberCounts;
    for (const auto& num : trainNumbers_) {
        if (num.length() > 1) { 
            baseNumberCounts[num.substr(0, num.length() - 1)]++;
        }
    }

    std::string mostCommonBaseNumber;
    int maxCount = 0;
    for (const auto& pair : baseNumberCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostCommonBaseNumber = pair.first;
        }
    }
    trainNumber_ = mostCommonBaseNumber;

    // 4. 修正最终结果：根据主车号、方向和标准车厢数，生成标准格式的字符串
    if (!trainNumber_.empty()) {
        std::stringstream ss;
        bool isForward = (direction_ == "1");

        for (int i = 1; i <= TRAIN_NUM; ++i) {
            ss << "#" << i << "&" << trainNumber_;
            if (isForward) {
                ss << i; // 正向：拼接 1, 2, 3...
            } else {
                ss << (TRAIN_NUM - i + 1); // 反向：拼接 6, 5, 4...
            }
        }
        correctedInput_ = ss.str();
    }
}

std::string MetroTrainParser::getDirection() const {
    return direction_;
}

std::string MetroTrainParser::getTrainNumber() const {
    return trainNumber_;
}

std::string MetroTrainParser::getCorrectedInput() const {
    return correctedInput_;
}

// 原始的 splitString 方法保持不变
std::vector<std::string> MetroTrainParser::splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : str) {
        if (c == delimiter) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}