#include "CRHTrainTypeAlg.h"

void TrainParser::parse(const std::string& input) {

    direction_.clear();
    trainNumbers_.clear();
    correctedInput_.clear();

    // 1. 按#分割字符串
    std::vector<std::string> parts = splitString(input, '#');

    // 2. 去掉包含"CR"的部分
    std::vector<std::string> filteredParts;
    for (const std::string& part : parts) {
        if (part.find("CR") == std::string::npos) {
            filteredParts.push_back(part);
        }
    }

    // 3. 提取数字部分
    std::vector<std::string> numbers;
    std::vector<std::string> lastTwoDigits;  // 新增：用于存储末尾两位数字
    for (const std::string& part : filteredParts) {
        size_t ampersandPos = part.find('&');
        if (ampersandPos != std::string::npos) {
            std::string modelNumber = part.substr(ampersandPos + 1);
            std::string num = extractNumbers(modelNumber);
            if (!num.empty()) {
                numbers.push_back(num);
                // 提取末尾两位数字用于方向判断
                if (num.length() >= 2) {
                    lastTwoDigits.push_back(num.substr(num.length() - 2));
                }
            }
        }
    }

    // 4. 提取前4位作为车号并统计出现次数
    std::vector<std::string> uniqueTrainNumbers;
    std::map<std::string, int> trainNumberCount; // 默认会排序
    std::vector<std::string> trainNumberPrefixes;
    for (const std::string& num : numbers) {
        std::string prefix = extractTrainNumberPrefix(num);
        if (!prefix.empty()) {
            // 如果这个车号是第一次出现，添加到uniqueTrainNumbers中
            if (trainNumberCount.find(prefix) == trainNumberCount.end()) {
                uniqueTrainNumbers.push_back(prefix);
            }
            trainNumberCount[prefix]++;
            trainNumberPrefixes.push_back(prefix);
        }
    }

    // 5. 判断重连车
    trainNumbers_.clear();
    for (const auto& trainNumber : uniqueTrainNumbers) {
        if (trainNumberCount[trainNumber] >= 4) {
            trainNumbers_.push_back(trainNumber);
        }
    }
    isReconnected_ = (trainNumbers_.size() > 1);

    // 6. 判断正反向
    if (lastTwoDigits.size() < 2) {

        direction_ = "2"; // 默认反向
        return;
    }
    if (isReconnected_)
    {
        // 如果是重连车
        // 按照车号分组末尾两位数字
        std::map<std::string, std::vector<std::string>> trainNumberDigits;

        // 将末尾两位数字按照车号分组
        for (size_t i = 0; i < trainNumberPrefixes.size() && i < lastTwoDigits.size(); ++i) {
            std::string prefix = trainNumberPrefixes[i];
            // 只处理已识别为车号的前缀
            if (std::find(trainNumbers_.begin(), trainNumbers_.end(), prefix) != trainNumbers_.end()) {
                trainNumberDigits[prefix].push_back(lastTwoDigits[i]);
            }
        }

        // 判断每个编组的方向
        std::string combinedDirection;
        // 使用trainNumbers_中的顺序，这已经是按照原始输入顺序的
        for (const auto& trainNumber : trainNumbers_) {
            const auto& digits = trainNumberDigits[trainNumber];
            if (digits.empty()) {
                combinedDirection += "2"; // 默认反向
                continue;
            }

            // 提取数字部分并转换为整数
            std::vector<int> nums;
            for (const std::string& num : digits) {
                nums.push_back(std::stoi(num));
            }

            // 判断是递增还是递减
            int increasingCount = 0;
            int decreasingCount = 0;
            for (size_t i = 0; i < nums.size() - 1; ++i) {
                if (nums[i] < nums[i + 1]) {
                    increasingCount++;
                }
                else if (nums[i] > nums[i + 1]) {
                    decreasingCount++;
                }
            }

            // 根据趋势判断方向
            if (increasingCount > decreasingCount) {
                combinedDirection += "1";  // 正向
            }
            else {
                combinedDirection += "2"; // 反向
            }
        }

        direction_ = combinedDirection;
    }
    else
    {
        // 提取数字部分并转换为整数
        std::vector<int> nums;
        for (const std::string& num : lastTwoDigits) {
            nums.push_back(std::stoi(num));
        }

        // 判断是递增还是递减
        int increasingCount = 0;
        int decreasingCount = 0;
        for (size_t i = 0; i < nums.size() - 1; ++i) {
            if (nums[i] < nums[i + 1]) {
                increasingCount++;
            }
            else if (nums[i] > nums[i + 1]) {
                decreasingCount++;
            }
        }

        // 根据趋势判断方向
        if (increasingCount > decreasingCount) {
            direction_ = "1";  // 正向
        }
        else {
            direction_ = "2"; // 反向
        }
    }

    // 7.修正输入字符串
    if(isReconnected_)
    {
        // 重连车情况下，修正每一个编组的CR开头异常数字
        std::string correctedInput = input;
        std::vector<std::string> parts = splitString(input, '#');
        
        // 判断parts的大小并计算中点
        if (parts.size() >= 2 && trainNumbers_.size() >= 2) {
            size_t midPoint = parts.size() / 2;
            
            // 前半部分用第一个车号修正
            std::string firstTrainNumber = trainNumbers_[0];
            for (size_t i = 0; i < midPoint; ++i) {
                if (parts[i].find("CR") != std::string::npos) {
                    size_t ampersandPos = parts[i].find('&');
                    if (ampersandPos != std::string::npos) {
                        std::string beforeAmpersand = parts[i].substr(0, ampersandPos + 1);
                        std::string afterAmpersand = parts[i].substr(ampersandPos + 1);
                        
                        // 提取数字部分
                        std::string numPart = extractNumbers(afterAmpersand);
                        if (!numPart.empty()) {
                            // 检查数字是否异常（长度超过4位或者不包含车号）
                            if (numPart.length() > 4 || numPart != firstTrainNumber) {
                                // 替换为正确的车号
                                std::string correctedAfterAmpersand = afterAmpersand;
                                size_t numPos = afterAmpersand.find_last_not_of("0123456789");
                                correctedAfterAmpersand = afterAmpersand.substr(0, numPos+1) + firstTrainNumber;
                                // 在原始字符串中查找并替换这一部分
                                std::string originalPart = "#" + parts[i];
                                std::string correctedPart = "#" + beforeAmpersand + correctedAfterAmpersand;
                                
                                size_t pos = correctedInput.find(originalPart);
                                if (pos != std::string::npos) {
                                    correctedInput.replace(pos, originalPart.length(), correctedPart);
                                }
                            }
                        }
                    }
                }
            }
            
            // 后半部分用第二个车号修正
            std::string secondTrainNumber = trainNumbers_[1];
            for (size_t i = midPoint; i < parts.size(); ++i) {
                if (parts[i].find("CR") != std::string::npos) {
                    size_t ampersandPos = parts[i].find('&');
                    if (ampersandPos != std::string::npos) {
                        std::string beforeAmpersand = parts[i].substr(0, ampersandPos + 1);
                        std::string afterAmpersand = parts[i].substr(ampersandPos + 1);
                        
                        // 提取数字部分
                        std::string numPart = extractNumbers(afterAmpersand);
                        if (!numPart.empty()) {
                            // 检查数字是否异常（长度超过4位或者不包含车号）
                            if (numPart.length() > 4 || numPart.find(secondTrainNumber) == std::string::npos) {
                                // 替换为正确的车号
                                // 替换为正确的车号
                                std::string correctedAfterAmpersand = afterAmpersand;
                                size_t numPos = afterAmpersand.find_last_not_of("0123456789");
                                correctedAfterAmpersand = afterAmpersand.substr(0, numPos + 1) + firstTrainNumber;
                                
                                // 在原始字符串中查找并替换这一部分
                                std::string originalPart = "#" + parts[i];
                                std::string correctedPart = "#" + beforeAmpersand + correctedAfterAmpersand;
                                
                                size_t pos = correctedInput.find(originalPart);
                                if (pos != std::string::npos) {
                                    correctedInput.replace(pos, originalPart.length(), correctedPart);
                                }
                            }
                        }
                    }
                }
            }
        } 
        // 将修正后的字符串保存回原始输入
        correctedInput_ = correctedInput; // 保存修正后的字符串到成员变量
    }
    else
    {
        // 非重连车情况
        std::string correctedInput = input;
        std::vector<std::string> parts = splitString(input, '#');
        
        // 只有一个车号
        if (trainNumbers_.size() > 0)
        {
            std::string trainNumber = trainNumbers_[0];
            // 对于每个部分，检查是否包含CR并且数字部分异常
            for (size_t i = 0; i < parts.size(); ++i) {
                if (parts[i].find("CR") != std::string::npos) {
                    size_t ampersandPos = parts[i].find('&');
                    if (ampersandPos != std::string::npos) {
                        std::string beforeAmpersand = parts[i].substr(0, ampersandPos + 1);
                        std::string afterAmpersand = parts[i].substr(ampersandPos + 1);
                        
                        // 提取数字部分
                        std::string numPart = extractNumbers(afterAmpersand);
                        if (!numPart.empty()) {
                            // 检查数字是否异常（长度超过4位或者不包含车号）
                            if (numPart.length() > 4 || numPart != trainNumber) {
                                // 替换为正确的车号
                                std::string correctedAfterAmpersand = afterAmpersand;
                                size_t numPos = afterAmpersand.find_last_not_of("0123456789");
                                correctedAfterAmpersand = afterAmpersand.substr(0, numPos + 1) + trainNumber;
                                
                                // 在原始字符串中查找并替换这一部分
                                std::string originalPart = "#" + parts[i];
                                std::string correctedPart = "#" + beforeAmpersand + correctedAfterAmpersand;
                                
                                size_t pos = correctedInput.find(originalPart);
                                if (pos != std::string::npos) {
                                    correctedInput.replace(pos, originalPart.length(), correctedPart);
                                }
                            }
                        }
                    }
                }
            }
        }
        // 将修正后的字符串保存回原始输入
        correctedInput_ = correctedInput; // 保存修正后的字符串到成员变量
    }
}

std::vector<std::string> TrainParser::splitString(const std::string& str, char delimiter) {
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

std::string TrainParser::extractNumbers(const std::string& str) {
    size_t pos = str.find_last_not_of("0123456789");
    if (pos == std::string::npos) {
        return str;
    }
    else {
        return str.substr(pos + 1);
    }
}

std::string TrainParser::extractTrainNumberPrefix(const std::string& number) {
    if (number.length() >= 4) {
        return number.substr(0, 4);
    }
    return "";
}

std::string TrainParser::getDirection() const {
    return direction_;
}

std::string TrainParser::getTrainNumber() const {
    std::string result;
    for (size_t i = 0; i < trainNumbers_.size(); ++i) {
        result += trainNumbers_[i];
        if (i != trainNumbers_.size() - 1) {
            result += "|";
        }
    }
    return result;
}

bool TrainParser::isReconnected() const {
    return isReconnected_;
}

std::string TrainParser::getCorrectedInput() const {
    return correctedInput_;
}
