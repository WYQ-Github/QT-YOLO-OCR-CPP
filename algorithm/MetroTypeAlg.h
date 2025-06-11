#ifndef METROTYPEALG_H
#define METROTYPEALG_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iostream>

class MetroTrainParser {
public:
    const int TRAIN_NUM = 6;                        // 地铁车厢数
    /**
     * @brief 解析输入的原始车号字符串
     * @param input 原始输入, 例如 "#1&02146#2&02145#3&02144#4&02143#5&02142#6&02141"
     */
    void parse(const std::string& input);

    /**
     * @brief 获取行驶方向
     * @return "1" 代表正向 (车厢号递增), "2" 代表反向 (车厢号递减)
     */
    std::string getDirection() const;

    /**
     * @brief 获取列车主车号 (不包含车厢序号)
     * @return 例如 "0214" 或 "12014"
     */
    std::string getTrainNumber() const;

    /**
     * @brief 获取根据算法修正后的完整车号字符串
     * @return 修正后的标准格式字符串
     */
    std::string getCorrectedInput() const;

private:   
    // 工具函数：按指定分隔符切分字符串
    std::vector<std::string> splitString(const std::string& str, char delimiter);

    // 私有成员变量，存储解析结果
    std::string direction_;                 // 存储方向
    std::string trainNumber_;               // 存储主车号
    std::vector<std::string> trainNumbers_; // 存储从输入中解析出的原始车号列表
    std::string correctedInput_;            // 存储修正后的完整字符串
};
#endif // METROTYPEALG_H