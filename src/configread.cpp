#include "configread.h"

ConfigRead::ConfigRead() : m_ini(true, false, false) {}

ConfigRead::~ConfigRead() {}

bool ConfigRead::ReadConfig(std::string path, GlobalParam& globalParam, UdpToolParam& udpToolParam, AlgorithmParam& algParam) {

    SI_Error rc = m_ini.LoadFile(path.c_str());
    if (rc < 0) {
        return false;
    }

    // 辅助函数：读取INI值并转换为指定类型
    auto ReadIniValue = [&](const std::string& section, const std::string& key, auto& value) -> bool {
        const char* curr_temp = m_ini.GetValue(section.c_str(), key.c_str());
        if (!curr_temp) return false;
        try {
            if constexpr (std::is_same_v<decltype(value), std::string&>) {
                value = curr_temp;
            } else if constexpr (std::is_same_v<decltype(value), int&>) {
                value = std::stoi(curr_temp);
            }else if constexpr (std::is_same_v<decltype(value), float&>) {
                value = std::stof(curr_temp);
            } else if constexpr (std::is_same_v<decltype(value), double&>) {
                value = std::stod(curr_temp);
            } else if constexpr (std::is_same_v<decltype(value), bool&>) {
                value = (strcmp(curr_temp, "true") == 0 || strcmp(curr_temp, "1") == 0);
            }
            else {
                return false;
            }
            return true;
        } catch (...) {
            return false; 
        }
    };

    // 辅助函数：读取字符串到字符数组
    auto ReadIniStringToArray = [&](const std::string& section, const std::string& key, char* dest, size_t dest_size) -> bool {
        const char* value = m_ini.GetValue(section.c_str(), key.c_str());
        if (!value) return false;
        strncpy_s(dest, dest_size, value, _TRUNCATE);
        return true;
    };

    // UDP配置
    const std::string udpSection = "UDPToolsParam";
    if (!ReadIniStringToArray(udpSection, "ListenIP", udpToolParam.listen_ip, sizeof(udpToolParam.listen_ip)) ||
        !ReadIniStringToArray(udpSection, "SendIP", udpToolParam.send_ip, sizeof(udpToolParam.send_ip)) ||
        !ReadIniValue(udpSection, "ListenPort", udpToolParam.listen_port) ||
        !ReadIniValue(udpSection, "SendPort", udpToolParam.send_port)) 
        {
        return false;
    }

    // 全局参数配置
    const std::string globalSection = "GlobalParam";
    if(!ReadIniValue(globalSection, "ModelPath", globalParam.modelPath) || 
       !ReadIniValue(globalSection, "ImagePath", globalParam.imagePath) ||
        !ReadIniValue(globalSection, "OCRRecPath", globalParam.OCRRecPath) ||
        !ReadIniValue(globalSection, "OCRDetPath", globalParam.OCRDetPath) ||
        !ReadIniValue(globalSection, "OCRClsPath", globalParam.OCRClsPath) ||
        !ReadIniValue(globalSection, "HeightReductionFactor", globalParam.factor) ||
        !ReadIniValue(globalSection, "ResizeWidth", globalParam.resizeWidth) || 
        !ReadIniValue(globalSection, "ReiszeHeight", globalParam.reiszeHeight) ||
        !ReadIniValue(globalSection, "Dictionary", globalParam.dictPath) ||
        !ReadIniValue(globalSection, "RecognitionMode", globalParam.recMode) ||
        !ReadIniValue(globalSection, "YOLOPath", globalParam.YOLOPath) ||
        !ReadIniValue(globalSection, "isSave", globalParam.isSave)   ||
        !ReadIniValue(globalSection, "SavePath", globalParam.savePath)
    ) {
        return false;
    }


    // 算法参数配置
    const std::string algorithmParam = "AlgorithmParam";
    if(!ReadIniValue(algorithmParam, "MAX_EMPTY_FRAMES", algParam.max_empty_frames) ||
       !ReadIniValue(algorithmParam, "MIN_LENGTH", algParam.min_length) ||
       !ReadIniValue(algorithmParam, "TRAIN_TYPE", algParam.trian_type)) {
        return false;
    }

    return true;
}







