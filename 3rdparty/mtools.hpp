#pragma once
#include <string>
#include <vector>
#include <sstream> // Added for ostringstream
#include <fstream>
#include <random>
#include <chrono>
// #include <format> // Removed C++20 header
#include <memory>
#include <iostream> // Added for cout, cerr
#include <iomanip>  // Added for std::put_time
#include <unordered_map>
#include <type_traits>
#include <opencv2/opencv.hpp>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <libgen.h>
    #include <unistd.h>
#endif


namespace MT{
    // 获取文件名
    inline std::string GetFileName(const std::string& path){
        size_t pos = path.find_last_of("/\\");
        return (pos == std::string::npos) ? path : path.substr(pos + 1);
    }
    // 获取文件后缀
    inline std::string GetFileExten(const std::string& path) {
        size_t pos = path.rfind('.');
        return (pos == std::string::npos) ? "" : path.substr(pos + 1);
    }
    // 相对路径转为绝对路径
    inline std::string GetFileAbsPath(const std::string& path){
        std::string absPath;
        #ifdef _WIN32
            // Windows平台使用GetFullPathName
            char buffer[1024];
            DWORD result = GetFullPathName(path.c_str(), 1024, buffer, nullptr);
            if (result > 0 && result < 1024) {
                absPath = buffer;
            } else {
                // 处理错误情况
                std::cerr << "GetFullPathName failed" << std::endl;
            }
        #else
            // POSIX平台使用realpath
            char buffer[1024];
            if (realpath(path.c_str(), buffer) != nullptr) {
                absPath = buffer;
            } else {
                // Consider throwing an exception or returning an empty string on failure
                // throw std::runtime_error("realpath failed");
                std::cerr << "realpath failed for path: " << path << std::endl;
            }
        #endif
        return absPath;
    }
    // 文件是否存在
    inline bool FileExists(const std::string& path){
        std::ifstream file(path.c_str());
        return file.good();
    }
    inline bool FileExists(const char* path){
        std::ifstream file(path);
        return file.good();
    }
    // 分割字符串
    inline std::vector<std::string> SplitString(const std::string& src,const char gap = ' '){
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream iss(src);
        while (std::getline(iss, token, gap)) {
            tokens.push_back(token);
        }
        return tokens; // Removed std::move for clarity, RVO should handle it
    }
    // 合并字符串
    inline std::string JoinStrings(const std::vector<std::string>& strings,const char gap = ' '){
        std::string result;
        bool first = true;
        for(const auto &str:strings){
            if (!first) {
                result += gap;
            }
            result += str;
            first = false;
        }
        return result;
    }
    // 字符串是否以start开头
    inline bool StartsWith(const std::string& src, const std::string& start) {return src.rfind(start, 0) == 0;} // More common way
    // 字符串是否以end结尾
    inline bool EndsWith(const std::string& src, const std::string& end) {
        if (src.length() >= end.length()) {
            return (0 == src.compare (src.length() - end.length(), end.length(), end));
        } else {
            return false;
        }
    }
    //替换文件后缀
    inline std::string ReplaceExten(const std::string& src,const std::string& exten){
        size_t pos = src.rfind('.');
        std::ostringstream ss;
        if(pos == std::string::npos){
            ss << src << exten;
        } else {
            ss << src.substr(0,pos) << exten;
        }
        return ss.str();
    }
    // 获取一个[min,max]范围内的随机浮点数字，均匀采样
    template<typename T>
    inline T GetRandom(T min, T max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        if constexpr (std::is_integral<T>::value) {
            std::uniform_int_distribution<T> dis(min, max);
            return dis(gen);
        } else if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(min, max);
            return dis(gen);
        } else {
            static_assert(std::is_arithmetic<T>::value, "GetRandom only supports arithmetic types");
            return T{}; // Should not reach here due to static_assert
        }
    }
    // 获取当前时间字符串
    inline std::string GetCurrentTime(){
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm buf{}; // Value-initialize
    #if defined(__unix__) || defined(__APPLE__)
        localtime_r(&in_time_t, &buf);
    #elif defined(_MSC_VER)
        localtime_s(&buf, &in_time_t);
    #else
        // Using static global buffer is not thread-safe, but std::localtime is what's available
        // std::tm *lt = std::localtime(&in_time_t);
        // if (lt) buf = *lt; else { /* handle error, e.g., return "Error:localtime failed" */ }
        // For C++17, it's better to rely on the above _MSC_VER or POSIX paths.
        // If neither, this part might need adjustment based on specific compiler/platform.
        // For now, let's assume one of the above will be met.
        // static_assert(false, "Platform not supported for localtime_s/localtime_r");
        // As a fallback, though not ideal due to thread-safety:
        std::tm* temp_tm = std::localtime(&in_time_t);
        if (temp_tm != nullptr) {
            buf = *temp_tm;
        } else {
            // Handle error, e.g. by returning an error string or throwing
            return "TimeConversionError";
        }
    #endif
        std::ostringstream ss;
        ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S"); // Format: YYYY-MM-DD HH:MM:SS
        return ss.str();
    }
    // 获取当前日期
    inline std::string GetCurrentDate() {
        auto now = std::chrono::system_clock::now();
        std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
        std::tm localTime{}; // Value-initialize
    #if defined(_MSC_VER)
        localtime_s(&localTime, &timeNow);
    #elif defined(__unix__) || defined(__APPLE__)
        localtime_r(&timeNow, &localTime);
    #else
        std::tm* temp_tm = std::localtime(&timeNow);
        if (temp_tm != nullptr) {
            localTime = *temp_tm;
        } else {
            return "DateConversionError";
        }
    #endif
        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y-%m-%d");
        return oss.str();
    }
    // 获取当前时间戳,按毫秒算
    inline uint64_t GetCurrentTimestamp(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        return static_cast<uint64_t>(milliseconds.count());
    }
    inline size_t GetStrNumber(const std::string& str) {
        size_t count = 0;
        for (size_t i = 0; i < str.size(); ) {
            unsigned char c = static_cast<unsigned char>(str[i]);
            if ((c & 0x80) == 0)  // 单字节字符 (ASCII)
                ++i;
            else if ((c & 0xE0) == 0xC0) // 双字节字符
                i += 2;
            else if ((c & 0xF0) == 0xE0) // 三字节字符
                i += 3;
            else if ((c & 0xF8) == 0xF0) // 四字节字符
                i += 4;
            else 
                ++i; // Invalid UTF-8, skip one byte
            ++count;
        }
        return count;
    }
    // 根据逻辑字符的起始和结束位置获取子字符串
    inline std::string GetSubStr(const std::string& str, size_t start, size_t stop) {
        size_t num_chars = GetStrNumber(str);
        if (start > stop || stop > num_chars) {
            throw std::out_of_range("Invalid start or stop index in GetSubStr");
        }
            
        size_t current_char_index = 0;
        size_t start_byte_index = 0;
        bool start_found = false;

        for (size_t i = 0; i < str.size(); ) {
            if (current_char_index == start) {
                start_byte_index = i;
                start_found = true;
                break;
            }
            unsigned char c = static_cast<unsigned char>(str[i]);
            if ((c & 0x80) == 0) { ++i; }
            else if ((c & 0xE0) == 0xC0) { i += 2; }
            else if ((c & 0xF0) == 0xE0) { i += 3; }
            else if ((c & 0xF8) == 0xF0) { i += 4; }
            else { ++i; } // Invalid UTF-8
            ++current_char_index;
        }
        if (!start_found && start == num_chars) { // If start is at the end
             start_byte_index = str.size();
        } else if (!start_found) {
            throw std::out_of_range("Start index not found in GetSubStr");
        }

        current_char_index = 0;
        size_t end_byte_index = str.size(); // Default to end of string if stop == num_chars
        bool end_found = false;
        if (start == stop) return ""; // Empty substring if start == stop

        for (size_t i = 0; i < str.size(); ) {
            if (current_char_index == stop) {
                end_byte_index = i;
                end_found = true;
                break;
            }
            unsigned char c = static_cast<unsigned char>(str[i]);
            if ((c & 0x80) == 0) { ++i; }
            else if ((c & 0xE0) == 0xC0) { i += 2; }
            else if ((c & 0xF0) == 0xE0) { i += 3; }
            else if ((c & 0xF8) == 0xF0) { i += 4; }
            else { ++i; } // Invalid UTF-8
            ++current_char_index;
        }
         if (!end_found && stop != num_chars) { // If stop is not end of string and not found
            throw std::out_of_range("Stop index not found in GetSubStr");
        }


        return str.substr(start_byte_index, end_byte_index - start_byte_index);
    }

    //************************************************************************
    // 图片预处理:将 图片填充至指定大小，并缩放至指定大小
    inline cv::Mat PaddingImg(cv::Mat& img,cv::Size target){
        int top=0,bottom=0,left=0,right=0;
        auto ori_w = static_cast<float>(img.cols);
        auto ori_h = static_cast<float>(img.rows);
        auto net_w = static_cast<float>(target.width);
        auto net_h = static_cast<float>(target.height);
        auto r = (std::min)(net_w/ori_w,net_h/ori_h);//缩放比例
        auto width = static_cast<int>(ori_w*r);
        auto height = static_cast<int>(ori_h*r);
        top = static_cast<int>((net_h - height) / 2);
        bottom = static_cast<int>((net_h - height) / 2 + static_cast<int>(net_h - height) % 2);
        left = static_cast<int>((net_w - width) / 2);
        right = static_cast<int>((net_w - width) / 2 + static_cast<int>(net_w - width) % 2);
        cv::Mat result_img;
        cv::resize(img,result_img,cv::Size(width,height));
        cv::copyMakeBorder(result_img, result_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
        return result_img;
    }
    inline cv::Mat UnpaddingImg(cv::Mat& img,cv::Size target,cv::Size ori_size){
        auto ori_w = static_cast<float>(ori_size.width);
        auto ori_h = static_cast<float>(ori_size.height);
        auto net_w = static_cast<float>(target.width);
        auto net_h = static_cast<float>(target.height);
        auto r = (std::min)(net_w/ori_w,net_h/ori_h);//缩放比例
        auto width = static_cast<int>(ori_w*r);
        auto height = static_cast<int>(ori_h*r);
        int top = static_cast<int>((net_h - height) / 2);
        int left = static_cast<int>((net_w - width) / 2);
        cv::Mat result_img = img(cv::Rect(left, top, width, height)).clone();
        cv::resize(result_img,result_img,ori_size);
        std::cout << "[" << top << "," << left << "," << width << "," << height << "]" << std::endl; // Replaced std::println
        return result_img;
    }
    inline float ComputeIou(const cv::Rect& a, const cv::Rect& b) {
        // 检查输入有效性（可选）
        assert(a.width >= 0 && a.height >= 0 && b.width >= 0 && b.height >= 0);
        // 计算交集区域
        int x1 = (std::max)(a.x, b.x);
        int y1 = (std::max)(a.y, b.y);
        int x2 =  (std::min)(a.x + a.width, b.x + b.width);
        int y2 =  (std::min)(a.y + a.height, b.y + b.height);
        int interWidth = (std::max)(0, x2 - x1);
        int interHeight = (std::max)(0, y2 - y1);
        // 计算面积（避免整数溢出）
        float intersection = static_cast<float>(interWidth) * interHeight;
        float areaA = static_cast<float>(a.width) * a.height;
        float areaB = static_cast<float>(b.width) * b.height;
        float unionArea = areaA + areaB - intersection;
        // 处理除零情况
        return unionArea > 0.0f ? intersection / unionArea : 0.0f;
    }
    
    //************************************************************************
    class OCRDictionary {
    public:
        OCRDictionary(){}
        OCRDictionary(const std::string& dict_path) {
            this->load(dict_path);
        }
        OCRDictionary(const char* dict_path) {
            if (dict_path) { // Check for nullptr
                std::string dict_path_str(dict_path);
                this->load(dict_path_str);
            } else {
                 std::cerr << "Error: Dictionary path is null." << std::endl;
            }
        }
        // 加载字典
        bool load(const std::string& dict_path) {
            std::ifstream file(dict_path);
            if (!file.is_open()) {
                std::cerr << "无法打开字典文件: " << dict_path << std::endl;
                return false;
            }
            std::string line;
            dictionary_.clear(); // Clear previous data if any
            char_to_index_.clear(); // Clear previous data if any
            while (std::getline(file, line)) {
                // Optional: Add UTF-8 BOM removal if necessary
                // if (line.rfind("\xEF\xBB\xBF", 0) == 0) { line.erase(0, 3); }
                if (!line.empty()) {
                    dictionary_.push_back(line);
                    char_to_index_[line] = static_cast<int>(dictionary_.size() - 1);
                }
            }
            file.close();
            // std::cout << "字典加载完成，字符数量: " << dictionary_.size() << std::endl;
            return true;
        }
        // 通过索引获取字符
        std::string get_char(int index) const {
            if (index < 0 || static_cast<size_t>(index) >= dictionary_.size()) {
                return " "; // 返回空字符串表示索引无效
            }
            return dictionary_[static_cast<size_t>(index)];
        }

        // 通过字符获取索引
        int get_index(const std::string& ch) const {
            auto it = char_to_index_.find(ch);
            if (it != char_to_index_.end()) {
                return it->second;
            }
            return -1; // -1 表示未找到
        }
        // 获取字典大小
        size_t size() const {
            return dictionary_.size();
        }
    private:
        std::vector<std::string> dictionary_;          // 字符列表
        std::unordered_map<std::string, int> char_to_index_; // 反向查找表
    };
} // namespace MT