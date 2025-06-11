#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>
#include <map>
#include <regex>
#include <filesystem>
#include <unordered_set>

class FileTools {
public:
    // 单例模式
    static FileTools& getInstance() {
        static FileTools instance;
        return instance;
    }

    // 获取程序运行路径
    std::string GetExePath();

    // 检查文件是否存在
    bool IsFileExist(const std::string& filePath);

    // 创建目录
    bool CreateDirectory(const std::string& dirPath);

    // 获取文件名（不含扩展名）
    std::string GetFileNameWithoutExtension(const std::string& filePath);

    // 获取文件扩展名
    std::string GetFileExtension(const std::string& filePath);

    // 解析字符串
    std::vector<std::string> split_string(const std::string& content, const std::string& delimiter);

    std::string get_filename(const std::string &path);

    std::string get_extension(const std::string &filename);

    std::string get_stem(const std::string &filename);


    // 拼接字符串
    template <typename... Args>
    inline std::string str_format(const std::string &format, Args... args) {
        auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
        std::unique_ptr<char[]> buf(new (std::nothrow) char[size_buf]);

        if (!buf)
            return std::string(""); // 内存分配失败则返回空字符串

        std::snprintf(buf.get(), size_buf, format.c_str(), args...);
        return std::string(buf.get(), buf.get() + size_buf - 1);
    }


private:
    FileTools() = default;  // 私有构造函数
    ~FileTools() = default;
    
    // 禁止拷贝和赋值
    FileTools(const FileTools&) = delete;
    FileTools& operator=(const FileTools&) = delete;
};









