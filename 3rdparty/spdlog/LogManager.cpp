#include "LogManager.h"

std::shared_ptr<LogManager> LogManager::instance_;  // 静态数据成员的定义

LogManager::LogManager(const std::string& filename, spdlog::level::level_enum defaultLevel) {
    logger_ = std::make_shared<Logger>(filename, defaultLevel);
}

LogManager::~LogManager() {
    // 进行必要的清理工作，如果有的话
}

std::shared_ptr<LogManager> LogManager::getInstance(const std::string& filename, spdlog::level::level_enum defaultLevel) {
    if (!instance_) {
        instance_ = std::shared_ptr<LogManager>(new LogManager(filename, defaultLevel));
    }
    return instance_;
}

std::shared_ptr<Logger> LogManager::getLogger() {
    return logger_;
}