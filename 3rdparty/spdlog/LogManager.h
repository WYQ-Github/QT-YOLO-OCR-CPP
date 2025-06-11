#pragma once

#include <memory>
#include <string>
#include "log.hpp"

class LogManager {
private:
    static std::shared_ptr<LogManager> instance_;
    std::shared_ptr<Logger> logger_;

    LogManager(const std::string& filename, spdlog::level::level_enum defaultLevel);

public:
    LogManager(const LogManager&) = delete;
    void operator=(const LogManager&) = delete;

    ~LogManager();

    static std::shared_ptr<LogManager> getInstance(const std::string& filename, spdlog::level::level_enum defaultLevel);

    std::shared_ptr<Logger> getLogger();
};