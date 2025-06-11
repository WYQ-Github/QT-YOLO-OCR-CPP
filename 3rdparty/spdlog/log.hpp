#include <memory>
#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>

class Logger {
private:
    std::shared_ptr<spdlog::logger> logger_;
    spdlog::level::level_enum current_level_;

public:
    Logger(const std::string& filename, spdlog::level::level_enum defaultLevel) {
        auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(filename, 0, 0); // 创建 daily_file_sink_mt
        logger_ = std::make_shared<spdlog::logger>("log", daily_sink);
        current_level_ = defaultLevel;
        spdlog::set_default_logger(logger_);
        spdlog::set_level(defaultLevel);
    }

    void setLogLevel(spdlog::level::level_enum logLevel) {
        current_level_ = logLevel;
        spdlog::set_level(logLevel);
    }

    void logInfo(const std::string& message, bool isDebug) {
        if (isDebug && current_level_ != spdlog::level::debug) {
            return;
        }
        spdlog::info(message);
        spdlog::default_logger()->flush(); // 手动调用 flush 方法
    }

    void logWarn(const std::string& message, bool isDebug) {
        if (isDebug && current_level_ != spdlog::level::debug) {
            return;
        }
        spdlog::warn(message);
        spdlog::default_logger()->flush(); // 手动调用 flush 方法
    }

    void logError(const std::string& message, bool isDebug) {
        if (isDebug && current_level_ != spdlog::level::debug) {
            return;
        }
        spdlog::error(message);
        spdlog::default_logger()->flush(); // 手动调用 flush 方法
    }
};