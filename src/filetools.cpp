#include "filetools.h"
#include <filesystem>
#include <regex>
#include <map>
#include <algorithm>

std::string FileTools::GetExePath() {
    char szFilePath[MAX_PATH + 1] = { 0 };
    GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
    (strrchr(szFilePath, '\\'))[0] = 0;
    return std::string(szFilePath);
}

bool FileTools::IsFileExist(const std::string& filePath) {
    return std::filesystem::exists(filePath);
}

bool FileTools::CreateDirectory(const std::string& dirPath) {
    try {
        return std::filesystem::create_directories(dirPath);
    }
    catch (const std::filesystem::filesystem_error&) {
        return false;
    }
}

std::string FileTools::GetFileNameWithoutExtension(const std::string& filePath) {
    std::filesystem::path path(filePath);
    return path.stem().string();
}

std::string FileTools::GetFileExtension(const std::string& filePath) {
    std::filesystem::path path(filePath);
    return path.extension().string();
}


std::vector<std::string>  FileTools::split_string(const std::string& content, const std::string& delimiter) {

    std::vector<std::string> result;

    std::string::size_type pos1 = 0;
    std::string::size_type pos2 = content.find(delimiter);

    while (std::string::npos != pos2)
    {
        result.push_back(content.substr(pos1, pos2 - pos1));

        pos1 = pos2 + delimiter.size();
        pos2 = content.find(delimiter, pos1);
    }

    if (pos1 != content.length())
    {
        result.push_back(content.substr(pos1));
    }

    return result;
}

std::string FileTools::get_filename(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    return (std::string::npos == pos) ? path : path.substr(pos + 1);
}


std::string FileTools::get_extension(const std::string &filename)
{
    size_t pos = filename.find_last_of('.');
    return (std::string::npos == pos) ? "" : filename.substr(pos);
}

std::string FileTools::get_stem(const std::string &filename)
{
    size_t pos = filename.find_last_of('.');
    return (std::string::npos == pos) ? filename : filename.substr(0, pos);
}
