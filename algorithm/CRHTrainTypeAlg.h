#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iostream>

class TrainParser {
public:
    void parse(const std::string& input);
    std::string getDirection() const;
    std::string getTrainNumber() const;
    bool isReconnected() const;
    std::string getCorrectedInput() const;
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter);
    std::string extractNumbers(const std::string& str);
    std::string extractTrainNumberPrefix(const std::string& number);
    std::string direction_;
    std::vector<std::string> trainNumbers_;
    bool isReconnected_;
    std::string correctedInput_;
};

