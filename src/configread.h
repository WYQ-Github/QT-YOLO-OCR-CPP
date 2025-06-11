#ifndef CONFIGREAD_H
#define CONFIGREAD_H

#include "header.h"

class ConfigRead{

public:
    ConfigRead();
    ~ConfigRead();

    bool ReadConfig(std::string path, GlobalParam& globalParam, UdpToolParam& udpToolParam, AlgorithmParam& algParam);

private:
    CSimpleIniA m_ini;
};

#endif // CONFIGREAD_H
