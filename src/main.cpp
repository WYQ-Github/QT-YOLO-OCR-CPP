#include "SideTrainNumberRec.h"
#include <QApplication>
#include <QMetaType>
#pragma comment(lib, "user32.lib")
int main(int argc, char *argv[])
{
    qRegisterMetaType<std::string>("std::string");
    QApplication a(argc, argv);
    SideTrainNumberRec w;
    w.show();
    return a.exec();
}