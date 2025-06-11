#pragma once

#include "ui_SideTrainNumberRec.h"
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QFileInfoList>
#include <QImage>
#include <QPixmap>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <queue>
#include <mutex>
#include <future>
#include <unordered_map>
#include "ThreadManager.h"

class SideTrainNumberRec : public QMainWindow {
    Q_OBJECT
    
public:
    SideTrainNumberRec(QWidget* parent = nullptr);
    ~SideTrainNumberRec();

public slots:
    void logMessage(const QString& message);
    void displaystring(const std::string& str);
    void updateProgress(int current, int total);
    void updateCurrentGroup(const QString& groupInfo);

private slots:
    void onSelectDirClicked();
    void onProcessClicked();

private:
    // 异步预加载图片
    void preloadImages(const QStringList& imageFiles, int startIdx, int count);
    bool processImages(const QString& dirPath);
    void sortImageFiles(QStringList& imageFiles);

    struct manualTask{
        int flag = 0; 
        cv::Mat image;
        std::string imageSequenceNumber;
        std::string timestamp;
    };

private:
    std::unique_ptr<ThreadManager> m_threadManager;
    std::unordered_map<QString, cv::Mat> m_imageCache; // 图片缓存
    std::mutex m_cacheMutex; // 缓存互斥锁

private:
    Ui_SideTrainNumberRec* ui;
    QString currentImageDir;
    QString recognizedNumber;
};