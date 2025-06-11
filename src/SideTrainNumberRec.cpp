#include "SideTrainNumberRec.h"
#include "ui_SideTrainNumberRec.h"
#include <QRegularExpression>
#include <QApplication>
#include <algorithm>

SideTrainNumberRec::SideTrainNumberRec(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui_SideTrainNumberRec)
{
    ui->setupUi(this);

    // 连接信号和槽
    connect(ui->btnSelectDir, &QPushButton::clicked, this, &SideTrainNumberRec::onSelectDirClicked);
    connect(ui->btnProcess, &QPushButton::clicked, this, &SideTrainNumberRec::onProcessClicked);
    
    // 创建并初始化ThreadManager
    m_threadManager = std::make_unique<ThreadManager>();
    connect(m_threadManager.get(), &ThreadManager::m_Logs, this, &SideTrainNumberRec::logMessage);
    connect(m_threadManager.get(), &ThreadManager::m_ShowString, this, &SideTrainNumberRec::displaystring);
    connect(m_threadManager.get(), &ThreadManager::m_UpdateProgress, this, &SideTrainNumberRec::updateProgress);
    connect(m_threadManager.get(), &ThreadManager::m_UpdateCurrentGroup, this, &SideTrainNumberRec::updateCurrentGroup);

    m_threadManager->startThreads();
    ui->btnProcess->setEnabled(false);
}

SideTrainNumberRec::~SideTrainNumberRec() 
{
    delete ui;
}

void SideTrainNumberRec::onSelectDirClicked()
{
    QString dirPath = QFileDialog::getExistingDirectory(this, "选择图片文件夹", "", QFileDialog::ShowDirsOnly);
    if (dirPath.isEmpty()) {
        return;
    }
    
    currentImageDir = dirPath;
    ui->lineEditImageDir->setText(currentImageDir);
    ui->btnProcess->setEnabled(true);
    logMessage(QString("已选择文件夹: %1").arg(currentImageDir));
}

void SideTrainNumberRec::onProcessClicked()
{
    if (currentImageDir.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择图片文件夹");
        return;
    }
    
    ui->btnProcess->setEnabled(false);
    ui->btnSelectDir->setEnabled(false);
    ui->lineEditTrainNumber->clear();
    
    QThread* workerThread = new QThread;
    QObject* worker = new QObject;
    worker->moveToThread(workerThread);
    
    connect(workerThread, &QThread::started, worker, [this, worker, workerThread]() {
        bool success = processImages(currentImageDir);
        
        // 在主线程中更新UI
        QMetaObject::invokeMethod(this, [this, success]() {
            ui->btnProcess->setEnabled(true);
            ui->btnSelectDir->setEnabled(true);
            
            if (success) {
                logMessage("图片处理完成");
            } else {
                logMessage("图片处理失败");
            }
        });
        
        worker->deleteLater();
        workerThread->quit();
    });
    
    connect(workerThread, &QThread::finished, workerThread, &QThread::deleteLater);
    
    logMessage("开始处理图片...");
    workerThread->start();
}

void SideTrainNumberRec::logMessage(const QString& message)
{
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    ui->textEditLog->append(QString("[%1] %2").arg(timestamp).arg(message));
    
    // 滚动到最新内容
    QTextCursor cursor = ui->textEditLog->textCursor();
    cursor.movePosition(QTextCursor::End);
    ui->textEditLog->setTextCursor(cursor);
    ui->textEditLog->ensureCursorVisible();
}

void SideTrainNumberRec::preloadImages(const QStringList& imageFiles, int startIdx, int count)
{
    // 确保索引范围有效
    if (startIdx < 0 || startIdx >= imageFiles.size() || count <= 0) {
        return;
    }
    
    // 计算实际要加载的图片数量（不超出范围）
    int endIdx = (std::min)(startIdx + count, static_cast<int>(imageFiles.size()));
    
    // 创建异步任务列表
    std::vector<std::future<void>> futures;
    
    // 启动异步加载任务
    for (int i = startIdx; i < endIdx; ++i) {
        futures.push_back(std::async(std::launch::async, [this, i, &imageFiles]() {
            QString fullPath = currentImageDir + "/" + imageFiles[i];
            cv::Mat img = cv::imread(fullPath.toStdString());
            if (!img.empty()) {
                std::lock_guard<std::mutex> lock(m_cacheMutex);
                m_imageCache[imageFiles[i]] = img;
            }
        }));
    }
    
    // 等待所有加载完成
    for (auto& future : futures) {
        future.wait();
    }
}

bool SideTrainNumberRec::processImages(const QString& dirPath)
{
    QDir dir(dirPath);
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png";
    dir.setNameFilters(filters);
    
    QStringList imageFiles = dir.entryList(filters, QDir::Files);
    if (imageFiles.isEmpty()) {
        QMetaObject::invokeMethod(this, [this]() {
            ui->labelCurrentGroup->setText("当前处理：无图片文件");
            ui->progressBar->setValue(0);
        });
        logMessage("文件夹中没有找到图片文件");
        return false;
    }
    
    // 对文件名进行排序，确保按照正确的顺序处理
    sortImageFiles(imageFiles);
    
    // 计算总组数
    int totalGroups = imageFiles.size() - 2;
    QMetaObject::invokeMethod(this, [this, totalGroups]() {
        ui->progressBar->setRange(0, totalGroups);
        ui->progressBar->setValue(0);
    });
    
    // 清空图片缓存
    {
        std::lock_guard<std::mutex> lock(m_cacheMutex);
        m_imageCache.clear();
    }
    
    // 预加载前6张图片（或全部，如果少于6张）
    int preloadCount = (std::min)(6, imageFiles.size());
    preloadImages(imageFiles, 0, preloadCount);
    
    manualTask task;
    task.timestamp = QDateTime::currentDateTime().toString("yyyyMMddhhmmss").toStdString();
    
    for (int i = 0; i <= imageFiles.size() - 3; i++) {
        if (!this->isVisible()) {
            return false;
        }
        
        // 当处理到第i组时，预加载i+6组的图片（如果存在）
        if (i + 6 < imageFiles.size()) {
            // 异步预加载，不等待完成
            std::thread([this, i, &imageFiles]() {
                preloadImages(imageFiles, i + 6, 3);
            }).detach();
        }
        
        if (i == 0) {
            task.flag = 0; // 开始
        } else if (i == imageFiles.size() - 3) {
            task.flag = 2; // 中间
        } else {
            task.flag = 1;// 结束
        }
        
        // 更新进度信息
        QMetaObject::invokeMethod(this, [this, i, totalGroups, &imageFiles]() {
            ui->labelCurrentGroup->setText(QString("当前处理：%1, %2, %3").arg(imageFiles[i]).arg(imageFiles[i + 1]).arg(imageFiles[i + 2]));
            ui->progressBar->setValue(i);
        });

        // 从缓存加载图片，如果缓存中没有则直接加载
        cv::Mat image1, image2, image3;
        {
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            
            // 尝试从缓存获取图片1
            auto it1 = m_imageCache.find(imageFiles[i]);
            if (it1 != m_imageCache.end()) {
                image1 = it1->second.clone(); // 使用clone避免多线程访问同一数据
                m_imageCache.erase(it1); // 使用后从缓存移除以节省内存
            } else {
                QString fullPath = currentImageDir + "/" + imageFiles[i];
                image1 = cv::imread(fullPath.toStdString());
            }
            
            // 尝试从缓存获取图片2
            auto it2 = m_imageCache.find(imageFiles[i + 1]);
            if (it2 != m_imageCache.end()) {
                image2 = it2->second.clone();
                m_imageCache.erase(it2);
            } else {
                QString fullPath = currentImageDir + "/" + imageFiles[i + 1];
                image2 = cv::imread(fullPath.toStdString());
            }
            
            // 尝试从缓存获取图片3
            auto it3 = m_imageCache.find(imageFiles[i + 2]);
            if (it3 != m_imageCache.end()) {
                image3 = it3->second.clone();
                m_imageCache.erase(it3);
            } else {
                QString fullPath = currentImageDir + "/" + imageFiles[i + 2];
                image3 = cv::imread(fullPath.toStdString());
            }
        }

        if (image1.empty() || image2.empty() || image3.empty()) {
            logMessage(QString("无法加载图片组，跳过处理"));
            continue;
        }
        
        cv::Size final_target_size = cv::Size(1200, 1200);
        int target_height = static_cast<int>(image1.rows / 2.0);
        int target_width_per_img = final_target_size.width / 3;
   
        cv::Mat final_image(final_target_size, image1.type());
   
        cv::Rect roi1(0, 0, target_width_per_img, final_target_size.height);
        cv::Rect roi2(target_width_per_img, 0, target_width_per_img, final_target_size.height);
        cv::Rect roi3(2 * target_width_per_img, 0, target_width_per_img, final_target_size.height);
   
        cv::resize(image1, final_image(roi1), roi1.size(), 0, 0, cv::INTER_AREA);
        cv::resize(image2, final_image(roi2), roi2.size(), 0, 0, cv::INTER_AREA);
        cv::resize(image3, final_image(roi3), roi3.size(), 0, 0, cv::INTER_AREA);

        // 把图像高度去除一般保留下半部分图片
        int half_height = final_image.rows / 2;
        cv::Rect roi(0, half_height, final_image.cols, half_height);
        cv::Mat cropped_image = final_image(roi);

        // 设置任务数据
        task.image = cropped_image;
        task.imageSequenceNumber = imageFiles[i].toStdString();

        {
            std::lock_guard<std::mutex> lock(*m_threadManager->m_mtx_picProcess);
            ThreadManager::StitchedImageData data;
            data.image = task.image;
            data.imageSequenceNumber = task.imageSequenceNumber;
            data.timestamp = task.timestamp;
            data.flag = task.flag;
            
            m_threadManager->m_queue_picProcess.push(data);
            logMessage(QString("已提交图片组 %1 到处理队列").arg(task.imageSequenceNumber.c_str()));
        }

        QThread::msleep(10);
    }
    
    // 更新最终进度
    QMetaObject::invokeMethod(this, [this, totalGroups]() {
        ui->labelCurrentGroup->setText("处理完成");
        ui->progressBar->setValue(totalGroups);
    });
    
    return true;
}

void SideTrainNumberRec::sortImageFiles(QStringList& imageFiles)
{
    // 使用正则表达式提取文件名中的数字部分进行排序
    // 假设文件名格式为：105-000-x.jpg, 105-001-x.jpg 等
    QRegularExpression re("(\\d+)-(\\d+)-");
    
    std::sort(imageFiles.begin(), imageFiles.end(), [&re](const QString& a, const QString& b) {
        QRegularExpressionMatch matchA = re.match(a);
        QRegularExpressionMatch matchB = re.match(b);
        
        if (matchA.hasMatch() && matchB.hasMatch()) {
            // 首先比较第一个数字部分
            int numA1 = matchA.captured(1).toInt();
            int numB1 = matchB.captured(1).toInt();
            
            if (numA1 != numB1) {
                return numA1 < numB1;
            }
            
            // 如果第一个数字相同，比较第二个数字部分
            int numA2 = matchA.captured(2).toInt();
            int numB2 = matchB.captured(2).toInt();
            return numA2 < numB2;
        }
        
        // 如果不匹配正则表达式，按字母顺序排序
        return a < b;
    });
}

void SideTrainNumberRec::displaystring(const std::string& str)
{
    ui->lineEditTrainNumber->setText(QString::fromStdString(str));
}

void SideTrainNumberRec::updateProgress(int current, int total)
{
    ui->progressBar->setRange(0, total);
    ui->progressBar->setValue(current);
}

void SideTrainNumberRec::updateCurrentGroup(const QString& groupInfo)
{
    ui->labelCurrentGroup->setText(groupInfo);
}


