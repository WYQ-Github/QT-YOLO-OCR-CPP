#include "ThreadManager.h"
ThreadManager::ThreadManager(QObject *parent)
    : QObject(parent)
    , threadStop(false)
    , m_mtx_udpTool(std::make_shared<std::mutex>())
    , m_mtx_udpProcess(std::make_shared<std::mutex>())
    , m_mtx_picProcess(std::make_shared<std::mutex>())
    , m_trainParser(std::make_unique<TrainParser>())
    , m_MetrotrainParser(std::make_unique<MetroTrainParser>())
    , m_trainNumberDetector(std::make_unique<TrainNumberDetector>())
{

    // 读取参数
    std::string exePath = FileTools::getInstance().GetExePath();
    m_ConfigRead = std::make_unique<ConfigRead>();
    m_ConfigRead->ReadConfig(exePath + "\\Config.ini", m_GlobalParam, m_udpToolParam, m_AlgParam);

    // 配置算法参数
    m_trainNumberDetector->MAX_EMPTY_FRAMES = m_AlgParam.max_empty_frames;
    m_trainNumberDetector->MIN_LENGTH = m_AlgParam.min_length;
    m_trainNumberDetector->TRAIN_TYPE = m_AlgParam.trian_type;

    // 创建UDP工具
    m_udpTool = std::make_unique<UdpTool>();
    m_udpTool->CreateSocket(m_udpToolParam, true);

    // 创建目标检测器
    deploy::InferOption option;
    option.enableSwapRB();

    // 创建模型实例
    if (m_GlobalParam.recMode == 0) {
        m_logger->logInfo(fmt::format("使用YOLO模式识别"), false);
        emit m_Logs(QString("使用YOLO模式识别"));
        m_detector = std::make_unique<deploy::DetectModel>(m_GlobalParam.YOLOPath, option);
    }
    else if (m_GlobalParam.recMode == 1) {
        m_logger->logInfo(fmt::format("使用OCR模式识别"), false);
        emit m_Logs(QString("使用OCR模式识别"));
        m_detector = std::make_unique<deploy::DetectModel>(m_GlobalParam.modelPath, option);
        m_paddleOcr = std::make_unique<PaddleOCR>();
        std::vector<std::string> onnx_paths{m_GlobalParam.OCRDetPath, m_GlobalParam.OCRClsPath, m_GlobalParam.OCRRecPath};
        std::variant<bool, std::string> init_status = m_paddleOcr->initialize(onnx_paths, true);
        if (init_status.index() == 1) { 
            std::string error_message = std::get<std::string>(init_status);
            m_logger->logError(fmt::format("OCR模型初始化失败: {}", error_message), false);
            return; 
        }
        m_logger->logInfo(fmt::format("OCR 引擎初始化成功!"), false);
        m_ParamsOCR.repeat = false;
        m_ParamsOCR.min_area = 100;
        m_ParamsOCR.text = 0.25f;
        m_ParamsOCR.thresh = 0.25f;
        m_ParamsOCR.unclip_ratio = 2.5f;
        m_ParamsOCR.dictionary = m_GlobalParam.dictPath.c_str(); ;
        if (m_paddleOcr->setparms(m_ParamsOCR) == 0) {
            m_logger->logError("OCR 字典文件加载失败，请检查配置文件中的字典路径", false);
            return;
        }
    }

    if (!m_detector) {
        m_logger->logError("模型加载失败", false);
        return;
    } else {
        // 预热模型
        m_logger->logInfo("开始预热YOLO模型...", false);
        try {
            int warmup_iterations = 3;
            int input_width = 1200;
            int input_height = 800;
            cv::Mat dummy_image = cv::Mat::zeros(cv::Size(input_width, input_height), CV_8UC3);
            deploy::Image warmup_img(dummy_image.data, dummy_image.cols, dummy_image.rows);
            for (int i = 0; i < warmup_iterations; ++i) {
                m_detector->predict(warmup_img);
            }
            m_logger->logInfo("YOLO模型预热完成。", false);
        } catch (const std::exception& e) {
            m_logger->logError(fmt::format("YOLO模型预热失败: {}", e.what()), false);
            return;
        }
    }
}

ThreadManager::~ThreadManager() {
    stopThreads();
    m_udpTool->Close();
    m_detector.reset();
    m_trainParser.reset();
    m_trainNumberDetector.reset();
    m_paddleOcr.reset();
    m_logger->logInfo("线程管理器已销毁", false);
    emit m_Logs(QString("线程管理器已销毁"));
}

void ThreadManager::startThreads() {
    // 启动线程
    if (m_udpTool)  m_threads.emplace_back(std::make_shared<std::thread>(&ThreadManager::UdpToolRecvMessage, this));     
    m_threads.emplace_back(std::make_shared<std::thread>(&ThreadManager::UdpProcessMessage, this));
    m_threads.emplace_back(std::make_shared<std::thread>(&ThreadManager::PicProcessThread, this));
}

void ThreadManager::stopThreads() {
    threadStop = true;
    for (auto &thread : m_threads) {
        if (thread->joinable()) {
            thread->join();
        }
    }
    m_threads.clear();
}

bool ThreadManager::UdpToolRecvMessage() {
    char buffer[1024];
    sockaddr_in client_address;
    if (false == m_udpTool->Bind()) {
        m_logger->logError("UDP绑定失败", false);
        emit m_Logs(QString("UDP绑定失败"));
        return false;
    }
    m_udpTool->SetRecvTimeout(1);
    m_udpTool->SetSendTimeout(1);
    emit m_Logs(QString("接收消息线程启动"));
    m_logger->logInfo(fmt::format("接收消息线程启动"), false);
    while (!threadStop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::memset(buffer, '\0', sizeof(buffer));
        try {
            int recv_len = m_udpTool->Recv(buffer, sizeof(buffer), &client_address);
            if (recv_len > 0) {

                std::string currentMsg(buffer);
                auto now = std::chrono::system_clock::now();
                bool isDuplicate = false; 
    
                if (lastTask)  // 有值的时候判断 刚开始没有值不会进来
                {
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastTask->timestamp).count();
                    // 如果是相同任务且小于10秒，则认为是重复任务
                    if(lastTask->taskId == currentMsg && duration < TASK_WINDOW) {
                        isDuplicate = true;
                        m_logger->logInfo(fmt::format("检测到重复任务: {}，已忽略: ", currentMsg), false);
                        emit m_Logs("检测到重复任务，已忽略: " + QString::fromStdString(currentMsg));
                    }
                }
    
                if(!isDuplicate) {
                    // 更新任务记录
                    lastTask = TaskRecord{currentMsg, now};
                    
                    // 处理新任务
                    m_mtx_udpTool->lock();
                    m_queue_udpTool.emplace(currentMsg);
                    m_mtx_udpTool->unlock();
    
                    emit m_Logs(QString("接收到的数据: %1").arg(currentMsg.c_str()));
                    m_logger->logInfo(fmt::format("接收到的数据: {}", currentMsg), false);
                }
            }
        }
        catch (const std::exception &e) {
            m_logger->logError( fmt::format("消息接收异常: {}", e.what()) , false);
            continue;
        }
    }

    m_udpTool->Close();
    m_logger->logInfo("接收消息线程退出", false);
    return true;
}

bool ThreadManager::UdpProcessMessage(){

    emit m_Logs (QString("处理消息线程启动"));
    m_logger->logInfo(fmt::format("处理消息线程启动"), false);
    while(!threadStop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        m_mtx_udpTool->lock();
        if (!m_queue_udpTool.empty()) {
            std::string msg = m_queue_udpTool.front();
            m_queue_udpTool.pop();
            m_mtx_udpTool->unlock();
            // Message format: {BC}&timestamp&105-x&any_char
            std::regex msg_regex(R"(\{BC\}&(\d+)&(105-x)&(.*))");
            std::smatch match;
            if (std::regex_match(msg, match, msg_regex)) {
                if (match.size() == 4) {
                    std::string timestamp = match[1].str();
                    std::string channel_info = match[2].str(); 
                    // std::string any_char = match[3].str(); 

                    if (channel_info.rfind("105-x", 0) == 0) { 
                        m_logger->logInfo(fmt::format("有效消息格式: 时间戳 {}, 通道 {}", timestamp, channel_info), false);
                        emit m_Logs(QString("有效消息格式: 时间戳 %1, 通道 %2").arg(timestamp.c_str()).arg(channel_info.c_str()));

                        std::filesystem::path image_base_path = m_GlobalParam.imagePath;
                        std::filesystem::path image_folder_path = image_base_path / timestamp;

                        m_logger->logInfo(fmt::format("图片文件夹路径: {}", image_folder_path.string()), false);

                        std::vector<std::filesystem::path> image_files;
                        // std::regex file_regex_pattern(R"(105-(\d{3})-x\.jpg");
                        std::regex file_regex_pattern(R"(^105-(\d{3})-x\.jpg$)");

                        if (std::filesystem::exists(image_folder_path) && std::filesystem::is_directory(image_folder_path)) {
                            for (const auto& entry : std::filesystem::directory_iterator(image_folder_path)) {
                                if (entry.is_regular_file()) {
                                    std::string filename = entry.path().filename().string();
                                    if (std::regex_match(filename, file_regex_pattern)) {
                                        image_files.push_back(entry.path());
                                    }
                                }
                            }

                            std::sort(image_files.begin(), image_files.end(),
                                      [&](const std::filesystem::path& a, const std::filesystem::path& b) {
                                          std::smatch sm_a, sm_b;
                                          std::string fn_a = a.filename().string();
                                          std::string fn_b = b.filename().string();
                                          std::regex_search(fn_a, sm_a, file_regex_pattern);
                                          std::regex_search(fn_b, sm_b, file_regex_pattern);
                                          return std::stoi(sm_a[1].str()) < std::stoi(sm_b[1].str());
                                      });

                            m_logger->logInfo(fmt::format("找到并排序 {} 个105通道图片文件", image_files.size()), false);
                            
                            int totalGroups = (image_files.size() >= 3) ? (image_files.size() - 2) : 0;
                            emit m_UpdateProgress(0, totalGroups);
                            emit m_UpdateCurrentGroup(QString("开始处理图片组"));

                            if (image_files.size() >= 3) {
                                for (size_t i = 0; i <= image_files.size() - 3; ++i) {
                                    if (threadStop) break; 

                                    std::vector<std::future<cv::Mat>> image_futures;
                                    for (int j = 0; j < 3; ++j) {
                                        image_futures.push_back(std::async(std::launch::async, [&, j]() {
                                            return cv::imread(image_files[i + j].string());
                                        }));
                                    }
                                    
                                    // 获取加载结果
                                    cv::Mat img1 = image_futures[0].get();
                                    cv::Mat img2 = image_futures[1].get();
                                    cv::Mat img3 = image_futures[2].get();

                                    // cv::Mat img1 = cv::imread(image_files[i].string());
                                    // cv::Mat img2 = cv::imread(image_files[i + 1].string());
                                    // cv::Mat img3 = cv::imread(image_files[i + 2].string());

                                    if (img1.empty() || img2.empty() || img3.empty()) {
                                        m_logger->logError(fmt::format("无法加载用于拼接的图片: {} 或 {} 或 {}", 
                                            image_files[i].string(), image_files[i+1].string(), image_files[i+2].string()), false);
                                        continue;
                                    }
                                    cv::Mat resized_image = getStitchImageOptimized(img1,img2,img3, m_GlobalParam.factor, 
                                                                           cv::Size(m_GlobalParam.resizeWidth,
                                                                           m_GlobalParam.reiszeHeight));

                                    // 把图像高度去除一般保留下半部分图片
                                    int half_height = resized_image.rows / 2;
                                    cv::Rect roi(0, half_height, resized_image.cols, half_height);
                                    cv::Mat cropped_image = resized_image(roi);

                                    StitchedImageData data;
                                    data.image = cropped_image;
                                    data.timestamp = timestamp;
                                    
                                    std::smatch seq_match;
                                    std::string first_img_filename = image_files[i].filename().string();
                                    if(std::regex_search(first_img_filename, seq_match, file_regex_pattern) && seq_match.size() > 1) {
                                        data.imageSequenceNumber = seq_match[1].str();
                                    } else {
                                        data.imageSequenceNumber = std::to_string(i); 
                                    }
                                    // 0 开始 1 中间 2 结束
                                    int total_stitched_images = image_files.size() - 2;
                                    if (i == 0) {
                                        data.flag = 0; 
                                    }
                                    if (i == total_stitched_images -1 ) { 
                                        data.flag = (data.flag == 0 && total_stitched_images == 1) ? 0 : 2; 
                                        if (total_stitched_images > 1 && i == total_stitched_images -1) data.flag = 2; 
                                            else if (total_stitched_images == 1) data.flag = 0; 
                                    }
                                    else if (i > 0 && i < total_stitched_images -1) {
                                        data.flag = 1; 
                                    }
                                    
                                    if (total_stitched_images == 1) { 
                                        data.flag = 0; 
                                    } else if (i == 0) {
                                        data.flag = 0; 
                                    } else if (i == total_stitched_images - 1) {
                                        data.flag = 2; 
                                    } else {
                                        data.flag = 1; 
                                    }

                                    m_mtx_picProcess->lock();
                                    m_queue_picProcess.push(data);
                                    m_mtx_picProcess->unlock();

                                    m_logger->logInfo(fmt::format("已拼接并推送图片: 序号 {}, 时间戳 {}, 标志 {}", data.imageSequenceNumber, data.timestamp, data.flag), false);
                                    emit m_Logs(QString("已拼接图片: 序号 %1, 时间戳 %2, 标志 %3").arg(data.imageSequenceNumber.c_str()).arg(data.timestamp.c_str()).arg(data.flag));
                                    
                                    // 更新进度条和当前处理组信息
                                    emit m_UpdateProgress(i + 1, image_files.size() - 2);
                                    emit m_UpdateCurrentGroup(QString("正在处理图片组: %1/%2").arg(i + 1).arg(image_files.size() - 2));
                                }
                            } else {
                                m_logger->logError(fmt::format("图片数量不足3张无法拼接，在目录: {}", image_folder_path.string()), false);
                                emit m_Logs(QString("图片数量不足3张无法拼接于目录: %1").arg(image_folder_path.string().c_str()));
                                emit m_UpdateProgress(0, 0);
                                emit m_UpdateCurrentGroup(QString("无法处理：图片数量不足"));
                            }
                        } else {
                            m_logger->logError(fmt::format("图片文件夹不存在或不是目录: {}", image_folder_path.string()), false);
                            emit m_Logs(QString("图片文件夹不存在: %1").arg(image_folder_path.string().c_str()));
                            emit m_UpdateProgress(0, 0);
                            emit m_UpdateCurrentGroup(QString("无法处理：文件夹不存在"));
                        }
                    } else {
                        m_logger->logError(fmt::format("接收到消息但通道非105-x: {}", msg), false);
                        emit m_Logs(QString("接收到消息但通道非105-x: %1").arg(msg.c_str()));
                        emit m_UpdateProgress(0, 0);
                        emit m_UpdateCurrentGroup(QString("无法处理：通道错误"));
                    }
                } else {
                    m_logger->logError(fmt::format("无效消息格式 (无法解析): {}", msg), false);
                    emit m_Logs(QString("无效消息格式 (无法解析): %1").arg(msg.c_str()));
                    emit m_UpdateProgress(0, 0);
                    emit m_UpdateCurrentGroup(QString("无法处理：消息格式无效"));
                }
            } else {
                m_logger->logError(fmt::format("接收到消息但不匹配指定格式: {}", msg), false);
                emit m_Logs(QString("处理消息: %1 (格式不匹配)").arg(msg.c_str()));
                emit m_UpdateProgress(0, 0);
                emit m_UpdateCurrentGroup(QString("无法处理：消息格式不匹配"));
            }
        } else {
            m_mtx_udpTool->unlock();
        }
    }
    m_logger->logInfo("处理消息线程退出", false);
    return true;

}

bool ThreadManager::PicProcessThread() {
    emit m_Logs(QString("图片处理线程启动"));
    m_logger->logInfo(fmt::format("图片处理线程启动"), false);
    while (!threadStop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
        m_mtx_picProcess->lock();
        if (!m_queue_picProcess.empty()) {
            StitchedImageData data = m_queue_picProcess.front();
            m_queue_picProcess.pop();
            m_mtx_picProcess->unlock();

            if (data.flag == 0) {
                emit m_ShowString("正在处理图片中...");
                trianNums.clear();
                trianString.clear();
                trainNumCount = 0;
                m_trainNumberDetector->lastReportedNumber.clear();
            }

            m_logger->logInfo(fmt::format("处理拼接图片: 序号 {}, 时间戳 {}, 标志 {}", data.imageSequenceNumber, data.timestamp, data.flag), false);
            emit m_Logs(QString("处理拼接图片: 序号 %1, 时间戳 %2, 标志 %3").arg(data.imageSequenceNumber.c_str()).arg(data.timestamp.c_str()).arg(data.flag));

            if (data.image.empty()) {
                m_logger->logError(fmt::format("当前图像为空，无法处理"), false);
                emit m_Logs(QString("当前图像为空，无法处理"));
                continue;
            }
            
            std::string currentTrianNum, extraTrainNum;
            deploy::Image stitched_image(data.image.data, data.image.cols, data.image.rows);
            deploy::DetectRes yolo_detection_result = m_detector->predict(stitched_image);
           
            if (m_GlobalParam.recMode == 0) {
                currentTrianNum = getCurrentNum(yolo_detection_result, m_labels, data.image.cols, data.image.rows, 5.0);
            }
            else if (m_GlobalParam.recMode == 1) {
                std::vector<std::string> ocrTexts;
                auto filterBoxes = preprocess_detection_result(yolo_detection_result, data.image.cols, data.image.rows);
                std::vector<PaddleOCR::YoloDetectionBox> custom_boxes_for_ocr;
                for (size_t i = 0; i < filterBoxes.num; ++i) {
                    const deploy::Box& y_box = filterBoxes.boxes[i];
                    PaddleOCR::YoloDetectionBox ocr_box;
                    ocr_box.left = y_box.left;
                    ocr_box.top = y_box.top;
                    ocr_box.right = y_box.right;
                    ocr_box.bottom = y_box.bottom;
                    ocr_box.score = filterBoxes.scores[i];
                    custom_boxes_for_ocr.push_back(ocr_box);
                }
                auto ocr_processing_result = m_paddleOcr->inference_from_custom_boxes(data.image, custom_boxes_for_ocr, ocrTexts);
                if (ocrTexts.size() == 0) {
                    currentTrianNum = "";
                }
                else {
                    currentTrianNum = ocrTexts[0];
                }
            }
            
            // 保存识别图像
            if (m_GlobalParam.isSave) {
                std::string save_path = m_GlobalParam.savePath + data.timestamp + "\\";
                if (!std::filesystem::exists(save_path)) {
                    std::filesystem::create_directories(save_path);
                    emit m_Logs(QString("创建保存目录：%1成功").arg(save_path.c_str()));
                    m_logger->logInfo(fmt::format("创建保存目录：{}成功", save_path), false);
                }
                
                // visualize(data.image, yolo_detection_result, m_labels);
                if(currentTrianNum.length() > 0) {
                    std::string saveFile = save_path + data.imageSequenceNumber + ".jpg";
                    cv::imwrite(saveFile, data.image);
                }
            }

            m_trainNumberDetector->processFrame(currentTrianNum, extraTrainNum);

            if (!extraTrainNum.empty()) {
                m_logger->logInfo(fmt::format("识别到车号: {}", extraTrainNum), false);
                emit m_Logs(QString("识别到车号: %1").arg(extraTrainNum.c_str()));
                trainNumCount++;
                trianString.append(FileTools::getInstance().str_format("#%d&%s", trainNumCount, extraTrainNum.c_str()));
                trianNums.push_back(extraTrainNum);
            }

            if (data.flag == 2) {
                // 结束标志
                m_logger->logInfo(fmt::format("处理结束标志: 序号 {}, 时间戳 {}, 标志 {}", data.imageSequenceNumber, data.timestamp, data.flag), false);
                emit m_Logs(QString("处理结束标志: 序号 %1, 时间戳 %2, 标志 %3").arg(data.imageSequenceNumber.c_str()).arg(data.timestamp.c_str()).arg(data.flag));
                
                // 初始化车号结果
                std::string trainPlants, trianDiretion, CorrectString;

                if (trainNumCount == 0) {
                    m_logger->logInfo(fmt::format("当前任务未检测到车号"), false);
                    emit m_Logs(QString("当前任务未检测到车号"));

                    std::string msg = FileTools::getInstance().str_format("{CHJG}&%s&2&0&NULL&0&NULL", data.timestamp.c_str());
                    // 发送消息
                    emit m_ShowString(msg);
                    m_udpTool->Send(msg.c_str(), msg.size(),m_udpToolParam.send_ip, m_udpToolParam.send_port);
                    m_logger->logInfo(fmt::format("发送消息: {}", msg), false);
                    emit m_Logs(QString("发送消息: %1").arg(msg.c_str()));
                }
                else {

                    switch (m_AlgParam.trian_type) {
                        case 0:
                            m_logger->logInfo(fmt::format("地铁纯数字车号识别结果"), false);
                            emit m_Logs(QString("地铁纯数字车号识别结果"));
                            m_MetrotrainParser->parse(trianString);
                            trainPlants = m_MetrotrainParser->getTrainNumber();
                            trianDiretion = m_MetrotrainParser->getDirection();
                            CorrectString = m_MetrotrainParser->getCorrectedInput();
                            break;
                        case 1:
                            trainPlants = "N/A";
                            trianDiretion = "N/A";
                            CorrectString = trianString;
                            break;
                        case 2:
                            m_logger->logInfo(fmt::format("高铁车号识别结果"), false);
                            emit m_Logs(QString("高铁车号识别结果"));
                            m_trainParser->parse(trianString);
                            trainPlants = m_trainParser->getTrainNumber();
                            trianDiretion = m_trainParser->getDirection();
                            CorrectString = m_trainParser->getCorrectedInput();
                            break;
                        default:
                            m_logger->logInfo(fmt::format("未配置车型，请在配置文件中配置车型"), false);
                            emit m_Logs(QString("未配置车型，请在配置文件中配置车型"));
                            trainPlants = "N/A";
                            trianDiretion = "N/A";
                            CorrectString = trianString;
                            break;
                    }
                    
                    std::string msg = FileTools::getInstance().str_format("{CHJG}&%s&2&%s&%s&%d&%s", data.timestamp.c_str(),
                                      trianDiretion.c_str(), trainPlants.c_str(), trainNumCount, CorrectString.c_str() );
                    emit m_ShowString(msg);
                    // 发送消息
                    m_udpTool->Send(msg.c_str(), msg.size(),m_udpToolParam.send_ip, m_udpToolParam.send_port);
                    m_logger->logInfo(fmt::format("当前任务识别完成，发送消息: {}", msg), false);
                    emit m_Logs(QString("当前任务识别完成，发送消息: %1").arg(msg.c_str()));
                }
            }

        } else {
            m_mtx_picProcess->unlock();
        }
    }
    m_logger->logInfo("图片处理线程退出", false);
    return true;
}

// 在图像上可视化推理结果
void ThreadManager::visualize(cv::Mat& image, const deploy::DetectRes& result, const std::vector<std::string>& labels) {

    for (size_t i = 0; i < result.num; ++i) {
        const auto& box        = result.boxes[i];
        int         cls        = result.classes[i];
        float       score      = result.scores[i];
        const auto& label      = labels[cls];
        std::string label_text = label + " " + cv::format("%.3f", score);

        // 绘制矩形和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
    }
}


std::string ThreadManager::getCurrentNum(const deploy::DetectRes& result, 
                                         const std::vector<std::string>& labels, 
                                         int image_width, int image_height, float margin) {
    if (result.num == 0) {
        return ""; // 没有检测到任何物体
    }
    std::vector<std::pair<float, int>> detected_digits;
    detected_digits.reserve(result.num);
    for (int i = 0; i < result.num; ++i) {
        const deploy::Box& box = result.boxes[i];
        int class_id = result.classes[i];
        float score = result.scores[i]; 
        if (box.left < margin || box.top < margin || box.right > image_width - margin || box.bottom > image_height - margin) {
            continue; 
        }
        detected_digits.push_back({box.left, class_id});
    }

    std::sort(detected_digits.begin(), detected_digits.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first < b.first;
              });

    std::string train_number_str;
    
    for (const auto& digit_info : detected_digits) {
        train_number_str += labels[digit_info.second];
    }

    return train_number_str;
}


cv::Mat ThreadManager::getStitchImageOptimized( const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3,
                                 double height_reduction_factor, const cv::Size& final_target_size) 
{
    
    if (img1.empty() || img2.empty() || img3.empty()) {
        return cv::Mat();
    }
    
    // 计算目标尺寸
    int target_height = static_cast<int>(img1.rows / height_reduction_factor);
    int target_width_per_img = final_target_size.width / 3;
    
    // 预分配最终图像
    cv::Mat final_image(final_target_size, img1.type());
    
    // 直接resize到目标区域，避免中间拷贝
    cv::Rect roi1(0, 0, target_width_per_img, final_target_size.height);
    cv::Rect roi2(target_width_per_img, 0, target_width_per_img, final_target_size.height);
    cv::Rect roi3(2 * target_width_per_img, 0, target_width_per_img, final_target_size.height);
    
    cv::resize(img1, final_image(roi1), roi1.size(), 0, 0, cv::INTER_AREA);
    cv::resize(img2, final_image(roi2), roi2.size(), 0, 0, cv::INTER_AREA);
    cv::resize(img3, final_image(roi3), roi3.size(), 0, 0, cv::INTER_AREA);
    
    return final_image;
}

deploy::DetectRes ThreadManager::preprocess_detection_result(
    const deploy::DetectRes& yolo_detection_result,
    int image_width,
    int image_height) {

    // 4. 没有检测到框就直接返回
    if (yolo_detection_result.num == 0) {
        return yolo_detection_result;
    }

    std::vector<int> filtered_classes;
    std::vector<float> filtered_scores;
    std::vector<deploy::Box> filtered_and_expanded_boxes;

    float image_w_float = static_cast<float>(image_width);
    float image_h_float = static_cast<float>(image_height);

    for (int i = 0; i < yolo_detection_result.num; ++i) {
        const deploy::Box& current_box = yolo_detection_result.boxes[i];

        // a. 去除在和边界的框
        // 假设边界框的定义是其任何一边紧贴图像边缘
        // 注意：浮点数比较可能需要 epsilon，但这里我们假设边界框的坐标是精确的
        if (current_box.left < (0.0f + 20) ||
            current_box.top < (0.0f + 20)||
            current_box.right > (image_w_float-20) ||
            current_box.bottom > (image_h_float-20)) {
            continue; // 跳过此边界框
        }

        // b. 框往外扩10%
        float box_width = current_box.right - current_box.left;
        float box_height = current_box.bottom - current_box.top;

        // 10% 扩张量，平均分配到两边，所以每边是 5%
        float expansion_w = box_width * 0.01f;
        float expansion_h = box_height * 0.01f;

        float new_left = current_box.left - expansion_w;
        float new_top = current_box.top - expansion_h;
        float new_right = current_box.right + expansion_w;
        float new_bottom = current_box.bottom + expansion_h;

        // 确保扩大的框不会超出图像边界 (裁剪)
        new_left = (std::max)(0.0f, new_left);
        new_top = (std::max)(0.0f, new_top);
        new_right = (std::min)(image_w_float, new_right);
        new_bottom = (std::min)(image_h_float, new_bottom);
        
        // 如果裁剪后框无效 (例如，宽度或高度变为0或负数)，则跳过
        if (new_left >= new_right || new_top >= new_bottom) {
            continue;
        }

        filtered_and_expanded_boxes.emplace_back(new_left, new_top, new_right, new_bottom);
        filtered_classes.push_back(yolo_detection_result.classes[i]);
        filtered_scores.push_back(yolo_detection_result.scores[i]);
    }

    return deploy::DetectRes(
        static_cast<int>(filtered_and_expanded_boxes.size()),
        filtered_classes,
        filtered_scores,
        filtered_and_expanded_boxes);
}
