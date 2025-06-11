#include "paddleocr.h"
#include <tbb/tbb.h> 
#include <mutex>
#include <algorithm>
std::mutex print_mutex;
PaddleOCR::PaddleOCR() {
    memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

PaddleOCR::~PaddleOCR() {
    delete session_det;
    delete session_cls;
    delete session_rec;
    session_det = nullptr;
    session_cls = nullptr;
    session_rec = nullptr;

    delete options_det;
    delete options_cls;
    delete options_rec;
    options_det = nullptr;
    options_cls = nullptr;
    options_rec = nullptr;

    delete env_det;
    delete env_cls;
    delete env_rec;
    env_det = nullptr;
    env_cls = nullptr;
    env_rec = nullptr;
    
    delete memory_info;
    memory_info = nullptr;

    clear_nodes_vector(input_nodes_det);
    clear_nodes_vector(input_nodes_rec);
    clear_nodes_vector(input_nodes_cls);
    clear_nodes_vector(output_nodes_det);
    clear_nodes_vector(output_nodes_rec);
    clear_nodes_vector(output_nodes_cls);
}

void print_results(
    const std::vector<std::vector<float>>& max_values,
    const std::vector<std::vector<size_t>>& max_indices)
{
    std::cout << "=== max_values ===" << std::endl;
    for (size_t i = 0; i < max_values.size(); ++i) {
        std::cout << "Batch " << i << ": ";
        for (size_t j = 0; j < max_values[i].size(); ++j) {
            std::cout << max_values[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== max_indices ===" << std::endl;
    for (size_t i = 0; i < max_indices.size(); ++i) {
        std::cout << "Batch " << i << ": ";
        for (size_t j = 0; j < max_indices[i].size(); ++j) {
            std::cout << max_indices[i][j] << " ";
        }
        std::cout << std::endl;
    }
}



void PaddleOCR::clear_nodes_vector(std::vector<yo::Node>& nodes) {
    for (auto& node : nodes) {
        delete[] node.name;
        node.name = nullptr;
    }
    nodes.clear();
}

void PaddleOCR::load_onnx_info(Ort::Session* session, std::vector<yo::Node>& input, std::vector<yo::Node>& output, const std::string& onnx_name) {
    if (!session) return;

    Ort::AllocatorWithDefaultOptions allocator;
    
    // 模型输入信息
    input.clear(); 
    for (size_t index = 0; index < session->GetInputCount(); index++) {
        Ort::AllocatedStringPtr input_name_Ptr = session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(index);
        
        yo::Node node;
        node.dim = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        
        const char* name_str = input_name_Ptr.get();
        if (name_str) {
            size_t name_length = strlen(name_str) + 1;
            node.name = new char[name_length];
            #ifdef _MSC_VER
                strcpy_s(node.name, name_length, name_str);
            #else
                strcpy(node.name, name_str);
            #endif
        }
        input.push_back(node); 
    }
    // 模型输出信息
    output.clear(); 
    for (size_t index = 0; index < session->GetOutputCount(); index++) {
        Ort::AllocatedStringPtr output_name_Ptr = session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(index);
        
        yo::Node node;
        node.dim = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

        const char* name_str = output_name_Ptr.get();
        if (name_str) {
            size_t name_length = strlen(name_str) + 1;
            node.name = new char[name_length];
            #ifdef _MSC_VER
                strcpy_s(node.name, name_length, name_str);
            #else
                strcpy(node.name, name_str);
            #endif
        }
        output.push_back(node);
    }
    // 打印日志
    std::cout << "***************" << onnx_name << "***************" << std::endl;
    for (const auto &node : input) {
        std::ostringstream dim_ss;
        dim_ss << "[";
        for (size_t i = 0; i < node.dim.size(); ++i) {
            dim_ss << node.dim[i];
            if (i != node.dim.size() - 1) dim_ss << ",";
        }
        dim_ss << "]";
        std::cout << "input_name= [" << (node.name ? node.name : "NULL") << "] ===> " << dim_ss.str() << std::endl;
    }
    for (const auto &node : output) {
        std::ostringstream dim_ss;
        dim_ss << "[";
        for (size_t i = 0; i < node.dim.size(); ++i) {
            dim_ss << node.dim[i];
            if (i != node.dim.size() - 1) dim_ss << ",";
        }
        dim_ss << "]";
        std::cout << "output_name= [" << (node.name ? node.name : "NULL") << "] ==> " << dim_ss.str() << std::endl;
    }
    std::cout << "************************************\n" << std::endl;
}


std::variant<bool, std::string> PaddleOCR::initialize(const std::vector<std::string>& onnx_paths, bool is_cuda) {
    assert(onnx_paths.size() == 3);

    for (size_t i = 0; i < onnx_paths.size(); ++i) {
        if (!MT::FileExists(onnx_paths[i])) {
            std::ostringstream oss;
            oss << "模型路径不存在 : " << onnx_paths[i];
            return oss.str();
        }
    }

    try {
        env_det = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ocr_det_env");
        env_cls = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ocr_cls_env");
        env_rec = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ocr_rec_env");

        options_det = new Ort::SessionOptions();
        options_cls = new Ort::SessionOptions();
        options_rec = new Ort::SessionOptions();

        options_det->SetIntraOpNumThreads(2);
        options_det->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        options_cls->SetIntraOpNumThreads(2);
        options_cls->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        options_rec->SetIntraOpNumThreads(2);
        options_rec->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (is_cuda) {
            OrtCUDAProviderOptions cuda_options{}; 

            options_det->AppendExecutionProvider_CUDA(cuda_options);
            options_cls->AppendExecutionProvider_CUDA(cuda_options);
            options_rec->AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Using CUDA..." << std::endl;
        } else {
            std::cout << "Using CPU..." << std::endl;
        }

    #ifdef _WIN32

        std::vector<std::wstring> w_onnx_paths;
        for(const auto& p : onnx_paths) {
            setlocale(LC_CTYPE, ""); 
            size_t out_len;
            std::wstring wstr(p.length() + 1, L'\0'); 
            #if _MSC_VER 
                mbstowcs_s(&out_len, &wstr[0], wstr.size(), p.c_str(), p.length());
            #else
                out_len = mbstowcs(&wstr[0], p.c_str(), p.length());
            #endif
            wstr.resize(out_len);
            w_onnx_paths.push_back(wstr);
        }
        session_det = new Ort::Session(*env_det, w_onnx_paths[0].c_str(), *options_det);
        session_cls = new Ort::Session(*env_cls, w_onnx_paths[1].c_str(), *options_cls);
        session_rec = new Ort::Session(*env_rec, w_onnx_paths[2].c_str(), *options_rec);
    #else
        session_det = new Ort::Session(*env_det, onnx_paths[0].c_str(), *options_det);
        session_cls = new Ort::Session(*env_cls, onnx_paths[1].c_str(), *options_cls);
        session_rec = new Ort::Session(*env_rec, onnx_paths[2].c_str(), *options_rec);
    #endif
    } catch (const Ort::Exception& e) {
        std::ostringstream oss;
        oss << "ONNX Runtime Exception during initialization: " << e.what();
        return oss.str();
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Standard Exception during initialization: " << e.what();
        return oss.str();
    }

    clear_nodes_vector(input_nodes_det); clear_nodes_vector(output_nodes_det);
    clear_nodes_vector(input_nodes_cls); clear_nodes_vector(output_nodes_cls);
    clear_nodes_vector(input_nodes_rec); clear_nodes_vector(output_nodes_rec);

    this->load_onnx_info(this->session_det, this->input_nodes_det, this->output_nodes_det, "det.onnx");
    this->load_onnx_info(this->session_cls, this->input_nodes_cls, this->output_nodes_cls, "cls.onnx");
    this->load_onnx_info(this->session_rec, this->input_nodes_rec, this->output_nodes_rec, "rec.onnx");
    
    this->is_inited = true;
    std::cout << "initialize ok!!" << std::endl;
    return true;
}

int PaddleOCR::setparms(ParamsOCR parms_in) { 
    this->params = std::move(parms_in);
    if (!this->params.dictionary || !MT::FileExists(this->params.dictionary)) {
        std::ostringstream oss;
        oss << "OCR字典文件不存在: " 
            << (this->params.dictionary ? this->params.dictionary : "NULL_PATH");
        std::cerr << oss.str() << std::endl;
        return 0;
    } else {
        std::string dict_path(this->params.dictionary);
        if (this->dictionary.size() <= 0) { 
            if(!this->dictionary.load(dict_path)){
                std::cerr << "加载字典文件失败: " << dict_path << std::endl;
                return 0;
            }
        }
    }
    return 1;
}

void PaddleOCR::preprocess(cv::Mat &image) {
    int original_width = image.cols;
    int original_height = image.rows;
    
    int new_width = static_cast<int>(std::ceil(static_cast<double>(original_width) / 32.0)) * 32;
    int new_height = static_cast<int>(std::ceil(static_cast<double>(original_height) / 32.0)) * 32;
    
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
    cv::subtract(resized_image, cv::Scalar(0.485, 0.456, 0.406), resized_image);
    cv::divide(resized_image, cv::Scalar(0.229, 0.224, 0.225), resized_image);
    
    cv::Mat blob = cv::dnn::blobFromImage(resized_image, 1.0, cv::Size(new_width, new_height), cv::Scalar(0, 0, 0), true, false);

    if(!input_nodes_det.empty()){ 
         this->input_nodes_det[0].dim = {1, 3, static_cast<int64_t>(new_height), static_cast<int64_t>(new_width)};
    } else {
        std::cerr << "Error: 前处理中input_nodes_det为空." << std::endl;
    }


    input_images.clear();
    input_images.push_back(std::move(blob));
}

std::variant<bool, std::string> PaddleOCR::inference(cv::Mat &image, std::vector<std::string>& texts) {
    if (image.empty()) return std::string("Image cannot be empty!");
    if (!this->is_inited) return std::string("Model not initialized!");
    
    this->ori_img = &image; 

    try {
        this->preprocess(image); 
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "图像前处理失败" << e.what();
        return oss.str();
    }

    std::optional<std::vector<cv::Mat>> det_result = this->infer_det(); 
    if (!det_result.has_value() || det_result.value().empty()) { 
        return std::string("检测失败!");
    }
    std::cout << "Detection success! Found " << det_result.value().size() << " text regions." << std::endl;

    std::optional<std::vector<cv::Mat>> cls_result = this->infer_cls(det_result.value());

    if (!cls_result.has_value() || cls_result.value().empty()) { 
        return std::string("方向分类失败");
    }

    std::cout << "方向分类成功！" << std::endl;

    std::optional<std::vector<Ort::Value>> rec_result = this->infer_rec(cls_result.value());
    if (!rec_result.has_value() || rec_result.value().empty()) {
        return std::string("文本识别失败！");
    }

    std::cout << "文本识别成功，开始后处理！" << std::endl;

    try {
        texts = this->postprocess(rec_result.value());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "后处理失败 " << e.what();
        return oss.str();
    }
    
    return true;
}

// void PaddleOCR::postprocess(std::vector<Ort::Value> &output_tensors) {
//     if (output_tensors.empty()) {
//         std::cerr << "后处理：没有输出张量." << std::endl;
//         return;
//     }
  
//     float* output_data = output_tensors[0].GetTensorMutableData<float>();
//     const auto& output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); 

//     if (output_shape.size() != 3) {
//         std::cerr << "后处理：输出张量的维度数量不符合预期: " << output_shape.size() << std::endl;
//         return;
//     }

//     long long batch_size = output_shape[0];
//     long long M_sequence_len = output_shape[1]; 
//     long long num_classes_dict = output_shape[2]; 


//     std::vector<std::vector<float>> max_values(batch_size, std::vector<float>(M_sequence_len, -std::numeric_limits<float>::infinity()));
//     std::vector<std::vector<size_t>> max_indices(batch_size, std::vector<size_t>(M_sequence_len, 0)); 

//     tbb::parallel_for(0, (int)batch_size, 1, [&](int batch_idx) {
//         tbb::parallel_for(0, (int)M_sequence_len, 1, [&](int m_idx) {
//             float current_max_val = -std::numeric_limits<float>::infinity();
//             size_t current_max_idx = 0; 
//             for (int class_idx = 0; class_idx < num_classes_dict; ++class_idx) {
//                 long long flat_index = batch_idx * M_sequence_len * num_classes_dict +
//                                    m_idx * num_classes_dict +
//                                    class_idx;
//                 float value = output_data[flat_index];
//                 if (value > current_max_val) {
//                     current_max_val = value;
//                     current_max_idx = static_cast<size_t>(class_idx);
//                 }
//             }
//             max_values[batch_idx][m_idx] = current_max_val;
//             max_indices[batch_idx][m_idx] = current_max_idx;
//         });
//     });

//     std::vector<std::string> results_text(batch_size);
//     for (size_t batch_idx = 0; batch_idx < static_cast<size_t>(batch_size); ++batch_idx) {
//         std::string current_text = "";
//         size_t last_char_idx = 0; 

//         for (size_t m_idx = 0; m_idx < static_cast<size_t>(M_sequence_len); ++m_idx) {
//             size_t char_idx = max_indices[batch_idx][m_idx];
//             float confidence = max_values[batch_idx][m_idx];

//             if (char_idx != 0 && (this->params.repeat || char_idx != last_char_idx)) {
//                  if (confidence >= this->params.text) { 
//                     std::string char_str = this->dictionary.get_char(static_cast<int>(char_idx - 1)); 
//                     current_text += char_str;
//                 }
//             }
//             if (char_idx != 0) { 
//                 // TODO 
//             }
//             last_char_idx = char_idx; 
//         }
//         results_text[batch_idx] = current_text;


//         if (batch_idx < this->polygons.size() && this->ori_img) { 

//             cv::RotatedRect rorect = cv::minAreaRect(this->polygons[batch_idx].points);
//             std::vector<cv::Point2f> ropoints(4);
//             rorect.points(ropoints.data()); 

//             cv::putText(*this->ori_img, std::to_string(batch_idx), ropoints[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 25, 255), 1);
//             if (results_text[batch_idx].empty()) {
//                 for (int i = 0; i < 4; i++) cv::line(*this->ori_img, ropoints[i], ropoints[(i + 1) % 4], cv::Scalar(50, 0, 255), 1);
//             } else {
//                 std::cout << "Text " << batch_idx << " => " << results_text[batch_idx] << std::endl;
//                 for (int i = 0; i < 4; i++) cv::line(*this->ori_img, ropoints[i], ropoints[(i + 1) % 4], cv::Scalar(25, 250, 50), 1);
//             }
//         }
//     }
// }

std::vector<std::string> PaddleOCR::postprocess(std::vector<Ort::Value> &output_tensors) {
    if (output_tensors.empty()) {
        std::cerr << "后处理：没有输出张量." << std::endl;
        return std::vector<std::string>();
    }
  
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    const auto& output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); 

    if (output_shape.size() != 3) {
        std::cerr << "后处理：输出张量的维度数量不符合预期: " << output_shape.size() << std::endl;
        return std::vector<std::string>();
    }

    long long batch_size = output_shape[0];
    long long M_sequence_len = output_shape[1]; 
    long long num_classes_dict = output_shape[2]; 

    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "M_sequence_len: " << M_sequence_len << std::endl;
    std::cout << "num_classes_dict: " << num_classes_dict << std::endl;
    
    std::vector<std::vector<float>> max_values(batch_size, std::vector<float>(M_sequence_len, -std::numeric_limits<float>::infinity()));
    std::vector<std::vector<size_t>> max_indices(batch_size, std::vector<size_t>(M_sequence_len, 0)); 

    tbb::parallel_for(0, (int)batch_size, 1, [&](int batch_idx) {
        tbb::parallel_for(0, (int)M_sequence_len, 1, [&](int m_idx) {
            float current_max_val = -std::numeric_limits<float>::infinity();
            size_t current_max_idx = 0; 
            for (int class_idx = 0; class_idx < num_classes_dict; ++class_idx) {
                long long flat_index = batch_idx * M_sequence_len * num_classes_dict +
                                   m_idx * num_classes_dict +
                                   class_idx;
                float value = output_data[flat_index];

                if (value > current_max_val) {
                    current_max_val = value;
                    current_max_idx = static_cast<size_t>(class_idx);
                }

            }
            max_values[batch_idx][m_idx] = current_max_val;
            max_indices[batch_idx][m_idx] = current_max_idx;
        });
    });

    // print_results(max_values, max_indices);

    std::vector<std::string> results_text(batch_size);
    for (size_t batch_idx = 0; batch_idx < static_cast<size_t>(batch_size); ++batch_idx) {
        std::string current_text = "";
        size_t last_char_idx = 0; 

        for (size_t m_idx = 0; m_idx < static_cast<size_t>(M_sequence_len); ++m_idx) {
            size_t char_idx = max_indices[batch_idx][m_idx];
            float confidence = max_values[batch_idx][m_idx];

            if (char_idx != 0 && (this->params.repeat || char_idx != last_char_idx)) {
                 if (confidence >= this->params.text) { 
                    std::string char_str = this->dictionary.get_char(static_cast<int>(char_idx - 1)); 
                    current_text += char_str;
                }
            }
            if (char_idx != 0) { 
                // TODO 
            }
            last_char_idx = char_idx; 
        }
        results_text[batch_idx] = current_text;
    }

    for (auto& text : results_text) {
        text.erase(
            std::remove_if(text.begin(), text.end(), [](char c) {
                return !std::isalnum(static_cast<unsigned char>(c));
            }),
            text.end()
        );
    }
    return results_text;
}





std::vector<cv::Point2f> PaddleOCR::unclip(const std::vector<cv::Point>& polygon) { // Made input const&
    std::vector<cv::Point2f> result;
    if (polygon.empty()) return result;

    double area = cv::contourArea(polygon);
    double length = cv::arcLength(polygon, true);
    
    if (length < 1e-3) { 
       
        for(const auto& pt : polygon) result.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        return result;
    }
    double distance = area * this->params.unclip_ratio / length;
    
    Clipper2Lib::Path64 path;
    for (const auto& pt : polygon) {
        path.push_back(Clipper2Lib::Point64(pt.x, pt.y));
    }
    Clipper2Lib::ClipperOffset clipper;
    clipper.AddPath(path, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
    
    Clipper2Lib::Paths64 solution;
    try {
        clipper.Execute(distance, solution); 
    } catch (const std::exception& e) {
        std::cerr << "ClipperOffset execution failed: " << e.what() << std::endl;
        // Fallback: return original points as Point2f
        for(const auto& pt : polygon) result.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        return result;
    }
    
    if (solution.empty() || solution[0].empty()) {
        std::cerr << "ClipperOffset failed: solution is empty or first path is empty!" << std::endl;
        // Fallback: return original points as Point2f
        for(const auto& pt : polygon) result.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        return result;
    }
    const Clipper2Lib::Path64& expanded = solution[0];
    for (const auto& pt : expanded) {
        result.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
    }
    return result;
}

float PaddleOCR::box_score_slow(cv::Mat &pred, const std::vector<cv::Point> &approx) { // Made approx const&
    if (pred.empty() || approx.empty()) {
        return 0.0f;
    }
    const int height = pred.rows;
    const int width = pred.cols;
    
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    
    std::vector<cv::Point> int_points;
    int_points.reserve(approx.size());
    for (const auto& pt : approx) {
        int_points.emplace_back(cv::Point(
            (std::min)((std::max)(pt.x, 0), width - 1),
            (std::min)((std::max)(pt.y, 0), height - 1)
        ));
    }
    if (int_points.size() < 3) return 0.0f;

    std::vector<std::vector<cv::Point>> contours = {int_points};
    cv::fillPoly(mask, contours, cv::Scalar(1));
    
    // int pixel_count = cv::countNonZero(mask);
    // if (pixel_count == 0) return 0.0f; 

    cv::Scalar mean_val = cv::mean(pred, mask);
    return static_cast<float>(mean_val[0]);
}


std::vector<PaddleOCR::Polygon> PaddleOCR::poly_from_bitmap(cv::Mat &pred, cv::Mat &bitmap) {
    std::vector<PaddleOCR::Polygon> result_polygons; // Renamed to avoid conflict
    cv::Mat binmat;

    if (bitmap.type() == CV_32F || bitmap.type() == CV_64F) {
        bitmap.convertTo(binmat, CV_8UC1, 255.0);
    } else if (bitmap.type() == CV_8UC1) {
        binmat = bitmap; 
    } else {
         std::cerr << "poly_from_bitmap: Unsupported bitmap type: " << bitmap.type() << std::endl;
         return result_polygons;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binmat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (input_nodes_det.empty() || ori_img == nullptr) {
        std::cerr << "poly_from_bitmap: input_nodes_det or ori_img not initialized." << std::endl;
        return result_polygons;
    }

    if (input_nodes_det[0].dim.size() < 4 || input_nodes_det[0].dim[3] == 0 || input_nodes_det[0].dim[2] == 0) {
         std::cerr << "poly_from_bitmap: Invalid dimensions in input_nodes_det." << std::endl;
         return result_polygons;
    }


    float scale_x = static_cast<float>(this->ori_img->cols) / static_cast<float>(this->input_nodes_det[0].dim[3]);
    float scale_y = static_cast<float>(this->ori_img->rows) / static_cast<float>(this->input_nodes_det[0].dim[2]);

    for (auto &contour : contours) {
        if (contour.size() < 3) continue; 
        double temp_area = cv::contourArea(contour);
        if (temp_area < this->params.min_area) continue;
        
        double epsilon = 0.005 * cv::arcLength(contour, true); 
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, epsilon, true);
        
        if (approx.size() < 3) continue; 

        float score = this->box_score_slow(pred, approx);
        if (score < this->params.thresh) {
            continue;
        }
        
        std::vector<cv::Point2f> unclip_poly_pts = this->unclip(approx);
        if(unclip_poly_pts.empty() || unclip_poly_pts.size() < 3) {
            std::cerr << "Unclip failed or produced too few points for a polygon." << std::endl;
            continue; 
        }

        std::transform(unclip_poly_pts.begin(), unclip_poly_pts.end(), unclip_poly_pts.begin(),
            [&](const cv::Point2f& pt) {
                return cv::Point2f(pt.x * scale_x, pt.y * scale_y);
            });
        result_polygons.push_back({score, unclip_poly_pts});
    }
    return result_polygons;
}

std::optional<std::vector<cv::Mat>> PaddleOCR::infer_det() {
    if (input_images.empty() || input_nodes_det.empty()) {
        std::cerr << "infer_det: No input images or det model input nodes not initialized." << std::endl;
        return std::nullopt;
    }
    if (this->input_images[0].empty()){
        std::cerr << "infer_det: Input image is empty." << std::endl;
        return std::nullopt;
    }


    std::vector<Ort::Value> input_tensor_values;
    try {
        input_tensor_values.push_back(Ort::Value::CreateTensor<float>(
            *memory_info, // Dereference pointer
            this->input_images[0].ptr<float>(),
            this->input_images[0].total(), // total number of elements
            this->input_nodes_det[0].dim.data(),
            this->input_nodes_det[0].dim.size())
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_det: Failed to create input tensor: " << e.what() << std::endl;
        return std::nullopt;
    }
    
    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto &node : this->input_nodes_det) input_names_cstr.push_back(node.name);
    for (const auto &node : this->output_nodes_det) output_names_cstr.push_back(node.name);
    if(input_names_cstr.empty() || output_names_cstr.empty()){
        std::cerr << "infer_det: Input or output node names are empty." << std::endl;
        return std::nullopt;
    }

    std::vector<Ort::Value> output_tensor_values;
    try {
        output_tensor_values = this->session_det->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(), 
            input_tensor_values.data(),
            input_tensor_values.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_det: Session Run failed: " << e.what() << std::endl;
        return std::nullopt;
    }
    
    if (output_tensor_values.empty()) {
        std::cerr << "infer_det: No output tensors from session run." << std::endl;
        return std::nullopt;
    }

    float* output_data = output_tensor_values[0].GetTensorMutableData<float>();
    const auto& output_shape = output_tensor_values[0].GetTensorTypeAndShapeInfo().GetShape();
    
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 1) {
         std::cerr << "infer_det: Unexpected output tensor shape." << std::endl;
         return std::nullopt;
    }
    std::cout << "det output shape=[";
    for(size_t i=0; i<output_shape.size(); ++i) std::cout << output_shape[i] << (i==output_shape.size()-1 ? "" : ",");
    std::cout << "]" << std::endl;

    cv::Mat prob_map = cv::Mat(static_cast<int>(output_shape[2]), static_cast<int>(output_shape[3]), CV_32FC1, output_data);
    
    cv::Mat bitmap;
    cv::threshold(prob_map, bitmap, this->params.thresh, 1.0, cv::THRESH_BINARY);
    // Ensure bitmap is CV_8U for findContours later if needed by poly_from_bitmap's current logic
    // poly_from_bitmap expects float [0,1] or uchar [0,255]. Here it's float [0,1].
    
    // cv::rectangle(bitmap, cv::Point(0,0), cv::Point(bitmap.cols-1, bitmap.rows-1), cv::Scalar(0), 2); // This blanks the border

    this->polygons = this->poly_from_bitmap(prob_map, bitmap);
    if (polygons.empty()) {
        std::cout << "infer_det: No polygons found from bitmap." << std::endl;
        return std::nullopt;
    }

    // Helper lambdas for geometry
    auto point_distance_sq = [](cv::Point2f p1, cv::Point2f p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
    };
     auto get_angle_deg = [](cv::Point2f p1, cv::Point2f p2) { // Angle of vector p1->p2
        return std::atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI;
    };


    std::vector<cv::Mat> cropped_text_images;
    if (input_nodes_rec.empty() || input_nodes_rec[0].dim.size() < 3) {
         std::cerr << "infer_det: Recognition model input nodes not properly initialized." << std::endl;
         return std::nullopt;
    }
    int rec_input_h = static_cast<int>(this->input_nodes_rec[0].dim[2]); // Target height for recognition model
    if (rec_input_h <= 0) {
        std::cerr << "infer_det: Recognition model input height is invalid: " << rec_input_h << std::endl;
        return std::nullopt; // Or use a default like 48
    }


    for (const auto &poly : polygons) {
        if (poly.points.size() < 4) continue; // Need at least 4 points for minAreaRect from DB output

        cv::RotatedRect rorect = cv::minAreaRect(poly.points);
        
        // Order points of the rotated rect:
        // points[0] = bottom-left, points[1] = top-left, points[2] = top-right, points[3] = bottom-right
        // This might vary. The goal is to get width along text lines, height perpendicular.
        // For text, width is usually longer than height.
        float w = rorect.size.width;
        float h = rorect.size.height;
        float angle = rorect.angle; // Angle in degrees. OpenCV: angle from horizontal axis to first edge (width).

        // Ensure width is the longer side for text lines
        if (w < h) {
            std::swap(w, h);
            angle += 90.0f; // Adjust angle if width/height are swapped
        }
        rorect.size = cv::Size2f(w,h);
        rorect.angle = angle;


        // Get the sub-image using warpAffine for better quality rotation
        cv::Mat M, rotated_img, cropped_img;
        M = cv::getRotationMatrix2D(rorect.center, rorect.angle, 1.0);
        
        // Calculate bounding box of the rotated rectangle to determine output size for warpAffine
        cv::Rect bbox = rorect.boundingRect(); // Bounding box in original image space
        // To avoid cutting off corners during rotation, warp the larger bounding box area
        // then crop. A simpler way is to use getRectSubPix if angle is small or image is padded.
        // The original code used a fixed-size canvas then rotated. Let's try a more direct approach for cropping.

        // Pad the ori_img if the rotated rect goes out of bounds to avoid warpAffine issues.
        // Or, more simply, use getRectSubPix which handles boundaries.
        try {
             cv::getRectSubPix(*this->ori_img, rorect.size, rorect.center, cropped_img);
        } catch(const cv::Exception& e) {
            std::cerr << "getRectSubPix failed: " << e.what() << " for poly centered at " << rorect.center << " size " << rorect.size <<std::endl;
            // Fallback or skip this polygon
            // Try to extract a simple bounding rect if rotated extraction fails.
            cv::Rect simple_brect = cv::boundingRect(poly.points);
            simple_brect.x = (std::max)(0, simple_brect.x);
            simple_brect.y = (std::max)(0, simple_brect.y);
            simple_brect.width = (std::min)(simple_brect.width, this->ori_img->cols - simple_brect.x);
            simple_brect.height = (std::min)(simple_brect.height, this->ori_img->rows - simple_brect.y);
            if (simple_brect.width > 0 && simple_brect.height > 0) {
                 cropped_img = (*this->ori_img)(simple_brect).clone();
                 // This image is not rotated, classification step might handle it or fail.
            } else {
                continue; // Skip if even simple bounding rect is invalid
            }
        }


        if (cropped_img.empty()) continue;

        // Resize to the fixed height for the recognition model, maintaining aspect ratio
        float aspect_ratio = static_cast<float>(cropped_img.cols) / static_cast<float>(cropped_img.rows);
        int rec_input_w = static_cast<int>(std::round(rec_input_h * aspect_ratio));
        rec_input_w = (std::max)(1, rec_input_w); // Ensure width is at least 1

        cv::Mat resized_for_rec;
        cv::resize(cropped_img, resized_for_rec, cv::Size(rec_input_w, rec_input_h));
        
        cropped_text_images.push_back(resized_for_rec);
    }

    if (cropped_text_images.empty()) {
        std::cout << "infer_det: No text images could be cropped." << std::endl;
        return std::nullopt;
    }
    
    // Original code reverses polygons and images. This implies an order dependency.
    // Let's maintain it. If it's not needed, this can be removed.
    std::reverse(cropped_text_images.begin(), cropped_text_images.end());
    std::reverse(this->polygons.begin(), this->polygons.end()); // Ensure polygons order matches images for postprocess

    return cropped_text_images;
}


std::optional<std::vector<cv::Mat>> PaddleOCR::infer_cls(std::vector<cv::Mat>& images) {
    if (images.empty() || input_nodes_cls.empty() || input_nodes_cls[0].dim.empty()) {
        std::cerr << "infer_cls: No images to classify or cls model input nodes not initialized." << std::endl;
        return std::nullopt;
    }

    int cls_input_h = static_cast<int>(this->input_nodes_cls[0].dim[2]);
    int cls_input_w = static_cast<int>(this->input_nodes_cls[0].dim[3]);

    // If model uses dynamic input size (-1), need a fixed size for batching or per-image inference.
    // The original code used a fixed size (e.g., 192x48 or 160x80).
    // Let's assume cls_input_h and cls_input_w are valid positive values from the loaded model.
    if (cls_input_h <=0) cls_input_h = 48; // Default fallback
    if (cls_input_w <=0) cls_input_w = 192; // Default fallback


    std::vector<cv::Mat> processed_cls_images(images.size());
    // The original code prepares two sets of images: `norm_images` for rec and `cls_images` for cls.
    // This function should return images ready for `infer_rec`.
    // Let's assume `norm_images` in the original code are the ones that are processed (rotated if needed) and returned.

    std::vector<cv::Mat> images_for_cls_model(images.size()); // These will be fed to cls model

    // The target size for recognition model (norm_images)
    // The original `infer_cls` also prepared `norm_images` for recognition.
    // It calculates a `max_w` based on recognition input height.
    if (input_nodes_rec.empty() || input_nodes_rec[0].dim.size() < 4) {
        std::cerr << "infer_cls: Recognition model input nodes not initialized for norm_images prep." << std::endl;
        return std::nullopt;
    }
    int rec_input_h = static_cast<int>(this->input_nodes_rec[0].dim[2]);
    if (rec_input_h <= 0) {
        std::cerr << "infer_cls: Rec model height invalid for norm_images prep." << std::endl;
        rec_input_h = 48; // Fallback
    }

    // int rec_max_w = 320; // Default max width for recognition preprocessing, can be dynamic
    // for (const auto& img : images) {
    //     if (img.rows == 0) continue; // Skip empty images
    //     float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
    //     // Calculate required width for rec_input_h, then round to nearest 32 multiple.
    //     int current_rec_w = static_cast<int>(std::ceil(rec_input_h * ratio / 32.0) * 32.0);
    //     rec_max_w = (std::max)(rec_max_w, current_rec_w);
    // }
    //  rec_max_w = (std::max)(rec_max_w, 32); // Ensure at least 32 width.


    // 动态计算rec_max_w，根据输入图像的宽高比
    int rec_max_w = 160; // 最小宽度
    std::cout << "Number of images to process: " << images.size() << std::endl;
    
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const auto& img = images[idx];
        if (img.rows == 0) continue;
        
        float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
        std::cout << "Image " << idx << ": size=" << img.cols << "x" << img.rows 
                  << ", original ratio=" << ratio;
        
        // 限制宽高比在合理范围内，避免过宽的图像影响识别
        float original_ratio = ratio;
        ratio = (std::min)(ratio, 8.0f);  // 降低最大宽高比限制为6
        ratio = (std::max)(ratio, 1.5f);  // 提高最小宽高比限制为1.5
        
        if (original_ratio != ratio) {
            std::cout << ", adjusted ratio=" << ratio;
        }
        
        int current_rec_w = static_cast<int>(std::ceil(rec_input_h * ratio / 32.0) * 32.0);
        rec_max_w = (std::max)(rec_max_w, current_rec_w);
        
        std::cout << ", calculated width=" << current_rec_w << std::endl;
    }
    
    rec_max_w = (std::min)(rec_max_w, 512); // 降低绝对上限
    rec_max_w = (std::max)(rec_max_w, 160);  // 确保最小宽度
    std::cout << "Final rec_max_w: " << rec_max_w << std::endl;



    // Parallel processing for image normalization
    tbb::parallel_for(0, (int)images.size(), 1, [&](int idx) {
        cv::Mat img_for_cls = MT::PaddingImg(images[idx], cv::Size(cls_input_w, cls_input_h));
        // Original cls preprocessing: convert to CV_32FC1 (is it grayscale?) then normalize.
        // If images[idx] is color, convert to gray first. Let's assume it should be color.
        img_for_cls.convertTo(img_for_cls, CV_32FC3, 1.0 / 255.0); // Assuming BGR from OpenCV
        cv::subtract(img_for_cls, cv::Scalar(0.5, 0.5, 0.5), img_for_cls);
        cv::divide(img_for_cls, cv::Scalar(0.5, 0.5, 0.5), img_for_cls);
        images_for_cls_model[idx] = img_for_cls;

        // Prepare images for recognition (these will be potentially rotated and returned)
        cv::Mat img_for_rec = MT::PaddingImg(images[idx], cv::Size(rec_max_w, rec_input_h));
        img_for_rec.convertTo(img_for_rec, CV_32FC3, 1.0 / 255.0);
        cv::subtract(img_for_rec, cv::Scalar(0.5, 0.5, 0.5), img_for_rec);
        cv::divide(img_for_rec, cv::Scalar(0.5, 0.5, 0.5), img_for_rec);
        processed_cls_images[idx] = img_for_rec; // These are the images to be returned, possibly rotated
    });

    cv::Mat blob = cv::dnn::blobFromImages(images_for_cls_model, 1.0, cv::Size(cls_input_w, cls_input_h), cv::Scalar(0,0,0), true, false);
    
    std::vector<Ort::Value> input_tensor_values;
    long long cls_batch_dim[] = {static_cast<long long>(images.size()), 3, static_cast<long long>(cls_input_h), static_cast<long long>(cls_input_w)};
    // Update input_nodes_cls[0].dim if it's used elsewhere after this, though blobFromImages uses its own size.
    // this->input_nodes_cls[0].dim = {cls_batch_dim[0], cls_batch_dim[1], cls_batch_dim[2], cls_batch_dim[3]};


    try {
        input_tensor_values.push_back(Ort::Value::CreateTensor<float>(
            *memory_info,
            blob.ptr<float>(),
            blob.total(),
            cls_batch_dim, // Use the actual dimensions of the blob
            4 // Number of dimensions
        ));
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_cls: CreateTensor failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto &node : this->input_nodes_cls) input_names_cstr.push_back(node.name);
    for (const auto &node : this->output_nodes_cls) output_names_cstr.push_back(node.name);
     if(input_names_cstr.empty() || output_names_cstr.empty()){
        std::cerr << "infer_cls: Input or output node names are empty." << std::endl;
        return std::nullopt;
    }


    std::vector<Ort::Value> output_tensor_values;
    try {
        output_tensor_values = this->session_cls->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensor_values.data(),
            input_tensor_values.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_cls: Session Run failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    if (output_tensor_values.empty()) {
        std::cerr << "infer_cls: No output from classification model." << std::endl;
        return std::nullopt;
    }

    // Output is typically [batch_size, num_classes] (e.g., [N, 2] for 0 vs 180 degrees)
    // And a second output for confidence scores perhaps, or just logits.
    // Original code assumes output_shape[0] is batch, output_shape[1] is 2 (scores for 0 and 180 deg).
    float* output_data = output_tensor_values[0].GetTensorMutableData<float>();
    const auto& output_shape = output_tensor_values[0].GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 2 || output_shape[1] != 2) { // Check for [N, 2] shape
        std::cerr << "infer_cls: Unexpected output shape from cls model. Expected [N, 2]." << std::endl;
        return std::nullopt;
    }

    for (int i = 0; i < output_shape[0]; ++i) {
        float score_0_deg = output_data[i * 2 + 0];
        float score_180_deg = output_data[i * 2 + 1];
        // If score for 180 is higher, rotate the corresponding image in `processed_cls_images`
        if (score_180_deg > score_0_deg) { // Original: output[pos] < output[pos+1] means 180 deg
            if (i < static_cast<int>(processed_cls_images.size())) { // Boundary check
                 cv::rotate(processed_cls_images[i], processed_cls_images[i], cv::ROTATE_180);
            }
        }
    }
    return processed_cls_images;
}


std::optional<std::vector<Ort::Value>> PaddleOCR::infer_rec(std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "infer_rec: No images to recognize." << std::endl;
        return std::nullopt;
    }
    if (input_nodes_rec.empty() || input_nodes_rec[0].dim.empty()) {
        std::cerr << "infer_rec: Rec model input nodes not initialized." << std::endl;
        return std::nullopt;
    }

    // All images should already be preprocessed (normalized, fixed height, padded width) by infer_cls
    // They should all have the same dimensions here.
    cv::Size rec_input_size = images[0].size(); // Assuming all images are same size
    for(const auto& img : images) {
        if (img.size() != rec_input_size) {
            std::cerr << "infer_rec: Mismatched image sizes for batching. Expected "
                      << rec_input_size << " got " << img.size() << std::endl;
            // Option: resize them all to a common size, or error out.
            // The logic in infer_cls (rec_max_w) should ensure they are same width.
            return std::nullopt;
        }
    }

    cv::Mat blob = cv::dnn::blobFromImages(images, 1.0, rec_input_size, cv::Scalar(0,0,0), true, false);

    std::vector<Ort::Value> input_tensor_values;
    long long rec_batch_dim[] = {static_cast<long long>(images.size()), 3, static_cast<long long>(rec_input_size.height), static_cast<long long>(rec_input_size.width)};
    // this->input_nodes_rec[0].dim = {rec_batch_dim[0], rec_batch_dim[1], rec_batch_dim[2], rec_batch_dim[3]};

    try {
        input_tensor_values.push_back(Ort::Value::CreateTensor<float>(
            *memory_info,
            blob.ptr<float>(),
            blob.total(),
            rec_batch_dim,
            4 
        ));
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_rec: CreateTensor failed: " << e.what() << std::endl;
        return std::nullopt;
    }

    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto &node : this->input_nodes_rec) input_names_cstr.push_back(node.name);
    for (const auto &node : this->output_nodes_rec) output_names_cstr.push_back(node.name);
    if(input_names_cstr.empty() || output_names_cstr.empty()){
        std::cerr << "infer_rec: Input or output node names are empty." << std::endl;
        return std::nullopt;
    }

    std::vector<Ort::Value> output_tensor_values;
    try {
        output_tensor_values = this->session_rec->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensor_values.data(),
            input_tensor_values.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "infer_rec: Session Run failed: " << e.what() << std::endl;
        return std::nullopt;
    }
    
    if (output_tensor_values.empty()) {
        std::cerr << "infer_rec: No output from recognition model." << std::endl;
        return std::nullopt;
    }

    return output_tensor_values;
}



// 加入YOLO检测结果
std::variant<bool, std::string> PaddleOCR::inference_from_custom_boxes( cv::Mat &image,
                                                                        const std::vector<PaddleOCR::YoloDetectionBox>& custom_boxes,
                                                                        std::vector<std::string>& texts) 
{


    std::cout << "=============================START=============================" << std::endl;


    if (image.empty()) return std::string("输入图像不能为空!");
    if (!this->is_inited) return std::string("模型未初始化! (请先调用 initialize，它也会加载 cls 和 rec 模型)");

    this->ori_img = &image; //
    this->polygons.clear(); //

    // 1. 将 YoloDetectionBox 转换为 PaddleOCR::Polygon 并填充 this->polygons
    for (const auto& y_box : custom_boxes) {
        PaddleOCR::Polygon poly; //
        poly.score = y_box.score; //
        poly.points.emplace_back(cv::Point2f(y_box.left, y_box.top)); //
        poly.points.emplace_back(cv::Point2f(y_box.right, y_box.top)); //
        poly.points.emplace_back(cv::Point2f(y_box.right, y_box.bottom)); //
        poly.points.emplace_back(cv::Point2f(y_box.left, y_box.bottom)); //
        this->polygons.push_back(poly); //
    }

    if (this->polygons.empty()) { //
        return std::string("没有从自定义检测框中解析出多边形区域。");
    }

    // 2. 根据 this->polygons 裁剪图像块
    std::vector<cv::Mat> cropped_text_images;
    if (input_nodes_rec.empty() || input_nodes_rec[0].dim.size() < 3 || this->input_nodes_rec[0].dim[2] <= 0) { //
        std::cerr << "错误: inference_from_custom_boxes: 识别模型的输入节点未正确初始化或高度无效。" << std::endl; //
        return std::string("识别模型参数错误，无法进行裁剪。");
    }
    int rec_input_h = static_cast<int>(this->input_nodes_rec[0].dim[2]); //

    for (const auto &poly : this->polygons) {
        if (poly.points.size() < 3) continue; // 至少需要3个点构成一个区域

        cv::RotatedRect rorect = cv::minAreaRect(poly.points); //

        // 标准化 rorect 的宽高和角度 (通常文本行宽度大于高度)
        float w = rorect.size.width; //
        float h = rorect.size.height; //
        float angle = rorect.angle; //
        if (w < h) { //
            std::swap(w, h); //
            angle += 90.0f; //
        }
        rorect.size = cv::Size2f(w, h); //
        rorect.angle = angle; //

        cv::Mat cropped_img_part;
        try {
            // 使用 getRectSubPix 从原始图像中提取旋转后的矩形区域
            cv::getRectSubPix(*this->ori_img, rorect.size, rorect.center, cropped_img_part); //
        } catch (const cv::Exception& e) {
            std::cerr << "警告: getRectSubPix 提取YOLO框失败: " << e.what()
                      << " (中心: " << rorect.center << ", 尺寸: " << rorect.size << ")" << std::endl;
            // 降级处理：尝试使用简单的边界框裁剪
            cv::Rect simple_brect = cv::boundingRect(poly.points); //
            simple_brect.x = (std::max)(0, simple_brect.x); //
            simple_brect.y = (std::max)(0, simple_brect.y); //
            simple_brect.width = (std::min)(simple_brect.width, this->ori_img->cols - simple_brect.x); //
            simple_brect.height = (std::min)(simple_brect.height, this->ori_img->rows - simple_brect.y); //
            if (simple_brect.width > 0 && simple_brect.height > 0) { //
                 cropped_img_part = (*this->ori_img)(simple_brect).clone(); //
            } else {
                std::cerr << "警告: 降级裁剪YOLO框也失败，跳过此框。" << std::endl;
                continue; //
            }
        }

        if (cropped_img_part.empty()) continue; //

        // 调整到识别模型所需的固定高度，保持宽高比
        float aspect_ratio = static_cast<float>(cropped_img_part.cols) / static_cast<float>(cropped_img_part.rows); //
        int rec_input_w = static_cast<int>(std::round(rec_input_h * aspect_ratio)); //
        rec_input_w = (std::max)(1, rec_input_w); // // 确保宽度至少为1

        cv::Mat resized_for_rec;
        cv::resize(cropped_img_part, resized_for_rec, cv::Size(rec_input_w, rec_input_h)); //
        cropped_text_images.push_back(resized_for_rec); //
    }

    if (cropped_text_images.empty()) { //
        return std::string("无法从自定义检测框中裁剪出任何有效的文本区域图像。");
    }

    // 3. 调用 cls 模型进行方向分类
    std::optional<std::vector<cv::Mat>> cls_result = this->infer_cls(cropped_text_images); //
    if (!cls_result.has_value() || cls_result.value().empty()) { //
        return std::string("方向分类(cls)阶段失败或未产生图像。");
    }

    // 4. 调用 rec 模型进行文本识别
    std::optional<std::vector<Ort::Value>> rec_result = this->infer_rec(cls_result.value()); //

    if (!rec_result.has_value() || rec_result.value().empty()) { //
        return std::string("文本识别(rec)阶段失败或未产生结果。");
    }

    // 5. 调用 postprocess进行后处理和结果绘制
    try {
        //this->postprocess(rec_result.value()); // postprocess会使用 this->polygons 和 this->ori_img //
        texts = this->postprocess(rec_result.value());
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "后处理阶段失败 (使用自定义检测框)! " << e.what();
        return oss.str();
    }
    for(int ii =0; ii<texts.size(); ii++){
        std::cout << "TEXT: " << texts[ii].c_str() << std::endl;
    }
       


    std::cout << "=============================END=============================" << std::endl;

    return true;
    
}
