cmake_minimum_required(VERSION 3.17)

# 项目信息
project(SideTrainNumberRec 
    VERSION 1.0.0
    DESCRIPTION "Side Train Number Recognition System"
    LANGUAGES CXX CUDA
)

# CMake 策略设置
cmake_policy(SET CMP0091 NEW)  

# 全局编译设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# 编译类型和优化设置
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MSVC特定设置
if(MSVC)
    add_compile_options(
        "$<$<C_COMPILER_ID:MSVC>:/utf-8>"
        "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
    )
    
    # Debug信息设置
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
endif()

# 源文件收集
set(PROJECT_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/udp/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/udp/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/LogManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog/LogManager.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocr/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocr/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/clipper2/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/clipper2/*.h"
)
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_FILES})

# 资源文件
set(RESOURCES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/icon/logo.qrc")
set(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/icon/resources.rc")

# 依赖库路径设置
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
set(DEPLOY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/yolo")
set(OpenCV_DIR "D:/Tools/opencv/opencv/build")
set(QT_PATH "D:/Tools/QT/5.14.2/msvc2017_64")
set(TRT_PATH "D:/Tools/TensorRT-8.6.1.6")
set(ONNXRUNTIME_DIR "D:/Tools/onnxruntime-win-x64-gpu-1.16.1")
set(Tbb_DIR "D:/Tools/Tbb")
# Qt设置
set(CMAKE_PREFIX_PATH "${QT_PATH}")
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

# CUDA配置
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89;86;75;70;61")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT配置
set(TRT_LIB_DIR "${TRT_PATH}/lib")
if(MSVC AND EXISTS "${TRT_LIB_DIR}/nvinfer_10.dll")
    set(TRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
else()
    set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
endif()

# 包含目录设置
include_directories(
    ${THIRD_PARTY_DIR}
    ${TRT_PATH}/include
    ${OpenCV_INCLUDE_DIRS}
    ${DEPLOY_PATH}
    ${ONNXRUNTIME_DIR}/include
    ${Tbb_DIR}/include
)

# 添加可执行文件
add_executable(${PROJECT_NAME}  WIN32
    ${SOURCE_FILES}
    ${RESOURCES_FILES}
    ${RESOURCE_FILES}
)

# 编译选项设置
function(set_target_compile_options target)
    if(MSVC)
        # 通用编译选项
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/O2 /Oi /GL /fp:fast>
            $<$<CONFIG:Debug>:/Od /Ob0 /RTC1>
        )

        # CUDA特定选项
        if(CMAKE_CUDA_COMPILER)
            target_compile_options(${target} PRIVATE
                $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2 -O3>
                $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Od -g -G>
            )
        endif()

        # 运行时库设置
        set_target_properties(${target} PROPERTIES
            MSVC_RUNTIME_LIBRARY "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        # Windows特定定义
        target_compile_definitions(${target} PRIVATE
            WIN32
            _WINSOCK_DEPRECATED_NO_WARNINGS
            _CRT_SECURE_NO_WARNINGS
        )
    endif()
endfunction()

# 设置目标属性
set_target_compile_options(${PROJECT_NAME})

# 链接设置
target_link_directories(${PROJECT_NAME} PRIVATE
    ${TRT_LIB_DIR}
    ${DEPLOY_PATH}/lib
    ${Tbb_Dir}/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Widgets
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TRT_LIBS}
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
    ${Tbb_DIR}/lib/*lib
    deploy
    $<$<BOOL:${WIN32}>:ws2_32>
)

# 输出配置
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/bin")
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "TrainNumberRec"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}"
)


