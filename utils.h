#pragma once
#include <iostream>
#include <string>
#include "rapidjson/document.h"         //https://github.com/Tencent/rapidjson
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"


struct MetaData {
public:
    double image_threshold;
    double pixel_threshold;
    double min;
    double max;
    int pred_image_size;
    int height;
    int width;
};


MetaData getJson(std::string json_path);