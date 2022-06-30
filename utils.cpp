#include "utils.h"


MetaData getJson(std::string json_path) {
    FILE* fp = fopen(json_path.c_str(), "r");
    //if (fp == NULL)
    //{
    //    std::cerr << "File does not exists!" << std::endl;
    //    return 0;
    //}
    char readBuffer[1000];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document doc;
    doc.ParseStream(is);
    fclose(fp);

    double image_threshold = doc["image_threshold"].GetDouble();
    double pixel_threshold = doc["pixel_threshold"].GetDouble();
    double min             = doc["min"].GetDouble();
    double max             = doc["max"].GetDouble();
    int pred_image_size    = doc["pred_image_size"].GetInt();

	//std::cout << image_threshold << std::endl;
	//std::cout << pixel_threshold << std::endl;
	//std::cout << min << std::endl;
	//std::cout << max << std::endl;
	//std::cout << pred_image_size << std::endl;

    MetaData meta;
    meta.image_threshold = image_threshold;
    meta.pixel_threshold = pixel_threshold;
    meta.min = min;
    meta.max = max;
    meta.pred_image_size = pred_image_size;

    return meta;
}