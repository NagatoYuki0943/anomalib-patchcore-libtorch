#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>
#include <fstream>
#include <iostream>
#include<string>
#include <utility>
#include<vector>
#include"opencv_utils.h"
#include"utils.h"
#include <Windows.h>
#include <direct.h>
#include<io.h>

using namespace cv;
using namespace std;


/*
 * 获取文件夹下全部图片的绝对路径
 *
 * @param path		    图片文件夹路径
 * @return result		全部图片绝对路径列表
 */
vector<cv::String> getImages(string& path) {
    vector<cv::String> result;
    cv::glob(path, result, false);
    //for (auto name : result) {
    //	cout << name << endl;
    //}
    //D:/ai/code/abnormal/anomalib/datasets/some/1.abnormal\OriginImage_20220526_113441_Cam1_47_crop.jpg
    return result;
}


/**
 * 创建文件夹
 *
 *  @param dir	路径
 */
void createDir(string& dir) {
    if (access(dir.c_str(), 0) == -1)
    {
        cout << dir << " is not existing" << endl;
        cout << "now make it" << endl;
#ifdef WIN32
        int flag = mkdir(dir.c_str());
#endif
#ifdef linux
        int flag = mkdir(dir.c_str(), 0777);
#endif
        if (flag == 0)
        {
            cout << "make successfully" << endl;
        }
        else {
            cout << "make errorly" << endl;
        }
    }
}


/**
 * 读取图像
 *
 * @param path	图片路径
 * @return		图片
 */
cv::Mat readImage(string& path) {
    return cv::imread(path);				// BGR
}


/**
 * 图片预处理
 *
 * @param path	图片路径
 * @param meta  超参数,这里存放原图的宽高
 * @return x	tensor类型的图片
 */
torch::Tensor preProcess(cv::Mat& image, MetaData& meta) {
    //保存原图宽高
    meta.height = image.rows;
    meta.width = image.cols;

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);	// BGR2RGB

    //缩放
    cv::Mat res = Resize(image, meta.pred_image_height, meta.pred_image_width, "bilinear");

    //归一化
    res = Divide(res);

    //转化为tensor
    torch::Tensor x = torch::from_blob(res.data, { 1, res.rows, res.cols, 3 });
    x = x.permute({ 0, 3, 1, 2 });
    //x = x.div(torch::full({ 1, 3, 512, 512 }, 255.0));

    //标准化
    auto mean = vector<double>{ 0.485, 0.456, 0.406 };
    auto std = vector<double>{ 0.229, 0.224, 0.225 };
    x = torch::data::transforms::Normalize<>(mean, std)(x);
    //cout << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << endl; // 1, 3, 512, 512
    return x;
}


/**
 * 读取模型
 *
 * @param path	模型路径
 * @return		模型
 */
torch::jit::script::Module loadTorchScript(string& path) {
    return torch::jit::load(path);;
}


/**
 * 推理
 *
 * @param model		热力图或者得分
 * @param x			阈值,meta中的参数
 * @return			热力图和得分
 */
vector<torch::Tensor> inference(torch::jit::Module& model, torch::Tensor& x) {
    //设置输入值，或者直接使用 {} 包裹数据
    //vector<torch::jit::IValue> input;
    //input.push_back(x);
    //x = torch::randn({ 1, 3, 512, 512 });
    auto y = model.forward({ x });
    //多个返回值的提取方式 toTuple() toList()
    torch::Tensor anomaly_map = y.toTuple()->elements()[0].toTensor();
    torch::Tensor pred_score = y.toTuple()->elements()[1].toTensor();
    //cout << pred_score << endl;

    return vector<torch::Tensor>{anomaly_map, pred_score};
}


/**
 * 分别标准化热力图和得分
 *
 * @param targets		热力图或者得分
 * @param threshold		阈值,meta中的参数
 * @param max_val		最大值,meta中的参数
 * @param min_val		最小值,meta中的参数
 * @return normalized	经过标准化后的结果
 */
torch::Tensor normalize(torch::Tensor& targets, double threshold, double max_val, double min_val) {
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    normalized = torch::minimum(normalized, torch::tensor(1));
    normalized = torch::maximum(normalized, torch::tensor(0));
    return normalized;
}


/**
 * opencv标准化热力图
 *
 * @param targets		热力图
 * @param threshold		阈值,meta中的参数
 * @param max_val		最大值,meta中的参数
 * @param min_val		最小值,meta中的参数
 * @return normalized	经过标准化后的结果
 */
cv::Mat cvNormalize(cv::Mat& targets, double threshold, double max_val, double min_val) {
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    return normalized;
}


/**
 * 后处理部分,标准化热力图和得分,还原热力图到原图尺寸
 *
 * @param anomaly_map   未经过标准化的热力图
 * @param pred_score    未经过标准化的得分
 * @return result		热力图和得分vector
 */
vector<torch::Tensor> postProcess(torch::Tensor& anomaly_map, torch::Tensor& pred_score, MetaData& meta) {
    //高斯滤波应该在这里,放到了superimposeAnomalyMap中，增大了kernel size适应更大的图像  TODO:实现卷积完成高斯滤波

    //标准化热力图和得分
    anomaly_map = normalize(anomaly_map, meta.pixel_threshold, meta.max, meta.min);
    pred_score = normalize(pred_score, meta.image_threshold, meta.max, meta.min);
    //cout << "pred_score:" << pred_score << endl;

    //还原到原图尺寸
    anomaly_map = torch::nn::functional::interpolate(anomaly_map,
        torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{meta.height, meta.width})
        .mode(torch::kBilinear).align_corners(true));

    anomaly_map.squeeze_();
    //cout << anomaly_map.size(0) << ", " << anomaly_map.size(1) << endl; //2711, 5351

    return vector<torch::Tensor>{anomaly_map, pred_score};
}


/**
 * 叠加图片
 *
 * @param anomaly_map   混合后的图片
 * @param meta	        超参数
 * @param origin_image  原始图片
 * @param kernel_rate	高斯滤波kernel_size缩放比例
 * @return result		叠加后的图像
 */
cv::Mat superimposeAnomalyMap(torch::Tensor& anomaly_map, MetaData& meta, cv::Mat& origin_image, int kernel_rate) {
    anomaly_map = anomaly_map.to(torch::kCPU);		//转移到CPU，不然报错
    cv::Mat anomaly(cv::Size(meta.width, meta.height), CV_32F, anomaly_map.data_ptr());
    //cout << anomaly.rows << ", " << anomaly.cols << ", " << anomaly.channels() << endl;

    //高斯滤波，应该放在post_process前面，这里放在了后面，因为要使用opencv 增大了kernel size适应更大的图像,效果仍然比放在前面效果差,因为之前实在缩小的图像处理的
    int sigma = int(4 * kernel_rate);
    int kernel_size = 2 * int(4.0 * sigma + 0.5) + 1;
    cv::GaussianBlur(anomaly, anomaly, { kernel_size, kernel_size }, sigma, sigma);

    // 归一化，图片效果更明显
    //python代码： anomaly_map = (anomaly - anomaly.min()) / np.ptp(anomaly) np.ptp()函数实现的功能等同于np.max(array) - np.min(array)
    double minValue, maxValue;    // 最大值，最小值
    cv::minMaxLoc(anomaly, &minValue, &maxValue);
    anomaly = (anomaly - minValue) / (maxValue - minValue);

    //转换为整形
    //anomaly = anomaly * 255;
    //anomaly.convertTo(anomaly, CV_8U);
    cv::normalize(anomaly, anomaly, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8U);

    //单通道转化为3通道
    cv::applyColorMap(anomaly, anomaly, cv::COLORMAP_JET);

    //RGB2BGR
    //cv::cvtColor(anomaly, anomaly, cv::COLOR_RGB2BGR);

    //合并原图和热力图
    cv::Mat result;
    cv::addWeighted(anomaly, 0.4, origin_image, 0.6, 0, result);

    return result;
}


/**
 * 给图片添加标签
 *
 * @param mixed_image   混合后的图片
 * @param score			得分
 * @param font			字体
 * @return mixed_image  添加标签的图像
 */
cv::Mat addLabel(cv::Mat& mixed_image, float score, int font = cv::FONT_HERSHEY_PLAIN) {
    string text = "Confidence Score " + to_string(score);
    int font_size = mixed_image.cols / 1024 + 1;
    int baseline = 0;
    int thickness = font_size / 2;
    cv::Size textsize = cv::getTextSize(text, font, font_size, thickness, &baseline);
    //cout << textsize << endl;	//[1627 x 65]

    //背景
    cv::rectangle(mixed_image, cv::Point(0, 0), cv::Point(textsize.width + 10, textsize.height + 10), Scalar(225, 252, 134), FILLED);

    //添加文字
    cv::putText(mixed_image, text, cv::Point(0, textsize.height + 10), font, font_size, cv::Scalar(0, 0, 0), thickness);

    //cv::imwrite("result.jpg", prediction);
    return mixed_image;
}


/**
 * 保存图片和分数
 *
 * @param score		得分
 * @param mixed_image_with_label 混合后的图片
 * @param img_path  输入图片的路径
 * @param save_dir  保存的路径
 */
void save(float score, cv::Mat& mixed_image_with_label, cv::String& img_path, string& save_dir) {
    //获取图片文件名
    auto start = img_path.rfind('\\');
    auto end = img_path.substr(start + 1).rfind('.');
    //cout << start << ", " << end << endl;	//53, 39
    auto image_name = img_path.substr(start + 1).substr(0, end);  //OriginImage_20220526_113036_Cam1_1_crop

    //打印并保存得分
    cout << "pred_score: " << score << endl;
    ofstream ofs;
    ofs.open(save_dir + "/" + image_name + ".txt", ios::out);
    ofs << score;
    ofs.close();

    cv::imwrite(save_dir + "/" + image_name + ".jpg", mixed_image_with_label);
}


/**
 * 预测过程
 *
 * @param img_list   图片列表
 * @param model_path 模型路径
 * @param meta_path  模型超参数路径
 */
void predict(vector<cv::String>& img_list, string& model_path, string& meta_path, string& save_dir) {
    //要在链接器命令行中添加 "/INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:?_torch_cuda_cu_linker_symbol_op_cuda@native@at@@YA?AVTensor@2@AEBV32@@Z "
    //refer https://github.com/pytorch/pytorch/issues/72396#issuecomment-1032712081
    auto cuda = torch::cuda::is_available();

    if (cuda) {
        cout << "cuda inference" << endl;
    }
    else {
        cout << "cpu inference" << endl;
    }

    //读取meta
    auto meta = getJson(std::move(meta_path));
    cout << meta.image_threshold << " " << meta.pixel_threshold << " " << meta.min << " " << meta.max << " "
        << meta.pred_image_height << " " << meta.pred_image_width << " " << meta.height << " " << meta.width << endl;
    // 0.92665 0.92665 0.000141821 1.70372 512 2711 5351
    //读取模型
    auto model = loadTorchScript(model_path);
    if (cuda) {
        model.to(torch::kCUDA);
    }

    //创建存储结果的文件夹
    createDir(save_dir);

    //循环图片列表
    for (auto& img_path : img_list) {
        //读取图片
        auto image = readImage(img_path);
        //图片预处理
        auto x = preProcess(image, meta);
        //使用cuda
        if (cuda) {
            x = x.to(torch::kCUDA);
        }
        //推理
        auto result = inference(model, x);
        //后处理
        result = postProcess(result[0], result[1], meta);
        //混合原图和热力图
        auto kernel_rate = int(meta.height / (meta.pred_image_height + meta.pred_image_width) * 2);
        auto mixed_image = superimposeAnomalyMap(result[0], meta, image, kernel_rate);
        //分数转化为float
        auto score = result[1].item<float>();	// at::Tensor -> float
        //auto score = result[1].item().toFloat();
        //添加标签
        auto mixed_image_with_label = addLabel(mixed_image, score);
        //保存图片和分数
        save(score, mixed_image_with_label, img_path, save_dir);
    }
}


/**
 * 测试cuda是否可以使用
 */
void testCuda() {
    LoadLibraryA("torch_cuda.dll");
    try {
        std::cout << torch::cuda::is_available() << std::endl;
        torch::Tensor tensor = torch::tensor({ -1, 1 }, torch::kCUDA);
        cout << tensor << endl;
    }
    catch (exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}


int main() {
	//testCuda();
	string imagedir   = "D:/ai/code/abnormal/anomalib/datasets/some/1.abnormal";
    //使用cpu设备导出的模型既能使用cpu也能使用cuda推理,而使用cuda导出的模型只能使用cuda推理
    string model_path = "./weights/512-0.1/output.torchscript";
    string meta_path  = "./weights/512-0.1/param.json";
    string save_dir   = "./result";

    auto image_list = getImages(imagedir);

    predict(image_list, model_path, meta_path, save_dir);

	return 0;
}