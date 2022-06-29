#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>
#include <fstream>
#include <iostream>
#include<string>
#include<vector>
#include"opencv_utils.h"
#include"utils.h"
using namespace cv;
using namespace std;


/**
 * 读取图像并预处理
 */
cv::Mat readImage(String path) {
	cv::Mat image = cv::imread(path);				// BGR
	return image;
}


/**
 * 图片预处理
 */
torch::Tensor preProcess(cv::Mat& image, MetaData& meta) {
	//保存原图信息
	meta.height = image.rows;
	meta.width = image.cols;

	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);	// BGR2RGB

	//缩放
	cv::Mat res = Resize(image, meta.pred_image_size, meta.pred_image_size, "bilinear");

	//归一化
	res = Divide(res);

	//转化为tensor
	torch::Tensor x = torch::from_blob(res.data, { 1, res.rows, res.cols, 3 });
	x = x.permute({ 0, 3, 1, 2 });
	//x = x.div(torch::full({ 1, 3, 512, 512 }, 255.0));

	//标准化
	auto mean = vector<double>{ 0.485, 0.456, 0.406 };
	auto std  = vector<double>{ 0.229, 0.224, 0.225 };
	x = torch::data::transforms::Normalize<>(mean, std)(x);
	//cout << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << endl; // 1, 3, 512, 512
	return x;
}


/**
 * 读取模型
 */
torch::jit::Module loadTorchScript(String path) {
	return torch::jit::load(path);
}


/**
 * 推理
 */
vector<torch::Tensor> inference(torch::jit::Module& model, torch::Tensor& x) {
	//设置输入值，或者直接使用 {} 包裹数据
	//vector<torch::jit::IValue> input;
	//input.push_back(x);
	//x = torch::randn({ 1, 3, 512, 512 });
	torch::jit::IValue y = model.forward({ x });
	//多个返回值的提取方式 toTuple() toList()
	torch::Tensor anomaly_map = y.toTuple()->elements()[0].toTensor();
	torch::Tensor pred_score = y.toTuple()->elements()[1].toTensor();
	//cout << pred_score << endl;

	return vector<torch::Tensor>{anomaly_map, pred_score};
}


/**
 * 分别标准化热力图和得分
 */
torch::Tensor normalize(torch::Tensor& targets, double threshold, double max_val, double min_val) {
	auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
	normalized = torch::minimum(normalized, torch::tensor(1));
	normalized = torch::maximum(normalized, torch::tensor(0));
	return normalized;
}


/**
 * opencv标准化热力图
 */
cv::Mat cvNormalize(cv::Mat& targets, double threshold, double max_val, double min_val) {
	auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
	return normalized;
}


/**
 * 后处理部分
 */
vector<torch::Tensor> postProcess(torch::Tensor& anomaly_map, torch::Tensor& pred_score, MetaData& meta) {
	//anomaly_map.squeeze_();	
	
	//高斯滤波应该在这里,放到了superimposeAnomalyMap中，增大了kernel size适应更大的图像

	//标准化热力图和得分
	anomaly_map = normalize(anomaly_map, meta.pixel_threshold, meta.max, meta.min);
	pred_score  = normalize(pred_score, meta.image_threshold, meta.max, meta.min);
	//cout << "pred_score:" << pred_score << endl;

	//还原到原图尺寸
	anomaly_map = torch::nn::functional::interpolate(anomaly_map,
		torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{meta.height, meta.width})
		.mode(torch::kBilinear).align_corners(true));

	anomaly_map.squeeze_();	// 1,1,512,512 -> 512,512
	//cout << anomaly_map.size(0) << ", " << anomaly_map.size(1) << endl; //2711, 5351

	return vector<torch::Tensor>{anomaly_map, pred_score};
}


/**
 * 叠加图片
 */
cv::Mat superimposeAnomalyMap(torch::Tensor& anomaly_map, MetaData& meta, cv::Mat& origin_image, int kernel_rate) {
	cv::Mat anomaly(cv::Size(meta.width, meta.height), CV_32F, anomaly_map.data_ptr());
	cout << anomaly.rows << ", " << anomaly.cols << ", " << anomaly.channels() << endl;

	//高斯滤波，应该放在post_process前面，这里放在了后面，因为要使用opencv 增大了kernel size适应更大的图像
	int sigma = int(4 * kernel_rate);
	int kernel_size = 2 * int(4.0 * sigma + 0.5) + 1;
	cv::GaussianBlur(anomaly, anomaly, { kernel_size, kernel_size }, sigma, sigma);

	// 归一化，图片效果更明显
	//// anomaly_map = (anomaly - anomaly.min()) / np.ptp(anomaly) np.ptp()函数实现的功能等同于np.max(array) - np.min(array)
	double minValue, maxValue;    // 最大值，最小值
	cv::minMaxLoc(anomaly, &minValue, &maxValue);
	anomaly = (anomaly - minValue) / (maxValue - minValue);

	//转换为整形
	anomaly = anomaly * 255;
	anomaly.convertTo(anomaly, CV_8U);

	//单通道转化为3通道
	cv::applyColorMap(anomaly, anomaly, cv::COLORMAP_JET);

	//RGB2BGR
	//cv::cvtColor(anomaly, anomaly, cv::COLOR_RGB2BGR);

	//合并原图和热力图
	cv::Mat result;
	cv::addWeighted(anomaly, 0.4, origin_image, 0.6, 0, result);

	//cv::imwrite("anomaly_map.png", anomaly);
	//cv::imwrite("result.png", result);
	return result;
}


/**
 * 给图片添加标签
 */
cv::Mat addLabel(cv::Mat& mixed_image, float score, int font = cv::FONT_HERSHEY_PLAIN) {
	string text = "Confidence Score " + to_string(score);
	int font_size = mixed_image.cols / 1024 + 1;
	int baseline = 0;
	int thickness = font_size / 2;
	cv::Size textsize = cv::getTextSize(text, font, font_size, thickness, &baseline);
	//cout << textsize << endl;	//[1627 x 65]
	
	//背景
	cv::rectangle(mixed_image, cv::Point(0, 0), cv::Point(textsize.width+10, textsize.height+10), Scalar(225, 252, 134), FILLED);

	//添加文字
	cv::putText(mixed_image, text, cv::Point(0, textsize.height + 10), font, font_size, cv::Scalar(0, 0, 0), thickness);

	//cv::imwrite("result.jpg", prediction);
	return mixed_image;
}


/**
 * 保存图片和分数 
 */
void save(float score, cv::Mat& mixed_image_with_label) {
	//打印并保存得分
	cout << "pred_score:" << score << endl;
	ofstream ofs;
	ofs.open("result.txt", ios::out);
	ofs << score;
	ofs.close();

	cv::imwrite("result.jpg", mixed_image_with_label);
}


/**
 * 预测过程 
 */
void predict() {
	cout << "cuda is_available:" << torch::cuda::is_available() << endl;
	cout << "cudnn cudnn_is_available:" << torch::cuda::cudnn_is_available() << endl;

	auto path = "D:\\ai\\code\\abnormal\\anomalib\\results\\export\\512-0.1\\output.torchscript";
	auto img_path = "D:\\ai\\code\\abnormal\\anomalib\\datasets\\some\\1.abnormal\\OriginImage_20220526_113206_Cam1_6_crop.jpg";

	//读取meta
	auto meta = getJson("D:\\ai\\code\\abnormal\\anomalib\\results\\export\\512-0.1\\param.json");
	cout << meta.image_threshold << " " << meta.pixel_threshold << " " << meta.min << " " << meta.max << " "
		<< meta.pred_image_size << " " << meta.height << " " << meta.width << endl;
	// 0.92665 0.92665 0.000141821 1.70372 512 2711 5351

	//读取图片
	auto image = readImage(img_path);
	//图片预处理
	auto x = preProcess(image, meta);
	//读取模型
	auto model = loadTorchScript(path);
	//推理
	auto result = inference(model, x);
	//后处理
	result = postProcess(result[0], result[1], meta);
	//混合原图和热力图
	auto kernel_rate = int(meta.height / meta.pred_image_size);
	auto mixed_image = superimposeAnomalyMap(result[0], meta, image, kernel_rate);
	//分数转化为float
	auto score = result[1].item<float>();	// at::Tensor -> float
	//添加标签
	auto mixed_image_with_label = addLabel(mixed_image, score);
	//保存图片和分数 
	save(score, mixed_image_with_label);
}


int main() {
	predict();

	return 0;
}