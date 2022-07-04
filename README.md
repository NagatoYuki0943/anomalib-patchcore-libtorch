# 说明

> anomalib patchcore 的 torchscript C++推理
>
> 对应版本 pytorch-1.11_cu113 libtorch-1.11_cu113 cuda113



# anomalib地址

> https://github.com/NagatoYuki0943/anomalib

# 环境注意事项

## 依赖

- cuda
- cudnn
- libtorch
  - 注意版本要和pytorch一致，包括cuda版本
  - 分为Release和Debug版本，自己的项目也要匹配对应版本
- opencv
- rapidjson 读取json文件 https://github.com/Tencent/rapidjson

## 环境变量

> `{opencv}`要替换为自己的对应的目录

```shell
{opencv}\build\x64\vc15\bin
{opencv}\build\x64\vc15\lib
{libtorch}\lib

# example
D:\AI\opencv\build\x64\vc15\bin
D:\AI\opencv\build\x64\vc15\lib
D:\AI\libtorch\lib
```

> 注意：cuda和cudnn的环境变量在安装时就自动写入，这里不需要手动添加。

## VS属性配置

### 调试

环境中添加`PATH={libtorch}\lib;%PATH%`

----

### C/C++

#### 常规

> 附加包含目录添加如下

```shell
{rapidjson}\include
{opencv}\build\include\opencv2
{opencv}\build\include
{libtorch}\include
{libtorch}\include\torch\csrc\api\include

# example
D:\AI\rapidjson\include
D:\AI\opencv\build\include\opencv2
D:\AI\opencv\build\include
D:\AI\libtorch\include
D:\AI\libtorch\include\torch\csrc\api\include
```

> SDL检查 `否`

#### 语言

> 符合模式 `否`

----

### 链接器

#### 常规

> 附加库目录

```shell
{opencv}\build\x64\vc15\lib
{libtorch}\lib

# example
D:\AI\opencv\build\x64\vc15\lib
D:\AI\libtorch\lib
```

#### 输入

> 附加依赖项，opencv和libtorch的lib
>
> 注意如果是Debug模式要将 `opencv_world455.lib` 更改为 `opencv_world455d.lib`,libtorch的Debug版本也有些许不同

```shell
# release
opencv_world460.lib
# debug
opencv_world460d.lib
```

`{libtorch}\lib`目录下所有的lib

```shell
asmjit.lib;
c10.lib;
c10_cuda.lib;
caffe2_nvrtc.lib;
clog.lib;
cpuinfo.lib;
dnnl.lib;
fbgemm.lib;
kineto.lib;
libprotobuf-lite.lib;
libprotobuf.lib;
libprotoc.lib;
pthreadpool.lib;
torch.lib;
torch_cuda.lib;
torch_cuda_cpp.lib;
torch_cuda_cu.lib;
torch_cpu.lib;
XNNPACK.lib;
```

#### 命令行

> 其他选项中添加如下，这样才能使用cuda
>
> 

```
/INCLUDE:?warp_size@cuda@at@@YAHXZ /INCLUDE:?_torch_cuda_cu_linker_symbol_op_cuda@native@at@@YA?AVTensor@2@AEBV32@@Z 
```

# 推理

> anomalib导出torchscript时可以选择cuda模式导出，根据网上的说法使用libtorch时调用显卡必须使用cuda导出模型，不过测试显示使用cpu导出的torchscript模型也能使用cuda推理

## anomalib导出

> https://github.com/NagatoYuki0943/anomalib
>
> 查看 anomalib 的 readme.md中的Export部分
>
> anomalib的export会导出模型和对应的超参数,超参数保存进了 `param.json` 中

## 推理

> 主文件在 `源.cpp`中
>
> 设置图片文件夹路径`imagedir`, 模型路径`model_path`和超参数路径`meta_path`进行推理.

# 问题

> anomalib中热力图的高斯模糊是在热力图和分数的标准化和放大热力图之前，使用的基于pytorch的conv2d，这里在放大图像之后，并且使用的opencv自带的高斯滤波，因为图片分辨率提高也因此增大了kernel_size，不过效果仍然没有python版本绘制的效果好



# opencv函数参考了mmdeploy的代码

`opencv_utils.cpp opencv_utils.h`

https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
