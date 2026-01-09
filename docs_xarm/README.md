# xarm_openpi使用方法

## 收集数据
### 文件介绍
record_and_transform/record.py 最开始用于采集原始数据的代码，需要改动保存路径，采集频率，文本命令
record_and_transform/record_randomOrder.py 改进后的采集数据代码，把桌面区域分成了4*4的区域，然后随机顺序遍历16个区域，每个区域内随机采样点
record_and_transform/bound_region_record.py 改进后的采集数据代码，用于采集边缘区域，沿着边缘区域进行采集，可以通过终端输入变换遍历边缘的方向（顺时针、逆时针）
record_and_transform/update_instructions.py 用于批量更改采集到的原始数据data.jsonl中的命令，用于采集多工件的时候忘记更改文本命令
record_and_transform/convert_realtime.py 用于转换原始数据到指定格式，可以实时转换，缺点是当删除某个错误的原始数据的时候不能自动删除转换后的数据，需要把整个转换后的数据目录都删除，然后重新运行
record_and_transform/record_withRandomNoise.py 用于模拟夹爪yaw对准且在工件正上方的情况下，朝着yaw方向运动过头再纠正回来的过程

### 使用方法
record_and_transform/record_randomOrder.py 采集适量区域数据
record_and_transform/bound_region_record.py 采集适量边缘数据
或者 record_and_transform/record_withRandomNoise.py 采集纠正数据
record_and_transform/convert_realtime.py 转换数据
tar -zcvf **.tar.gz ** 压缩转换后的数据，然后传到训练端服务器

## 推理
### 文件介绍
xarm_inf_2cam_withcropedImage.py 用于推理，是最原始的运行一次只能推理一次的代码
images/inf_with_image.py 用于推理的同时可视化记录成功和失败点及其yaw角度，用点+箭头表示在图片中并保存
images/zero_latency.py在inf_with_image.py的基础上加入了降低延时的代码，起因是推理时inf_with_image.py看上去和最原始的xarm_inf_2cam_withcropedImage.py代码效果不同，更慢且成功率更低，故加入了强制清除相机缓存的操作。后来发现机械臂经常在目标上方停住（犹豫）后来把推理代码中EXECUTE_STEPS = 1 改成了EXECUTE_STEPS = 4，效果好了很多，不会出现上述在目标上方犹豫的问题的同时，成功率提高了很多
images/sample_point_test.py 用于生成采样点的json文件和可视化图片（紫色箭头，用来可视化随机生成的点分布是否合理），为后续统一模型推理做准备
images/read_json_inf.py 用于读取保存好的随机点json文件进行自动推理，同时生成可视化结果图（成功是绿色箭头，失败是红色箭头，蓝色是机械臂初始点）

### 使用方法
现阶段运行 images/read_json_inf.py 用于推理