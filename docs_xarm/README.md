# xarm_openpi使用方法

## 收集数据
### 文件介绍
#### 收集数据类
record_and_transform/record.py 最开始用于采集原始数据的代码，需要改动保存路径，采集频率，文本命令
record_and_transform/record_randomOrder.py 改进后的采集数据代码，把桌面区域分成了4*4的区域，然后随机顺序遍历16个区域，每个区域内随机采样点
record_and_transform/bound_region_record.py 改进后的采集数据代码，用于采集边缘区域，沿着边缘区域进行采集，可以通过终端输入变换遍历边缘的方向（顺时针、逆时针）
record_and_transform/record_withRandomNoise.py 用于模拟夹爪yaw对准且在工件正上方的情况下，朝着yaw方向运动过头再纠正回来的过程
record_and_transform/record_with2situation.py 优化了上面的record_withRandomNoise.py，纠正过程的前置即从初始位置移动到上方附近偏移点的过程不计入data，尝试规避模型把运动过头后再调整的整个过程当做完整的正确过程来学习，同时加入了随机home点
record_and_transform/record_with2sit_wudi.py 在上述基础上做改进，工件放在一个地方不动的无敌模式，方便添加干扰物，按p暂停按c切换固定目标点
record_and_transform/bound_region_record_wudi.py 和 bound_region_record_withRandomHome.py 用于普通模式和无敌模式收集边缘数据
#### 转换数据类
record_and_transform/convert_realtime.py 用于转换原始数据到指定格式，可以实时转换，缺点是当删除某个错误的原始数据的时候不能自动删除转换后的数据，需要把整个转换后的数据目录都删除，然后重新运行，关节角度值
record_and_transform/convert_usingRelativeCartesian_pos.py 在上述convert_realtime.py 的基础上更改数据，存的state是相对于本次起始home点的末端笛卡尔相对位置zyxrpy和gripper，action是相对于上一帧的delta值
record_and_transform/convert_using_all_is_RelativeCartesian_pos.py stata同上，但是action是相对于本轮起始点的delta，与上述不同
#### 工具类
record_and_transform/update_instructions.py 用于批量更改采集到的原始数据data.jsonl中的命令，用于采集多工件的时候忘记更改文本命令
record_and_transform/verify_converted_data.py 用于检查转换后的数据格式，会保存在当前目录下的episode_0_numeric_data.json 文件中，方便检查state和action是否正确
record_and_transform/rename_and_move_episode.py 和 cp_episode.py 用于复制/移动另一个数据集中的episode数据到另一个位置
record_and_transform/sum_episode_instruction.py 用于统计每个工件的数据数量



### 使用方法
record_and_transform/record_randomOrder.py 采集适量区域数据
record_and_transform/bound_region_record.py 采集适量边缘数据
或者 record_and_transform/record_with2situation (record_withRandomNoise.py) 采集纠正数据
record_and_transform/convert_realtime.py 转换数据
tar -zcvf **.tar.gz ** 压缩转换后的数据，然后传到训练端服务器

最新版使用record_with2situation.py 采集，使用convert_using_all_is_RelativeCartesian_pos.py 转换，使用verify_converted_data.py 肉眼检验

## 推理
### 文件介绍
#### 推理类
xarm_inf_2cam_withcropedImage.py 用于推理，是最原始的运行一次只能推理一次的代码
images/inf_with_image.py 用于推理的同时可视化记录成功和失败点及其yaw角度，用点+箭头表示在图片中并保存
images/zero_latency.py在inf_with_image.py的基础上加入了降低延时的代码，起因是推理时inf_with_image.py看上去和最原始的xarm_inf_2cam_withcropedImage.py代码效果不同，更慢且成功率更低，故加入了强制清除相机缓存的操作。后来发现机械臂经常在目标上方停住（犹豫）后来把推理代码中EXECUTE_STEPS = 1 改成了EXECUTE_STEPS = 4，效果好了很多，不会出现上述在目标上方犹豫的问题的同时，成功率提高了很多
images/sample_point_test.py 用于生成采样点的json文件和可视化图片（紫色箭头，用来可视化随机生成的点分布是否合理），为后续统一模型推理做准备
images/read_json_inf.py 用于读取保存好的随机点json文件进行自动推理，同时生成可视化结果图（成功是绿色箭头，失败是红色箭头，蓝色是机械臂初始点）
images/read_json_relative_inf.py 在上述基础上，把逻辑改成了适配于修改后的转换后数据集（使用末端笛卡尔位姿相对值 相对于本轮起始点），同时还有可视化图表显示模型输出的xyz趋势images/live_debug_status.png
#### 工具类
images/check_model.py 用于检查模型推理输出和真实值是否差距过大，需要先根据数据集生成episode_0_numeric_data，无需连接机械臂
images/sample_point_test.py 用于生成测试点

### 使用方法
现阶段运行 images/read_json_relative_inf.py 用于推理，在推理前需要运行sample_point_test.py生成推理点集test_points.json和test_points_map.png, 如果有test_progress.json的话需要根据需要决定是否删除此文件（此文件保存着test_points.json还没有测试的点，如果上一次测试完但是没有删除test_progress.json文件的话，新一轮测试推理没有目标点，会直接结束）

## 解决屏幕部分区域无法点击的问题
sudo pkill gnome-session-b
