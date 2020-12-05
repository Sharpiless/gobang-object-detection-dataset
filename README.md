# gobang-object-detection-dataset

相关大作业进度的博客：

[【BIT大作业】人工智能+五子棋实战（一）棋子目标检测](https://blog.csdn.net/weixin_44936889/article/details/109862218)

[【BIT大作业】人工智能+五子棋实战（二）博弈搜索算法](https://blog.csdn.net/weixin_44936889/article/details/110380769)

北理BIT人工智能大作业，写脚本收集了黑/白棋子检测数据集

数据集为pygame游戏界面截图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201120194534365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70#pic_center)

这里使用PaddleX提供的YOLOv3目标检测算法。

同时由于目标比较好识别，所以使用轻量级的MobileNet作为主干网络。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020112019354172.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70#pic_center)

写代码不易，求个star~
