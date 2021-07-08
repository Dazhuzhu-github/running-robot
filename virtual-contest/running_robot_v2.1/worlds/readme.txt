说明
本环境为大赛虚拟标准赛C-P的2.1竞赛环境，替换了2.0版本控制器中的旧函数，以适配新的R2021a版本的webots。


注意
推荐使用python3.7版本，其它版本加载环境时可能会出现ImportError。在webots-工具-首选项-Python command中指向python3.7的路径。


调整赛道的方法
切换DEF Reset_Ruler Robot的controller：

controller"Rst_Ruler_random"：随机生成赛道，包含计时记分功能的裁判系统。每次重新仿真时会刷新赛道，并输出block、block_direc、block_type1、block_type11四项赛道生成的变量值。

controller"Rst_Ruler1"：生成一个固定赛道，包含计时记分功能的裁判系统，重新仿真时不会变更，以便参赛者训练测试。替换Rst_Ruler1.py开头的MAP_block等四项全局变量即可改变生成的关卡顺序。
