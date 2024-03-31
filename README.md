# PV-Wallpaper-Extractor
  
**WIP**  

## 目标

将游戏/动画等 PV 内容提取所有帧后再将不同内容的帧软链接到不同的文件夹中。  
当前方法是花里胡哨的 读图片——特征提取——特征点聚类——根据聚类结果划分图片类别 的流程，复杂度爆炸。事实上也不如现有根据图片哈希分类的方法，但是花里胡哨。  

当前测试效果在约 2380 张 3840x2160 WEBP 图片上大约需要 337s，没有将不同类划分在一起的情况，但是存在将同一类图片划分为多类的情况。
  
## TODO
- [ ] 自动完成 pv 提取帧
- [ ] 采用正经滑动窗口？等方法实现图片的分类
- [ ] 对不同分类图片评价清晰度