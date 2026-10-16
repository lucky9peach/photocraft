# Beauty 需求统计与移植路线图（PortraitLab → PhotoCraft）

> 状态：Stage 1 已落地（`crates/beauty` 参数/预设/蒙版数学 + 11 测试）。
> 本文档是需求总表 + 架构决策记录。分支：`beauty/plugin-port`。

## 0. 结论先行：为什么不是纯 WASM 插件

PhotoCraft 的插件 ABI v1 是**沙箱 WASM 滤镜**（`docs/plugins.md`）：无 FS/网络/时钟、
按 ~4MiB tile 并行跑、band 间无状态、无 SIMD、解释执行慢 10-100 倍。
这决定了：

- **像素局部效果**（提亮/饱和/锐化/小半径平滑）适合 WASM：manifest 参数自动生成
  滑块对话框 + 实时预览，正好是“拖进度条”的体验。
- **人脸/人体检测（ML 推理）与大幅几何变形**（瘦脸/瘦腿/长腿动辄数百像素位移、
  需要整图 + 跨 tile 状态）在 WASM 沙箱里不可行。
- 因此：**美颜主体做 fork 内的原生 `beauty` crate（L2）**，走 engine command +
  原生 UI（egui 滑块）；像素级小效果后续可再导出一份 WASM 滤镜版本。
  全部新代码隔离在新文件/新 crate，主分支保持与上游同步友好。

## 1. 需求总表

### 1.1 已有功能（PortraitLab Python 版，需移植，算法已验证）

| # | 功能 | 参数 | 说明 |
|---|---|---|---|
| F1 | 边缘保留磨皮 | skin_smooth 0-100 + texture_keep | bilateral + 细节回填，防塑料感 |
| F2 | 肤色均匀 | skin_tone_uniform | Lab ab 通道局部均值 |
| F3 | 瑕疵淡化 | blemish_fade | 双尺度中值背景 + 斑点检测 |
| F4 | 眼睛放大 | eye_enlarge | 余弦衰减 warp，可关 |
| F5 | 下颌瘦脸 | jaw_slim | 同上 |
| F6 | 鼻部微调 | nose_refine | 水平收窄 warp |
| F7 | 局部提亮 | brighten | 脸区 gamma |
| F8 | 饱和度 | saturation -100-100 | 脸区 HSV |
| F9 | 眼部细节 | eye_detail | CLAHE + 微锐化 |
| F10 | 锐化 | sharpen | 脸区 unsharp mask |
| F11 | 人脸多选/单选 | face selection | 多脸分别处理 |
| F12 | 预设 | Natural/Cosplay/Editorial + 自定义 | 纯参数集合 |
| F13 | 非破坏性管线 | — | 恒从原图渲染；撤销=参数快照 |

### 1.2 新增：脸型五官（本轮用户需求）

| # | 功能 | 关键点/模型依赖 | 备注 |
|---|---|---|---|
| N1 | 瓜子脸/V 脸 | 脸 oval 轮廓点（jawline  dense） | 自然优先：小步多点位移场，限制最大位移比 |
| N2 | 鼻子大小（鼻翼/鼻头） | 鼻部 landmarks | 缩放 + 宽度双滑块？先做统一 size |
| N3 | 嘴角（微笑唇/嘴角上扬） | 唇部 corners + 边缘 | 小角度旋转 warp，防牙齿变形 |
| N4 | 眼睛微调增强 | iris 中心 + 眼轮廓 | 复用 F4，半径自适应 |
| N5 | 苹果肌/法令纹淡化 | 脸颊分区 + 鼻唇沟线 | 局部平滑 + 提亮，需分割辅助 |

### 1.3 新增：美体（本轮用户需求，优先级最高）

| # | 功能 | 模型依赖 | 备注 |
|---|---|---|---|
| B1 | 瘦身/瘦腰 | 人体分割 + 关节点 | 躯干横向收窄，背景不动 |
| B2 | 瘦腿 | 人体分割 + 腿部关节点（hip/knee/ankle） | 双腿分别收窄 |
| B3 | 长腿/增高 | 人体分割 + 关节点 | 纵向拉伸腿段，**背景重建**（inpaint 边缘过渡） |
| B4 | 全身肤色提亮统一 | 皮肤分割（脸+身体） | 沿用 F7 逻辑扩展到身体皮肤蒙版 |
| B5 | 头身比微调 | 关节点 | 保守范围 |

共同约束：所有美体 warp 必须配人物分割蒙版 + 边缘羽化，背景像素零改动（除 B3
拉伸接缝处的局部 inpaint，该接缝必须可视化可调）。

### 1.4 交互需求

- 全部自动识别（脸/身体/关节点/分割），用户只拖滑块。
- 滑块实时预览；长腿等重操作可接受“松开后渲染”（debounce）。
- 每个效果独立开关；几何类默认保守/关闭。

## 2. 模型选型（待核验许可证后定）

方向：ONNX + 纯 Rust 推理（`tract`，与 100% Rust / wasm32 兼容规则一致；
若算子覆盖不足再评估 `ort` + cfg(not(wasm32)) 隔离）。

| 任务 | 候选 ONNX 模型 | 备注 |
|---|---|---|
| 人脸检测+468/106 点 | SCRFD + 106pt landmark（如 scrfd_500m + 2d106det） | 替代 MediaPipe .task（TFLite，tract 不支持） |
| 人体分割 | SelfieMulticlass / MODNet / BiSeNet-person 移植 | 许可证逐个核验 |
| 人体关节点 | RTMPose-tiny / DWPose ONNX | 腿/躯干 warp 定位 |
| 脸部分割（可选增强） | BiSeNet face-parsing 移植 | 同上，许可证先行 |

铁律（沿用 PortraitLab）：权重不入库、来源+版本+许可证+SHA256 登记、
许可证不明不集成。模型放用户目录，首次使用下载。

## 3. 移植阶段

- **Stage 1（本提交）**：`crates/beauty` 参数/预设/蒙版数学 + 测试 + 层级注册。
- **Stage 2**：几何位移场 + remap（对接 `algo` 的 tile/halo 框架；warp 用 `Halo::Bounds`）。
- **Stage 3**：皮肤算法（guided/bilateral、tone、blemish）+ 效果验证（与 Python 版数值对拍）。
- **Stage 4**：tract 推理 + 人脸检测/landmark；N1-N5。
- **Stage 5**：人体分割 + 关节点；B1-B5。B3 背景 inpaint 单独评审。
- **Stage 6**：engine command + egui 面板（滑块组）+ 预设；WASM 小滤镜导出（可选）。
- **收尾**：Python PortraitLab 归档为算法参考（已删 1.7GB `.venv`，代码保留在 git）。

## 4. 与上游同步

主分支永远干净（= upstream 同步点），功能全在 `beauty/*` 分支。
同步流程见 `scripts/sync-fork.sh`（fetch upstream → main fast-forward → 提示 rebase 特性分支）。
