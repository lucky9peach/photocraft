# 美颜模块安装与试用指南（Filter › Beautify）

分支 `beauty/plugin-port` 把 PortraitLab 美颜做成了 PhotoCraft 原生功能：
**Filter › Beautify** 下三个命令（Portrait Beautify / Skin / Reshape），
macOS 用 Apple Vision 自动识别人脸/人体/分割，无需下载任何模型，
全部本地运行、不联网。

> 这不是 `.wast` 插件文件，而是一组 engine 原生命令（原因见
> `docs/beauty-requirements.md` §0/§5：WASM 沙箱做不了 ML 检测和大幅 warp）。
> “安装”= 拉分支、构建、运行。

## 1. 环境准备（macOS，一次性）

```bash
# 命令行工具（含 swiftc，Apple Vision 桥接需要）
xcode-select --install

# Rust（已装可跳过）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## 2. 拉代码、构建

```bash
git clone git@github.com:lucky9peach/photocraft.git
cd photocraft
git checkout beauty/plugin-port

# 图形界面（Filter › Beautify 菜单）
cargo run -p photocraft

# 或只构建 headless 命令行（试效果最快）
cargo build -p photocraft-cli
```

> 如果 `apple-vision` 构建失败并提示 SwiftPM / property list 错误，
> 是本机 SwiftPM 初始化问题：`cargo clean -p apple-vision -p apple-cf`
> 后重试；仍失败则检查 `xcode-select -p` 是否指向有效的 CLT/Xcode。
> 已验证环境：cargo 1.99.1 + Swift 6.4（仅 CLT）。

## 3. 试用方式一：图形界面

1. `cargo run -p photocraft`，打开一张人像照片。
2. 菜单 **Filter › Beautify › Portrait Beautify…**，对话框里拖滑块，
   预览实时更新（同一张照片第二次渲染约 10ms 级，检测结果已缓存）。
3. 常用起点：`preset = Natural`（自然）/ `Cosplay`（含几何微调）/
   `Editorial`（质感）；再单独微调某个滑块覆盖预设值。
4. 撤销：普通历史记录撤销即可（`s.undo()` 路径已测试）。

## 4. 试用方式二：命令行（推荐先跑通）

```bash
CLI=./target/debug/photocraft-cli

# 确认命令已注册
$CLI commands --filter beautify

# 自然预设，一键美颜
$CLI run portrait.jpg \
  --cmd filter.beautify.portrait --params '{"preset":"Natural"}' \
  --out portrait_natural.png

# 只磨皮提亮（半身像/证件照常用）
$CLI run portrait.jpg \
  --cmd filter.beautify.skin \
  --params '{"skinSmooth":55,"skinTone":30,"blemishFade":50,"brighten":20,"eyeDetail":40}' \
  --out portrait_skin.png

# 只 reshape（全身像瘦身/瘦腿；无腿分叉时自动只瘦不拉长）
$CLI run fullbody.jpg \
  --cmd filter.beautify.reshape \
  --params '{"slimBody":60,"waistSlim":50,"slimLegs":40,"legLengthen":15}' \
  --out fullbody_reshape.png

# 脸型五官单项（V 脸/嘴角/苹果肌/鼻子/大眼/瘦脸）
$CLI run portrait.jpg \
  --cmd filter.beautify.reshape \
  --params '{"eyeEnlarge":20,"jawSlim":15,"noseRefine":12,"vline":10,"mouthLift":12,"cheekLift":10}' \
  --out portrait_face.png
```

每次命令会打印 JSON 结果：`faces`/`bodies`（检出数）、`cacheHit`
（第二次跑同一张图为 true，约快 20 倍）、`ms`（耗时）。

参考耗时（debug 构建，Apple Silicon）：1MP 半身像约 0.8s，
3MP 全身像约 0.7s，28MP 全身像约 14s。release 构建更快。

## 5. 滑块一览（全部 0–100，`saturation` 为 −100–100）

| 分组 | 参数（JSON key） | 说明 |
|---|---|---|
| 预设 | `preset` | `Natural`/`Cosplay`/`Editorial`，先铺底再被显式滑块覆盖 |
| 皮肤 | `skinSmooth` + `textureKeep` | 边缘保留磨皮；`textureKeep` 越高越不塑料 |
| 皮肤 | `skinTone` | Lab 肤色均匀 |
| 皮肤 | `blemishFade` | 瑕疵淡化（3px 以上斑点） |
| 皮肤 | `nasolabialFade` | 法令纹局部平滑 |
| 脸型 | `eyeEnlarge` / `jawSlim` / `noseRefine` | 大眼 / 瘦脸 / 鼻翼收窄 |
| 脸型 | `vline` / `mouthLift` / `cheekLift` | V 脸 / 嘴角上扬 / 苹果肌上提 |
| 美体 | `slimBody` / `waistSlim` | 瘦身 / 瘦腰（mask 驱动，半身像也可用） |
| 美体 | `slimLegs` / `legLengthen` | 瘦腿 / 长腿（无分叉自动禁用拉长） |
| 美体 | `headRatio` | 头身比微调（保守） |
| 光照 | `brighten` / `saturation` | 脸+身体皮肤提亮 / 饱和 |
| 光照 | `eyeDetail` / `sharpen` | 眼部 CLAHE 细节 / 锐化 |

几何类默认全 0（关闭），皮肤/光照按需开。一次跑多个命令可串行：
`run in.jpg --cmd A --params '{…}' --cmd B --params '{…}' --out out.png`。

## 6. 限制与说明

- **macOS 效果最全**（Apple Vision 检测）。Linux/Windows 当前降级为
  chroma 肤色回退：皮肤/光照可用，几何 reshape 跳过（ONNX 解码器后续补）。
- 工作区超过 3600 万像素会拒绝（内存保护）；28MP 已验证通过。
- `legLengthen` 保守：检测不到双腿分叉（裙装/并腿/侧身）时自动置 0，
  只做瘦腿，避免拉伸接缝伪影。
- 背景像素：warp 经人物蒙版门控，实测全身像背景角落逐像素一致（0.0）。

## 7. 跑测试（改代码后）

```bash
cargo test -p photocraft-beauty              # 52 passed
cargo test -p photocraft-engine --lib beauty_cmds  # 10 passed
cargo clippy -p photocraft-beauty --all-targets
cargo fmt --check -p photocraft-beauty -p photocraft-engine
cargo xtask layers                            # 层级检查
cargo check -p photocraft-beauty --no-default-features --target wasm32-unknown-unknown
```

## 8. 同步上游

主分支保持干净（= upstream 同步点），美颜只在 `beauty/*` 分支：

```bash
./scripts/sync-fork.sh   # fetch upstream → main fast-forward → 提示 rebase
```
