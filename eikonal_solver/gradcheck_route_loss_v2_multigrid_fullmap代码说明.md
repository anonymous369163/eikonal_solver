下面我按你的要求给出\*\*详细实施计划\*\*，并且我已经把代码按计划改完并生成了可直接运行的新脚本（支持“整幅 3k×3k 大图 sliding 分割 + 全图下采样路由测试 + multigrid+tube”）。



✅ 更新后的代码在这里：

\[下载 gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py](sandbox:/mnt/data/gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py)



---



\## 一、实施计划（逐步落地）



\### Step 0：明确“整图测试”为什么不能直接复用 ROI 版



你现有的 `gradcheck\_route\_loss\_v2\_multigrid.py` 走的是：



\* dataset 采样 512×512 patch → 得到 `road\_prob\[512,512]`

\* 再基于 src/tgt 在该 patch 内构建 ROI



但如果我们把 src/tgt 放到 3k×3k 图上，\*\*ROI 的 span 可能让 `P = 2\*(span+margin)+1` 远大于 H/W\*\*，这会触发：



\* 生成一个巨大 padded patch（比如 6k×6k）→ 内存直接爆/非常慢



所以整图测试必须改成：

✅ \*\*“全图下采样网格求解”\*\*，而不是“ROI 可能大于图的 padded patch 求解”。



---



\### Step 1：把 `test\_inference\_route` 的 sliding segmentation 移植进 gradcheck



新增：



\* `--tif /path/to/crop\_xxx.tif`

\* `--stride 256`（sliding 步长）

\* `--smooth\_sigma`（可选）

\* `--cache\_prob /path/to/road\_prob.npy`（缓存整图概率，避免每次都跑 sliding）



实现逻辑：



1\. 读 tif → `rgb\_uint8\[H,W,3]`

2\. sliding-window 调 `model.\_predict\_mask\_logits\_scores` → 拼接得到 `road\_prob\_full\[H,W]`



---



\### Step 2：在整图上选 src/tgt（两种方式）



新增两套入口：



\*\*A. 从 NPZ 节点采样（有 GT 距离，最适合做 loss）\*\*



\* `--sample\_from\_npz`：从 `distance\_dataset\_all\_\*\_p{p\_count}.npz` 抽一个 anchor + K 个 target

\* 自动把 `matched\_node\_norm` (bottom-left) → 转成像素 top-left `(y,x)`

\* `gt\_dist = undirected\_dist\_norm \* H`（像素单位）



\*\*B. 手动给点（没有 GT）\*\*



\* `--src y,x --tgt y,x`

\* 可选：`--teacher\_hard\_eval` 用 hard\_eval 大迭代算一个 pseudo-GT，让脚本仍能做优化/梯度验证



---



\### Step 3：实现“全图下采样 multigrid(+tube) 可微求解”



新增一个新的可微求解函数（脚本内实现）：



`fullgrid\_multigrid\_diff\_solve(...)`



核心流程：



1\. `road\_prob\_full\[B,H,W]` → 通过 `max\_pool2d(ds)` 得到 `prob\_f\[B,Hf,Wf]`

2\. `cost\_f = model.\_road\_prob\_to\_cost(prob\_f)`

3\. coarse 网格 `ds\_coarse = ds \* mg\_factor` 同理得到 `prob\_c、cost\_c`

4\. coarse solve 得到 `T\_c`

5\. 上采样 `T\_c -> T\_init`

6\. fine refine（可选 tube：只在走廊 bbox 上 refine）

7\. 读 targets 的 `T(target)` 得到 `T\_eik\[B,K]`

8\. 用模型里同款 gate 残差混合得到最终 `pred\_dist\[B,K]`



⚠️ 这个“全图下采样求解”不会出现 ROI padding 爆炸问题。



---



\### Step 4：把训练/gradcheck 框架接上



整图模式下仍保留：



\* `--steps N --lr`

\* `loss\_dist = Huber( cap(pred)/norm, gt/norm )`

\* 打印 `sat\_ratio / max\_pred / tube\_meta / timing`



并新增：



\* `--optimize\_prob`：把 `road\_prob\_full` 当作 \*\*可学习变量\*\*（不是分割网络输出），验证“距离监督能否把概率图往更合理方向推”（梯度只会落在求解涉及的区域，仍然可控）



> 解释一下：

> 整图 sliding segmentation 默认在 `no\_grad()` 下跑，否则计算图巨大不可控。

> 所以距离 loss 的梯度默认\*\*不会回传到 encoder/decoder\*\*。

> 但仍能回传到 routing 分支参数（alpha/gamma/gate），以及（可选）road\_prob 变量本身。



---



\### Step 5：整图可视化（优化前/后路径对比）



新增：



\* `--vis\_full\_route`：保存整图 overlay

\* `route\_before.png` / `route\_after.png`



路径计算采用：



\* hard\_eval Eikonal（下采样网格）+ backtrack → 映射回 full-res 像素坐标画在图上



---



\## 二、我已经实现并给你的最终代码



✅ 新脚本：

\[下载 gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py](sandbox:/mnt/data/gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py)



它保留了原本 dataset/ROI 模式（不传 `--tif` 时行为类似原脚本），并新增 fullmap 模式（传 `--tif`）。



---



\## 三、怎么运行（给你三条最常用命令）



\### 1）整图推理 + 画路径（不训练）



```bash

python gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py \\

&nbsp; --ckpt /path/to/your.ckpt \\

&nbsp; --tif /path/to/crop\_xxx.tif \\

&nbsp; --src 1200,800 --tgt 2500,2200 \\

&nbsp; --multigrid --mg\_factor 4 --tube\_roi \\

&nbsp; --steps 0 \\

&nbsp; --vis\_full\_route --vis\_ds 4 --vis\_iters 600 \\

&nbsp; --save\_debug /tmp/fullmap\_test \\

&nbsp; --cache\_prob /tmp/fullmap\_test/road\_prob.npy

```



\### 2）整图（有 NPZ GT）做 5 步优化，看 loss/梯度是否正常



```bash

python gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py \\

&nbsp; --ckpt /path/to/your.ckpt \\

&nbsp; --tif /path/to/crop\_xxx.tif \\

&nbsp; --sample\_from\_npz --p\_count 20 --k\_targets 4 \\

&nbsp; --multigrid --mg\_factor 4 --tube\_roi \\

&nbsp; --steps 5 --lr 1e-4 \\

&nbsp; --save\_debug /tmp/fullmap\_gt \\

&nbsp; --vis\_full\_route

```



\### 3）整图（没 GT）用 teacher hard\_eval 做 pseudo-GT 再训练



```bash

python gradcheck\_route\_loss\_v2\_multigrid\_fullmap.py \\

&nbsp; --ckpt /path/to/your.ckpt \\

&nbsp; --tif /path/to/crop\_xxx.tif \\

&nbsp; --src 1200,800 --tgt 2500,2200 \\

&nbsp; --teacher\_hard\_eval --teacher\_iters 800 \\

&nbsp; --multigrid --mg\_factor 4 --tube\_roi \\

&nbsp; --steps 5 --lr 1e-4 \\

&nbsp; --save\_debug /tmp/fullmap\_teacher \\

&nbsp; --vis\_full\_route

```



---



\## 四、你接下来最值得验证的两点（强烈建议）



1\. \*\*tube 的 bbox 是否永远包含 targets（不再用 clamp 掩盖）\*\*

&nbsp;  我在 fullgrid solve 里对 target 越界直接返回 `large\_val`，这样一旦 tube bbox 算错你会立刻看到 dist 爆掉，而不是被 clamp 伪装成“还能跑”。



2\. \*\*训练阶段 iters\_fine floor 是否应该用全图尺寸\*\*

&nbsp;  我加了 `--tube\_iters\_floor\_full`（默认打开）：



\* 训练更稳（迭代下限用全图网格而不是 tube bbox）

\* 推理如果你只追求速度，可以关掉它



---



如果你愿意，把你实际跑整图时的日志（特别是 `tube\_meta / pred\_time / sat\_ratio / max\_pred`）贴出来，我可以进一步帮你判断：



\* tube 现在是不是“过宽退化/过窄误差增大”

\* mg\_factor、ds、iters\_coarse/fine 在 3k 图上该怎么设才能最划算。



