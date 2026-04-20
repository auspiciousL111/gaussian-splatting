这个文档是本阶段（修改backward轮）最初的审计结果 不是当前状态 最后的后续工作指导可以参考

1. 当前分支真实状态判定

我审计的是当前磁盘上的工作树版本，不是假设的 clean commit。父仓库在 research/ab-viewer-baseline，CUDA 子模块在 isar-stage-c，而且这四个目标文件当前都有未提交修改。

基于代码和文档的“最新阶段”而不是文档前半段的历史记录，当前主线不是“原版 perspective-only 3DGS”，也不是“真实 ISAR 成像模型”，而是一个双分支 rasterizer：

- projection_mode == 0：保留原版 perspective 主链。[forward.cu:89](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L89) [backward.cu:181](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L181)
- projection_mode != 0：走当前的 orthographic/isar surrogate 分支，即“view-space 线性映射 + 像素尺度 Jacobian + 仍保留 z 深度/alpha 合成/invdepth”的折中实现。[forward.cu:127](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L127) [backward.cu:194](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L194) [auxiliary.h:175](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L175)
  
文档里有历史段落仍写着“backward 仍是透视假设”，那是旧阶段记录，不代表当前状态。[isar_principle_math_notes.md:98](/D:/3DGS_new/gaussian-splatting/isar_principle_math_notes.md#L98) 已被后面的 math-1 记录覆盖。[isar_principle_math_notes.md:377](/D:/3DGS_new/gaussian-splatting/isar_principle_math_notes.md#L377) [isar_principle_math_notes.md:790](/D:/3DGS_new/gaussian-splatting/isar_principle_math_notes.md#L790) [isar_engineering_change_log.md:773](/D:/3DGS_new/gaussian-splatting/isar_engineering_change_log.md#L773)

当前代码与“math-1 主线，math2 coarse visibility 已回退”的文档结论一致：forward.cu / backward.cu / auxiliary.h 保持 math-1，rasterizer_impl.cu 的 coarse visibility 接口已回退。[isar_engineering_change_log.md:921](/D:/3DGS_new/gaussian-splatting/isar_engineering_change_log.md#L921) [isar_principle_math_notes.md:987](/D:/3DGS_new/gaussian-splatting/isar_principle_math_notes.md#L987)

结论：当前仓库真实实现是“perspective 原链 + orthographic/isar surrogate 分支”，不是原版单一路径，也不是完整 ISAR 物理模型。


---

2. forward 数学审查

2.1 mean 投影链

perspective 分支保持原链：
$$\mathbf m_h = \mathbf P [\mathbf X_w,1]^\top,\quad
\mathbf m_{ndc} = \left(\frac{m_x}{m_w}, \frac{m_y}{m_w}, \frac{m_z}{m_w}\right)$$
见 [forward.cu:269](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L269)。

orthographic/isar 分支当前定义是：
$$\mathbf t = \mathbf V \mathbf X_w + \mathbf t_v$$
若 ortho_scale_x/y 任一接近 0，则二者都退回
$$F=\max(\text{isar\_window\_size},10^{-3})$$
然后
$$x_{ndc}=\operatorname{clamp}\!\left(\frac{2t_x}{|\tilde S_x|},-1.3,1.3\right),\quad
y_{ndc}=\operatorname{clamp}\!\left(\frac{2t_y}{|\tilde S_y|},-1.3,1.3\right)$$
并通过
$$u=\frac{W}{2}(x_{ndc}+1)-\frac12,\quad
v=\frac{H}{2}(y_{ndc}+1)-\frac12$$
映射到像素。[forward.cu:276](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L276) [forward.cu:348](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L348) [auxiliary.h:40](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L40)

这确实实现了文档声称的 orthographic/isar surrogate，但注意它只是 x/y 正交化；z 仍保留 view-space 深度语义，不是“真实 ISAR range-crossrange 投影”。

2.2 2D covariance 投影链

perspective 分支还是原版 EWA 风格 Jacobian：
$$J_{\text{persp}}=
\begin{bmatrix}
f_x/t_z & 0 & -f_x t_x/t_z^2\\
0 & f_y/t_z & -f_y t_y/t_z^2\\
0 & 0 & 0
\end{bmatrix}$$
并先对 $$t_x/t_z,t_y/t_z$$ 做 $$\pm 1.3\tan(\mathrm{FoV})$$ 限幅。[forward.cu:96](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L96)

orthographic 分支当前不是旧的 NDC Jacobian，而是文档所说的像素尺度 Jacobian：
$$J_{\text{ortho,pix}}=
\begin{bmatrix}
W/|\tilde S_x| & 0 & 0\\
0 & H/|\tilde S_y| & 0\\
0 & 0 & 0
\end{bmatrix}$$
代码里通过
$$W=2f_x\tan fov_x,\quad H=2f_y\tan fov_y$$
恢复像素宽高，所以单位是对的。[forward.cu:137](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L137) [rasterizer_impl.cu:228](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L228)

然后两分支统一走
$$\Sigma_{2D}=T^\top \Sigma_{3D} T,\quad T=W_{\text{view}}J$$
这里矩阵写法受代码的列主序/转置约定影响，但前后向内部是自洽的。[forward.cu:115](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L115) [forward.cu:156](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L156)

2.3 conic / radius / tile coverage

当前 forward 是：
$$\Sigma'_{2D}=\Sigma_{2D}+0.3I$$
$$Q=\Sigma_{2D}'^{-1}$$
$$r=\left\lceil 3\sqrt{\lambda_{\max}(\Sigma'_{2D})}\right\rceil$$
然后用 getRect(point_image, r) 做 tile coverage。[forward.cu:322](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L322) [forward.cu:340](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L340)

在 math-1 之后，这条链在 orthographic 下单位是自洽的：point_image 是像素，cov2D 是像素平方，conic 是像素逆平方，radius 是像素。

但它仍是工程性包围：
- lambda 里有 max(0.1f, ...) 的硬底；
- coverage 用的是最大特征值生成的圆形上界，不是精确椭圆 tile 裁剪。
  
这两点是“工程可跑”，不是严格几何最优。

2.4 in_frustum / visibility

preprocessCUDA 走的是 projection-aware in_frustum。[forward.cu:251](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L251) [auxiliary.h:151](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L151)

orthographic 分支确实加入了 plane test：
$$-1.3 \le \frac{2t_x}{|\tilde S_x|}\le 1.3,\quad
-1.3 \le \frac{2t_y}{|\tilde S_y|}\le 1.3$$

但两个分支都仍共享：
$$t_z > 0.2$$
这个近裁剪条件。[auxiliary.h:194](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L194)

所以这里的判断是：
- 对“当前 surrogate rasterizer”来说，内部一致；
- 对“纯正交/ISAR 语义”来说，明显还残留 optical camera 的 z 近裁剪假设。
  
2.5 NDC -> pixel 尺度一致性

这部分是当前 forward 最自洽的地方。point_image = ndc2Pix(...) 给出
$$\frac{\partial u}{\partial x_{ndc}}=\frac W2,\quad
\frac{\partial v}{\partial y_{ndc}}=\frac H2$$
而 orthographic computeCov2D 已改到像素 Jacobian $W/S_x,H/S_y$，所以 mean/cov 两条链单位一致。[forward.cu:143](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L143) [forward.cu:348](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L348)

forward 小结

数学上自洽：
- orthographic mean 的 x/y 线性投影；
- orthographic cov 的像素尺度 Jacobian；
- conic/radius/tile coverage 的单位一致性；
- preprocess 路径下的 projection-aware visibility。
  
只是工程可跑：
- isar_window_size 兜底；
- focal*tan_fov 反推像素宽高；
- z>0.2 近裁剪；
- 仍按 z 排序和做 invdepth。
  
最可疑：
- public markVisible 没跟上 projection-aware 定义；
- orthographic 仍共享 optical 深度语义；
- fabs(scale) 和“任一缺失则双轴同退回”的尺度语义很粗糙。
  

---

3. backward 数学审查

3.1 dL/dmean2D -> dL/dmean3D

渲染 backward 里，dL_dmean2D 不是像素单位，而是 NDC 单位，因为代码先乘了
$$\frac{\partial u}{\partial x_{ndc}}=\frac W2,\quad
\frac{\partial v}{\partial y_{ndc}}=\frac H2$$
才写入 dL_dmean2D。[backward.cu:611](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L611) [backward.cu:713](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L713)

perspective 分支继续用原版齐次除法反传。[backward.cu:477](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L477)

orthographic 分支当前写法虽然绕了一层“像素 Jacobian”，但化简后正好是：
$$\frac{\partial \mathcal L}{\partial t_x}
=
\mathbf 1_{\left|\frac{2t_x}{\tilde S_x}\right|\le 1.3}
\frac{2}{|\tilde S_x|}
\frac{\partial \mathcal L}{\partial x_{ndc}}$$
$$\frac{\partial \mathcal L}{\partial t_y}
=
\mathbf 1_{\left|\frac{2t_y}{\tilde S_y}\right|\le 1.3}
\frac{2}{|\tilde S_y|}
\frac{\partial \mathcal L}{\partial y_{ndc}}$$
$$\frac{\partial \mathcal L}{\partial t_z}=0$$
再乘 view 旋转转回 world。[backward.cu:489](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L489) [backward.cu:520](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L520)

这条链严格对应当前 forward 的 orthographic mean 定义。在 orthographic mean 分支里，我没有看到残留的透视 Jacobian 项。

3.2 dL/dconic -> dL/dcov2D -> dL/dcov3D

渲染 backward 先对 conic（逆协方差）积累梯度。[backward.cu:716](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L716)

computeCov2DCUDA 再重建当前 forward 的 cov2D，并做：
- antialias 缩放对 opacity 的反传；
- (\Sigma_{2D}+0.3I)^{-1} 的反传；
- \Sigma_{2D}\to\Sigma_{3D} 的对称矩阵反传。[backward.cu:227](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L227) [backward.cu:234](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L234) [backward.cu:291](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L291)
  
orthographic 分支使用的就是和 forward 一致的常数像素 Jacobian：
$$J_{\text{ortho,pix}}=
\begin{bmatrix}
W/|\tilde S_x| & 0 & 0\\
0 & H/|\tilde S_y| & 0\\
0 & 0 & 0
\end{bmatrix}$$
所以
$$\frac{\partial J}{\partial t}=0$$
于是 orthographic 下 cov-path 对 mean 的几何梯度应为 0。代码确实也是这样：dL_dtx/dty/dtz 的 cov 部分只在 perspective 分支里计算，orthographic 不写；最后只额外叠加 invdepth 项。[backward.cu:331](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L331) [backward.cu:334](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L334) [backward.cu:351](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L351)

这点和文档一致，也和当前 forward 一致。

3.3 scale / rotation -> cov3D

这条链没有 projection_mode 分支，forward/backward 都沿用原链：
$$S=\operatorname{diag}(m\cdot s_x, m\cdot s_y, m\cdot s_z),\quad
M=SR,\quad
\Sigma_{3D}=M^\top M$$
见 [forward.cu:171](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L171) 和 [backward.cu:368](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L368)。

注意 CUDA 内部并没有重新归一化四元数；这意味着这条链的正确性依赖上游传入的 rotation 已经规范化。就这四个文件本身而言，forward/backward 是匹配的，不构成当前 projection_mode 的闭环错误。

3.4 invdepth 相关项

当前 forward 对两个分支都保留：
$$d=\frac{1}{t_z}$$
并参与 alpha 累积。[forward.cu:489](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L489)

backward 也共享：
$$\frac{\partial \mathcal L}{\partial t_z}
\mathrel{+}= -\frac{\partial \mathcal L}{\partial d}\frac{1}{t_z^2}$$
见 [backward.cu:351](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L351)。

所以：
- 从“当前 surrogate forward”角度看，这项应保留；
- 从“纯正交/真实 ISAR”角度看，这项不是投影必然项，而是当前系统刻意保留的共享深度辅助量。
  
3.5 clamp / visibility / boundary 的分段导数

当前 backward 对可见主链是配套的，但不是对整个离散渲染过程的严格全导数：

- orthographic mean clamp：实现了分段导数，超出平面则梯度为 0。[backward.cu:503](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L503)
- perspective cov-path clamp：也有 x_grad_mul / y_grad_mul。[backward.cu:340](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L340)
- radii <= 0、det == 0、rect_area == 0、power > 0、alpha < 1/255、T < 1e-4 都是硬分段；这些点当前统一表现为“早退/零梯度”，不是光滑导数。[forward.cu:335](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L335) [forward.cu:351](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L351) [forward.cu:467](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L467) [backward.cu:167](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L167) [backward.cu:651](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L651)
  
backward 小结

- 对“当前 active surrogate 主链”而言，backward 基本是可信的。
- 对“visibility / coverage / tail truncation 边界”而言，它不是严格光滑全导数。
- 对“真实 ISAR”而言，它仍不是你最终要的 backward。
  

---

4. 仍残留的 perspective 假设

- auxiliary.h::in_frustum 在 orthographic 下仍共享 t_z > 0.2 近裁剪；这是光学相机式深度可见性假设，不是纯正交定义。[auxiliary.h:194](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L194)
- forward/backward/render 都仍把 depth = t_z、排序 key = t_z、invdepth = 1/t_z 当共享深度语义；这是正交 surrogate，可不是 ISAR 成像语义。[forward.cu:372](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L372) [forward.cu:489](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L489) [rasterizer_impl.cu:102](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L102)
- markVisible/checkFrustum 仍调用旧 wrapper，硬编码 perspective-only in_frustum，没有走 projection_mode。[rasterizer_impl.cu:54](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L54) [auxiliary.h:206](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L206)
- orthographic 分支仍借助 focal_x * tan_fovx、focal_y * tan_fovy 恢复 W/2,H/2；这在当前封装下数值没错，但接口语义仍依赖“FoV 占位参数”。[forward.cu:141](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L141) [rasterizer_impl.cu:228](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L228)
- fabs(ortho_scale) 和“双轴同 fallback”意味着当前正交尺度是工程占位，不是严格几何/物理标定定义。[forward.cu:128](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L128) [backward.cu:196](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L196)
  

---

5. 需要重推或重核对的公式族清单

1. Perspective mean 投影链  
对应 forward 定义：齐次投影 + 透视除法。  
数学表达式：
$$\mathbf m_h=\mathbf P[\mathbf X_w,1]^\top,\quad
\mu_{ndc}=(m_x/m_w,m_y/m_w)$$
在 orthographic 下与 perspective 的差异：orthographic 完全去掉 $$1/z$$ 和透视除法。  
对应代码位置：[forward.cu:269](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L269) [backward.cu:477](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L477)  
是否建议优先重写：否。当前看是原链回归基线。

2. Orthographic mean 投影链（含 NDC->pixel 口径）  
对应 forward 定义：view-space 线性映射到 NDC-like 平面，再经 ndc2Pix 到像素。  
数学表达式：
$$x_{ndc}=\operatorname{clamp}\!\left(\frac{2t_x}{|\tilde S_x|},-1.3,1.3\right),\quad
u=\frac{W}{2}(x_{ndc}+1)-\frac12$$
以及
$$\frac{\partial \mathcal L}{\partial t_x}
=
\mathbf 1_{|2t_x/\tilde S_x|\le 1.3}
\frac{2}{|\tilde S_x|}
\frac{\partial \mathcal L}{\partial x_{ndc}}$$
在 orthographic 下与 perspective 的差异：没有 $$z$$ 依赖，mean 直传链对 $$t_z$$ 理论上应为 0。  
对应代码位置：[forward.cu:276](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L276) [backward.cu:489](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L489) [backward.cu:611](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L611)  
是否建议优先重写：否。应重核对，但当前是闭环的。

3. Perspective covariance Jacobian 与 cov->mean 链  
对应 forward 定义：
$$J_{\text{persp}}=
\begin{bmatrix}
f_x/t_z & 0 & -f_x t_x/t_z^2\\
0 & f_y/t_z & -f_y t_y/t_z^2\\
0 & 0 & 0
\end{bmatrix}$$
数学表达式：需要手推 $\partial J/\partial t_x,\partial J/\partial t_y,\partial J/\partial t_z$，尤其是 $-f_xt_x/t_z^2$、$-f_yt_y/t_z^2$ 对均值的贡献。  
在 orthographic 下与 perspective 的差异：orthographic 的 $$J$$ 常数，不再通过 cov-path 给 mean 几何梯度。  
对应代码位置：[forward.cu:96](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L96) [backward.cu:334](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L334)  
是否建议优先重写：否。只要不改 perspective 主链，就先保持原样。

4. Orthographic covariance Jacobian（像素尺度版）  
对应 forward 定义：
$$J_{\text{ortho,pix}}=
\begin{bmatrix}
W/|\tilde S_x| & 0 & 0\\
0 & H/|\tilde S_y| & 0\\
0 & 0 & 0
\end{bmatrix}$$
数学表达式：
$$\Sigma_{2D}=T^\top\Sigma_{3D}T,\quad T=W_{\text{view}}J_{\text{ortho,pix}}$$
在 orthographic 下与 perspective 的差异：$J$ 不依赖 $(t_x,t_y,t_z)$，所以 cov-path 对 mean 的几何梯度应为 0。  
对应代码位置：[forward.cu:137](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L137) [backward.cu:205](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L205)  
是否建议优先重写：否。当前应是“核对项”，不是“首改项”。

5. Conic 逆矩阵 + antialias opacity scaling  
对应 forward 定义：
$$\Sigma'=\Sigma_{2D}+0.3I,\quad
h=\sqrt{\max\!\left(2.5\times10^{-5},\frac{\det\Sigma_{2D}}{\det\Sigma'}\right)},\quad
Q=\Sigma'^{-1}$$
数学表达式：需手推 $$Q$$ 对 $$\Sigma'$$ 的导数，以及 $$h$$ 对 $$(c_{xx},c_{xy},c_{yy})$$ 的导数。  
在 orthographic 下与 perspective 的差异：形式相同，只是 $$\Sigma_{2D}$$ 的来源不同。  
对应代码位置：[forward.cu:322](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L322) [backward.cu:234](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L234) [backward.cu:274](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L274)  
是否建议优先重写：否。当前没有分支错配迹象。

6. \Sigma_{2D} -> \Sigma_{3D} 对称矩阵回传族  
对应 forward 定义：
$$\Sigma_{2D}=T^\top\Sigma_{3D}T$$
数学表达式：需把对称 3D covariance 的 6 个独立分量正确展开，尤其是 off-diagonal 的倍数。  
在 orthographic 下与 perspective 的差异：同一公式，差在 $T$。  
对应代码位置：[backward.cu:291](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L291)  
是否建议优先重写：否。当前更像稳定基线。

7. Shared invdepth 链  
对应 forward 定义：
$$d=\frac{1}{t_z}$$
数学表达式：
$$\frac{\partial \mathcal L}{\partial t_z}
\mathrel{+}= -\frac{\partial \mathcal L}{\partial d}\frac{1}{t_z^2}$$
在 orthographic 下与 perspective 的差异：从当前代码语义看两者共享；但从纯 orthographic/ISAR 理论看，这不是必须项。  
对应代码位置：[forward.cu:489](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L489) [backward.cu:351](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L351)  
是否建议优先重写：是，但属于语义升级，不是当前闭环 bug。优先级 P1/P2。

8. scale -> \Sigma_{3D}  
对应 forward 定义：
$$S=\operatorname{diag}(m s_x,m s_y,m s_z),\quad
\Sigma_{3D}=(SR)^\top(SR)$$
数学表达式：对 $$s_x,s_y,s_z$$ 的导数来自 $M=SR$。  
在 orthographic 下与 perspective 的差异：无。  
对应代码位置：[forward.cu:171](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L171) [backward.cu:411](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L411)  
是否建议优先重写：否。

9. rotation -> \Sigma_{3D}  
对应 forward 定义：四元数转矩阵，再进入 $\Sigma_{3D}=(SR)^\top(SR)$。  
数学表达式：需要对四元数到旋转矩阵的各项求导。  
在 orthographic 下与 perspective 的差异：无。  
对应代码位置：[forward.cu:179](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L179) [backward.cu:421](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L421)  
是否建议优先重写：否。除非你怀疑上游 rotation 规范化契约失效。

10. Visibility / clamp / radius / tile coverage 边界族  
对应 forward 定义：
$$\chi_{\text{ortho}}=
\mathbf 1[t_z>0.2]\,
\mathbf 1\!\left[\left|\frac{2t_x}{\tilde S_x}\right|\le1.3\right]\,
\mathbf 1\!\left[\left|\frac{2t_y}{\tilde S_y}\right|\le1.3\right]$$
以及
$$r=\left\lceil 3\sqrt{\lambda_{\max}(\Sigma_{2D}+0.3I)}\right\rceil$$
外加 det==0、rect_area==0、power>0、alpha<1/255、T<1e-4 等早退。  
在 orthographic 下与 perspective 的差异：orthographic preprocess 多了 plane test；perspective public visibility 仍没跟上。  
对应代码位置：[auxiliary.h:151](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L151) [forward.cu:340](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L340) [backward.cu:167](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L167) [rasterizer_impl.cu:54](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L54)  
是否建议优先重写：是。这是当前四个文件里最值得优先统一的一族。


---

6. 修改优先级（P0 / P1 / P2）

P0：必须先改，否则“系统层面的投影定义”不统一

- 把 markVisible/checkFrustum 也改成 projection-aware，和 preprocessCUDA 使用同一套 in_frustum 定义。当前它仍硬编码走旧 wrapper，这意味着仓库里其实并存两套 visibility 语义。[rasterizer_impl.cu:54](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu#L54) [auxiliary.h:206](/D:/3DGS_new/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h#L206)  
如果你确认 markVisible 永远不在 orthographic 工作流里用，这条会降成 P1；但从“仓库真实数学定义统一性”看，它是当前最像 P0 的点。

P1：建议尽快改，否则结果会偏，但不属于主链 Jacobian 已错

- 明确 orthographic surrogate 的 z 语义：是否继续保留 t_z>0.2、depth=t_z、排序按 t_z、invdepth=1/t_z。当前这些项前后向一致，但它们决定了你到底是在做“正交 surrogate”还是“半正交半 optical”的混合体。
- 明确 ortho_scale_x/y 的符号和 fallback 语义。当前 fabs 会吃掉符号，且任一轴缺失会双轴同退回 isar_window_size。
- 边界验证应单独补：|x_{ndc}|=1.3、radius==0、rect_area==0、alpha tail、T cutoff。这些点当前统一被当成零梯度处理。
  
P2：后续再改，属于“更真实 ISAR 化”而不是当前 surrogate 问题

- 把当前 z-排序 alpha splatting 改成更接近 ISAR 的成像/观测语义。
- 把 invdepth 从共享光学辅助量改成你真正要的 ISAR 深度/距离定义，或者彻底移除。
- 把当前几何正交 surrogate 升级成真实 range-crossrange 或其它 ISAR 物理投影。
- 如果后续要优化相机正交尺度，再补 \partial L/\partial ortho\_scale_x,\partial L/\partial ortho\_scale_y。
  

---

7. 最小安全修改顺序

1. 先不要碰 mean Jacobian 和 cov Jacobian。当前这两条在 active render/backward 主链上已经基本闭环。
2. 第一改只做 visibility 统一：把 rasterizer_impl.cu 的 checkFrustum/markVisible 接到 projection-aware in_frustum，并透传 projection_mode / ortho_scale_x / ortho_scale_y / isar_window_size。不要同时改 depth/invdepth。
3. 第二改再决定 z 语义。如果你要维持 surrogate，就把 z>0.2、depth=t_z、invdepth=1/t_z 明确冻结并补验证；如果你要更接近正交/ISAR，就把 forward depth、排序 key、invdepth forward、invdepth backward 一起改，作为一个原子 patch。
4. 第三改才考虑尺度语义：是否允许 signed ortho_scale、是否保留“双轴同 fallback”。
5. 每一步都做最小验证，但只验证被改的一条链：
  - visibility 改后：只看 orthographic 可见集是否与 preprocess 一致。
  - z 语义改后：只做 invdepth 数值梯度和排序稳定性检查。
  - scale 语义改后：只做 orthographic mean/cov 数值梯度检查。
6. 不要把“visibility 统一”和“mean/cov Jacobian 重写”放在同一轮。否则一旦数值差异出现，原因会完全混在一起。