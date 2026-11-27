---
layout: post
title: "vLLM-GR 开源发布：昇腾 NPU 加速生成式推荐推理"
author: vLLM-GR Team
date: 2026-11-24 00:00:00 +0800
summary: "vLLM-GR 在昇腾 NPU 上可从长用户历史中一次生成一组符合物品目录约束的推荐候选。模型前向、候选筛选和逐轮解码在设备端衔接，减少候选逐轮回传主机的开销。"
description: "vLLM-GR 在昇腾 NPU 上可从长用户历史中一次生成一组符合物品目录约束的推荐候选。模型前向、候选筛选和逐轮解码在设备端衔接，减少候选逐轮回传主机的开销。"
image: /assets/figures/2026-11-24-vllm-gr-announcement/vLLM-GR.jpg
social_image: /assets/figures/2026-11-24-vllm-gr-announcement/vLLM-GR.jpg
tags:
  - performance
  - generative-recommendation
---

![vLLM-GR 标志](/assets/figures/2026-11-24-vllm-gr-announcement/vLLM-GR.jpg)

**一次请求，从长用户历史中生成一组可用的推荐候选。** vLLM 社区正式发布 [**vLLM-GR**](https://github.com/vllm-project/vllm-gr)。以 OneRec-1.7B 为例，启用物品目录约束后，vLLM-GR 在昇腾（Ascend）NPU 上筛选候选，筛选结果直接用于下一轮解码，减少候选逐轮回传主机的开销。

在昇腾 950 的初步评测中，输入长度为 1,024 个 token、批大小为 8、束宽为 128 时，vLLM-GR 的 P99 端到端延迟仅为 **128.2 ms**。

## 一次推荐请求在昇腾上怎么执行

以视频信息流为例，OneRec-1.7B 用语义 ID（SID）表示视频。单个 SID 只需几个 token，一次推荐请求则要在延迟预算内搜索多条候选路径。沿图中的候选生成路径，vLLM-GR 可以基于同一段观看历史在一次请求中生成 128 个候选 SID，并返回各候选的分数。

![交互历史和任务相关输入经过输入表示、推荐模型和候选预测三个环节。模型示例包括 OneRec-1.7B、HSTU 和 OneTrans，输出包括候选 ID 和排序分数。](/assets/figures/2026-11-24-vllm-gr-announcement/framework-overview.png)

vLLM-GR 复用 vLLM 的模型执行、调度和分页 KV 缓存。EngineCore 从接收请求起管理整个束搜索过程，将预填充（prefill）与各轮解码（decode）交给 Worker。昇腾后端基于 CANN 和 vLLM-Ascend 运行，并调用 xllm-ops 提供的昇腾算子执行注意力计算、KV 缓存处理和候选筛选。[昇腾安装与算子说明](https://github.com/vllm-project/vllm-gr/blob/main/docs/getting_started/installation/npu.md)

![蓝色表示复用的 vLLM 组件，例如 HTTP 入口和模型前向计算。黄色表示对 EngineCore 调度和 Worker 执行的扩展。粉色表示 vLLM-GR 新增的组件，包括离线和 gRPC 入口、束搜索会话、BeamAttn、后缀 KV，以及目录约束和全词表两种候选筛选方式。](/assets/figures/2026-11-24-vllm-gr-announcement/gr-inference-architecture.png)

### 候选在 NPU 上筛完，直接进入下一轮

在昇腾 NPU 的生成式推荐束搜索路径上，vLLM-GR 用专门优化的高性能算子替代原生 vLLM 的通用 Sampler 流程，尽可能压低每轮解码的后处理延迟。模型输出 logits 后，候选筛选和束搜索决策直接在 NPU 上执行。[昇腾束搜索执行路径](https://github.com/vllm-project/vllm-gr/blob/main/vllm_gr/v1/worker/npu/npu_model_runner_patch.py#L204-L220)

启用物品目录约束时，约束 Top-K 算子（`rec_constrained_topk`）把目录查表和 Top-K 筛选融合在设备端，直接为每条父路径选出能够延伸为目录内物品的候选 token，并给出对应的对数概率。[约束表与算子实现](https://github.com/vllm-project/vllm-gr/blob/main/docs/design/module/constraint_tables.md)

设备端束搜索算子再按累计分数选出保留路径，并记录对应的父路径索引。筛选结果直接用于下一轮解码，省去逐轮回传候选的开销。使用 NPU 约束表时，可以按阶段设置束宽和每条父路径的候选上限。昇腾也支持全词表搜索。[功能兼容性](https://github.com/vllm-project/vllm-gr/blob/main/docs/user_guide/feature_compatibility.md)

![一次请求依次搜索两条、三条、两条符合目录约束的路径。虚线分支表示被过滤的前缀。响应包含两个完整 SID 及其示例分数。](/assets/figures/2026-11-24-vllm-gr-announcement/catalog-beam-search.png)

*图示请求按 2 → 3 → 2 的束宽搜索，最终返回两个候选。目录中没有物品以 `A1 · B1` 为 SID 前缀，这条路径会被过滤。全词表搜索会跳过目录过滤。图中的 SID 和分数均为示例。*

### 前缀与后缀注意力，一次算子调用完成

vLLM-GR 让同一请求中的候选路径共用用户历史的前缀 KV，避免为每条路径重复存储。各路径维护分叉后的后缀 KV，下一轮解码根据入选路径的父路径索引读取对应的缓存。安装 xllm-ops 后，满足融合条件的解码批次可以用昇腾后端的 XAttention，在一次算子调用中完成前缀和后缀的注意力计算，省去分别计算后再合并的步骤。[昇腾解码实现](https://github.com/vllm-project/vllm-gr/blob/main/docs/design/feature/npu_decode_execution.md)

### 一次请求的预填充与解码流程

vLLM-GR 启用异步调度后，主机可以在昇腾 NPU 执行当前阶段时准备下一轮解码，缩短轮次之间的等待。以三段式 SID 为例，预填充和随后两轮解码各确定一段 SID，完整候选在最后一轮结束后统一返回。

![BeamSearch 的 PDD 流水线优化。调度器接纳请求、准备提示词，并把预填充阶段交给设备端的 Beam Decoder。随后，每轮解码的准备工作都在前一阶段的设备执行期间进行。候选筛选先保留两个前缀，再保留三个，最终返回两个完整的 SID。](/assets/figures/2026-11-24-vllm-gr-announcement/00-async-pipeline-time-aligned.png)

*图中，调度器接纳一个 `BeamSearch()` 请求并将预填充交给 Worker。预填充为第一段 SID 打分时，主机准备第 1 轮解码。第 1 轮执行期间，主机准备第 2 轮解码。第 2 轮继承入选父路径的后缀 KV，完成两个 SID。绿色标出各阶段选中的 SID 组件。*

## 昇腾 910（A2）与昇腾 950 实测

昇腾基准测试结果为初步数据，反映当前的优化进度。我们将继续优化 vLLM-GR 在昇腾上的性能。

- **昇腾 950 在两档束宽下的尾延迟。** 输入长度为 1,024 个 token、批大小为 8 时，束宽 128 和 256 对应的 P99 端到端延迟分别为 **128.2 ms** 和 **151.7 ms**。
- **昇腾 910（A2）的长历史与宽束测试。** 输入长度为 8,192 个 token、批大小为 8 时，vLLM-GR 完成了束宽最高为 1,024 的整组测试。

下表列出 vLLM-GR 在昇腾 910（A2）和昇腾 950 上的 P99 端到端延迟。输入长度固定为 1,024 个 token，所有配置均启用物品目录约束表。延迟包含主机 CPU 和 NPU 的执行时间，单位为毫秒（ms）。数据来自多组独立测试，实际结果会随测试环境而变化。

| 昇腾型号 | 束宽 | 批大小 1 | 批大小 4 | 批大小 8 |
| --- | ---: | ---: | ---: | ---: |
| 昇腾 910（A2） | 128 | 58.58 | 143.33 | 256.25 |
| 昇腾 910（A2） | 256 | 66.26 | 172.87 | 312.69 |
| 昇腾 950 | 128 | 29.12 | 72.1 | 128.2 |
| 昇腾 950 | 256 | 35.53 | 86.7 | 151.7 |

## 从离线验证到在线部署

vLLM-GR 0.30.0 基于 vLLM 0.30.0。首次在昇腾上部署时，可以先验证 OneRec-1.7B 的候选输出，再将模型接入 HTTP 服务。

1. **准备昇腾环境。** 按设备型号选择镜像，准备 CANN 运行环境和 vLLM-GR 所需算子，完成环境检查。[昇腾 NPU 安装指南](https://github.com/vllm-project/vllm-gr/blob/main/docs/getting_started/installation/npu.md)
2. **跑通离线推理。** 运行 OneRec-1.7B，查看返回的候选 SID 及其累计对数概率。[昇腾离线快速入门](https://github.com/vllm-project/vllm-gr/blob/main/docs/getting_started/quickstart.md#ascend)
3. **接入在线服务。** 启动 OneRec HTTP 服务时指定 `--constraint-backend npu`，加载物品目录约束表。发送请求，检查返回的 SID 是否能对应到目录中的物品。[OneRec HTTP 示例](https://github.com/vllm-project/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#run-one-request)

## 后续路线图

vLLM-GR 的目标是让同一框架同时支持候选生成与排序，覆盖长用户历史和宽束搜索等场景。

### 扩展模型与候选搜索

- **扩大模型覆盖。** 将候选打分扩展到 HSTU、MTGR 和 RankMixer，为 OneTrans 增加原生服务支持，并探索总参数量更大的推荐模型的服务优化，包括混合专家（MoE）模型。
- **让束搜索适应不同的 SID 结构。** 探索在筛选时考虑候选价值，研究为语义 ID 增加输出头。计划在不同类型的设备上支持层级更深的 SID 和可变长度生成，让单条路径遇到结束标记（EOS）时即可停止，释放槽位与 KV 缓存，以便接纳新请求。

### 优化长历史和在线调度

- **减少长历史的重复计算。** 优化 10,000 个 token 及以上历史序列的预填充，探索在线性或稀疏注意力下执行束搜索。用户历史增长时，尝试在不同请求间复用未变化前缀的 KV 块，避免重复预填充。
- **按请求特点动态组批。** 探索根据队列情况决定组批时机和批内请求，使历史长度不同的请求可以一起执行，并在过载时限制队列长度。将形状相近的批次归组，重放同一张已捕获的计算图以减少填充。

### 推进分布式推理与上游协作

- **跨设备 KV 复用。** 探索让服务实例复用同一节点或其他节点上的设备 KV 块，评估两种部署方式下的模型分片与 KV 传输开销。
- **跟进上游 vLLM。** 基于 vLLM 的引擎和平台接口，将上游的调度和 KV 缓存管理改进引入推荐工作负载。定期跟进较新的稳定版 vLLM，在支持的设备上验证引擎正确性、推荐质量与服务性能。

## 加入社区

- **接入新模型。** 欢迎分享模型如何表示用户交互历史、输出哪些预测，以及你希望支持的服务场景。[贡献指南](https://github.com/vllm-project/vllm-gr/blob/main/CONTRIBUTING.md)
- **反馈性能问题。** 请附上模型、历史长度、束宽和硬件配置，方便社区复现。[提交 Issue](https://github.com/vllm-project/vllm-gr/issues)

[浏览 vLLM-GR 项目仓库](https://github.com/vllm-project/vllm-gr)

## 致谢

感谢 InferMatrix 团队的 Hongsheng Liu 和 Han Zhang 率先提出让 vLLM 支持高性能生成式推荐推理。感谢 Inferact 的 Roger Wang 和 Zhipeng Wang 参与深入讨论，帮助确定 vLLM-GR 的路线图和架构。

同时感谢以下团队和社区成员为 vLLM-GR 作出的贡献，以及他们分享的大规模生产部署经验和反馈。

- [Inferact](https://inferact.ai/): [Zhipeng Wang](https://github.com/princepride), [Roger Wang](https://github.com/ywang96)
- [InferMatrix Team](https://infermatrix.si/): [Han Zhang](https://github.com/zhanghanleo10), [Shucheng Zhang](https://github.com/candanzg), [Xiaodong Tian](https://github.com/parallel-star8), [Yunpeng Zhi](https://github.com/ZhiYunpenghuawei), [Yi Ding](https://github.com/DINGEde), [Jierui Yu](https://github.com/JerryTcl), [Hongsheng Liu](https://github.com/hsliuustc0106)
- openPangu: [Xinming Han](https://github.com/Lechmere-1), [Zhiwei Yi](https://github.com/zhiweiyi11), [Mengyuan Ma](https://github.com/mengyuanuom), [Yongxiang Feng](https://github.com/fxy18)
- TikTok-Global E-commerce: [Bin Wang](https://github.com/wangbin-cs), [Junhua Long](https://github.com/LJHG)
- ByteDance-Commerce & Advertising: [Haoran Ma](https://github.com/MaTwickenham), [Wenqiang Zhou](https://github.com/SENGEL-code), [Junwen Guo](https://github.com/CrazyKing5023)
- Lingquai: [Chunxu Jin](https://github.com/Trenx-J), [Wuping Liu](https://github.com/wupingL)
