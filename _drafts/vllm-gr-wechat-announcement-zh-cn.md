# vLLM-GR 开源发布：一次请求生成一组推荐候选

刷短视频时，推荐系统需要根据用户的观看历史，快速找到一组可能感兴趣的视频。像 OneRec-1.7B 这样的生成式推荐模型，可以逐步生成视频的语义 ID（Semantic ID，简称 SID）。生成一个 SID 通常只需要几个 token，难点在于如何在一次请求的延迟预算内搜索足够多的候选，并让结果对应目录中的真实物品。

**vLLM-GR** 是基于 vLLM 的开源生成式推荐推理与服务框架。它在设备上完成多轮束搜索（Beam Search），让候选路径共享用户历史的 KV 缓存，并支持按物品目录约束生成。应用提交一次请求，就能拿到一组候选 SID 及其分数。[vLLM 官方发布公告](https://vllm.ai/blog/2026-11-24-vllm-gr-announcement)

![vLLM-GR 项目标识](https://vllm.ai/blog-assets/figures/2026-11-24-vllm-gr-announcement/vLLM-GR.jpg)

## 一次请求，候选是怎么生成的？

vLLM-GR 复用了 vLLM 的调度器、模型执行和分页 KV 缓存。每次搜索由引擎统一调度。模型为下一段 SID 打分，设备筛选候选，下一轮沿着保留的路径继续生成。

**让多条候选共享用户历史。** 同一请求里的候选来自同一段用户历史，可以共用这段历史的前缀 KV。分叉后的候选各自读取对应的后缀 KV，省去为每条路径复制用户历史的开销。

**搜索时过滤目录中不存在的路径。** 引擎支持全词表搜索，也可以启用物品目录约束。举例来说，目录里没有以 `A1 · B1` 开头的 SID，搜索走到这个前缀时就会剪掉该分支。各轮还能单独设定搜索宽度，例如先保留 2 条，再扩展到 3 条，最后返回 2 个完整 SID。

![开启目录约束后，搜索保留可对应真实物品的前缀，最终返回两个完整 SID](https://vllm.ai/blog-assets/figures/2026-11-24-vllm-gr-announcement/catalog-beam-search.png)

*图中 SID 和分数仅用于说明搜索过程。*

**让主机准备和设备计算重叠。** 在 Prefill–Decode–Decode 三阶段的例子中，设备执行当前阶段时，主机已经在准备下一轮解码，从而缩短阶段之间的等待。[架构与流水线说明](https://vllm.ai/blog/2026-11-24-vllm-gr-announcement#inside-the-architecture)

## 昇腾 NPU 上的初步结果

总览测试覆盖昇腾 910（A2）和昇腾 950，固定输入为 **1,024 token**，测试了 Beam 宽度 128、256 以及批大小 1、4、8。P99 端到端延迟包含主机 CPU 和设备执行时间。

- **昇腾 950，批大小为 8。** Beam 宽度为 128 时，vLLM-GR 的 P99 为 **128.2 ms**。宽度增至 256 时，P99 为 **151.7 ms**。
- **昇腾 910（A2），批大小为 8、Beam 宽度为 256。** vLLM-GR 的 P99 为 **312.69 ms**，xLLM 为 **324.60 ms**，两者均启用了目录约束。
- **更长用户历史。** 在昇腾 910（A2）上，输入增至 8,192 token、批大小为 8 时，vLLM-GR 完成了多档 Beam 宽度测试，最高为 **1,024**。

在 Beam 宽度为 256、批大小为 4 或 8 的配置下，vLLM-GR 在两款昇腾平台上均取得图中所列后端的最低 P99。以上为初步结果。各组测试的计时口径、样本量和后端设置存在差异，原稿也未记录 SGLang 的目录约束配置。[昇腾评测原稿与完整图表](https://github.com/zhanghanleo10/vllm-project.github.io/blob/npu/_posts/2026-11-24-vllm-gr-announcement.md#performance-evaluation)

## 从 OneRec-1.7B 开始

vLLM-GR 0.30.0 基于 vLLM 0.30.0，支持昇腾 NPU 和 NVIDIA GPU。在昇腾上，可以先安装运行环境，再用 OneRec-1.7B 跑通离线示例。默认示例会返回 **128 条候选**，并打印前五条的 token 序列及累计对数概率。[昇腾安装指南](https://github.com/vllm-project/vllm-gr/blob/main/docs/getting_started/installation/npu.md) · [离线快速上手](https://github.com/vllm-project/vllm-gr/blob/main/docs/getting_started/quickstart.md#ascend)

要体验在线服务，可以在 OneRec HTTP 示例中加上 `--constraint-backend npu`。示例会用一份小型目录验证返回的完整 SID。框架也提供 gRPC 入口。[OneRec HTTP 示例](https://github.com/vllm-project/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#run-one-request)

## 接下来

项目将继续推进 HSTU、MTGR 和 RankMixer 的候选打分，以及 OneTrans 原生服务。引擎侧将探索更灵活的 SID 生成、在线批处理和调度、长历史的跨请求前缀 KV 复用，以及跨设备 KV 缓存。[项目路线图](https://vllm.ai/blog/2026-11-24-vllm-gr-announcement#future-roadmap)

欢迎在 GitHub 分享你正在使用的推荐模型和推理场景。如果遇到性能瓶颈，可以附上模型、历史长度、Beam 宽度和硬件，方便社区复现。[项目仓库](https://github.com/vllm-project/vllm-gr) · [提交反馈](https://github.com/vllm-project/vllm-gr/issues)

感谢 InferMatrix 团队的 Hongsheng Liu 和 Han Zhang 提出在 vLLM 中支持高性能生成式推荐推理，感谢 Inferact 的 Roger Wang 和 Zhipeng Wang 参与路线图与架构讨论。也感谢各团队和社区成员贡献代码、反馈与生产部署经验。[完整致谢](https://github.com/zhanghanleo10/vllm-project.github.io/blob/npu/_posts/2026-11-24-vllm-gr-announcement.md#acknowledgements)
