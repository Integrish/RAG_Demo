# RAG系统实战：智能新闻问答助手

## 系统设计与实现

### 系统架构

我们的新闻RAG系统包含三个核心组件：

1. **文档加载模块**：从本地文件夹加载新闻文档

2. **检索引擎**：使用BM25算法找到与查询最相关的文档

3. **生成模型**：基于检索结果生成自然语言回答

   ​

![deepseek_rmaid_20250609_6cc9d4](C:\Users\12534\Desktop\deepseek_mermaid_20250609_6cc9d4.png)

### 准备工作

#### 1. 文本生成与数据准备

创建News文件夹，通过爬虫或人工整理生成新闻文本库后将文本库仿佛文件夹中（此处为演示采用了deepseek生成的新闻，生成文章放在最底部）。

#### 2. 核心依赖程序包

```python
!pip install rank_bm25 jieba zhipuai
```

### 完整代码实现

```python
import os
import jieba
from zhipuai import ZhipuAI
from rank_bm25 import BM25Okapi


def call_large_model(prompt, api_key):
    """调用AI大模型接口"""
    client = ZhipuAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="glm-3-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=500
        )
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"调用大模型出错: {str(e)}")
        return "无法获取回答"


class NewsRAGSystem:
    def __init__(self, folder_path="News"):
        """初始化新闻RAG系统"""
        self.load_news_data(folder_path)

    def load_news_data(self, folder_path):
        """加载新闻数据并建立索引"""
        self.news_data = {}
        self.tokenized_corpus = []
        self.news_titles = []
        self.file_paths = []

        # 确保文件夹存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建了文件夹: {folder_path}，请将新闻文档放入此文件夹")
            return

        # 加载所有新闻文档
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                news_title = file_name.split(".")[0].replace("_", " ")
                file_path = os.path.join(folder_path, file_name)

                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.news_data[news_title] = content
                    self.news_titles.append(news_title)
                    self.file_paths.append(file_path)
                    self.tokenized_corpus.append(jieba.lcut(content))

        # 如果加载了文档，则建立BM25索引
        if self.tokenized_corpus:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)
            print(f"已加载 {len(self.news_titles)} 篇新闻文档")
        else:
            print(f"文件夹 {folder_path} 中没有找到任何新闻文档")

    def retrieve(self, user_query, top_k=3):
        """检索最相关的新闻文档"""
        if not hasattr(self, 'bm25_model'):
            print("尚未加载新闻数据，请先加载数据")
            return []

        query_tokens = jieba.lcut(user_query)
        scores = self.bm25_model.get_scores(query_tokens)
        doc_scores = list(enumerate(scores))
        sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # 返回top_k个最相关的结果
        results = []
        for i in range(min(top_k, len(sorted_scores))):
            index = sorted_scores[i][0]
            results.append({
                "title": self.news_titles[index],
                "content": self.news_data[self.news_titles[index]],
                "file_path": self.file_paths[index],
                "score": sorted_scores[i][1]
            })
        return results

    def query(self, user_query, api_key, use_rag=True):
        """处理用户查询"""
        print("\n" + "=" * 50)
        print(f"用户查询: {user_query}")

        if use_rag:
            # RAG模式：检索相关信息并生成回答
            retrieved_results = self.retrieve(user_query)

            if not retrieved_results:
                print("未找到相关新闻信息")
                return "抱歉，没有找到相关信息"

            # 显示检索结果
            print("\n检索到的新闻:")
            for i, result in enumerate(retrieved_results, 1):
                print(f"{i}. [{result['title']}] 相似度: {result['score']:.4f}")
                print(f"   文件路径: {result['file_path']}")

            # 构建提示词
            context = "\n\n".join([
                f"新闻标题: {res['title']}\n内容摘要: {res['content'][:200]}..."
                for res in retrieved_results
            ])

            prompt = (
                "你是一个新闻分析专家，请根据以下检索到的新闻内容回答用户问题。"
                "如果问题需要具体数据，请确保使用新闻中的精确数据。\n\n"
                f"### 检索到的新闻内容:\n{context}\n\n"
                f"### 用户问题:\n{user_query}\n\n"
                "### 回答要求:\n"
                "1. 直接回答问题，不要重复问题\n"
                "2. 如果答案来自多个新闻，请注明来源\n"
                "3. 如果问题无法从新闻中回答，请说明原因"
            )
        else:
            # 非RAG模式：直接回答问题
            prompt = user_query

        # 调用大模型
        response_text = call_large_model(prompt, api_key)

        print("\n模型回答:")
        print(response_text)
        print("=" * 50)

        return response_text


if __name__ == "__main__":
    # 直接定义API密钥 - 请替换为您自己的密钥
    MY_API_KEY = "xxxxxxxxx"  # 替换为您的实际API密钥

    # 创建RAG系统
    rag = NewsRAGSystem()

    # 示例查询
    queries = [
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 替换为您的问题
    ]

    # 使用RAG模式回答
    for query in queries:
        rag.query(query, MY_API_KEY)

    # 对比非RAG模式
    print("\n" + "=" * 50)
    print("非RAG模式测试:")
    rag.query("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", MY_API_KEY, use_rag=False)  # 替换为您的问题
```

### 技术优化

#### 1. BM25检索算法

我们选择**BM25算法**作为检索核心，原因在于：

- 计算效率高，适合实时检索
- 对短文本查询效果优异
- 不需要预先训练，开箱即用

```python
# 初始化BM25模型
self.bm25_model = BM25Okapi(self.tokenized_corpus)
# 执行检索
tokens = jieba.lcut(query)
scores = self.bm25_model.get_scores(tokens)
```

#### 2. 提示工程优化

设计提示词以提升回答质量：

```python
prompt = (
"作为新闻分析专家，请根据以下新闻内容回答用户问题：\n\n"
f"### 相关新闻:\n{context}\n\n"
f"### 用户问题:\n{user_query}\n\n"
"### 回答要求:\n"
"1. 直接回答问题，不重复问题\n"
"2. 使用新闻中的精确数据\n"
"3. 如答案来自多篇新闻，注明来源"
)
```

## 测试结果分析

**查询1：特斯拉2023年发布了什么电池技术？**

##### 相关生成文本

```
特斯拉在2023年电池日公布了新一代4680电池技术。这种新型圆柱电池直径46mm，高80mm，相比前代2170电池能量密度提升16%，充电速度加快20%。特斯拉通过干电极工艺和无极耳设计，将生产成本降低14%。首批4680电池已在德州工厂量产，将用于Model Y车型。
```

##### RAG模式效果

```
==================================================
用户查询: 特斯拉在2023年发布了什么电池技术？

检索到的新闻:

1. [document1_tech_tesla_battery] 相似度: 12.1520
   文件路径: News\document1_tech_tesla_battery.txt
2. [document7_ai_multimodal] 相似度: 3.3915
   文件路径: News\document7_ai_multimodal.txt
3. [document11_env_arctic] 相似度: 2.4955
   文件路径: News\document11_env_arctic.txt

模型回答:

特斯拉在2023年电池日公布了新一代4680电池技术。这种新型圆柱电池直径46mm，高80mm，相比前代2170电池能量密度提升16%，充电速度加快20%。特斯拉通过干电极工艺和无极耳设计，将生产成本降低14%。首批4680电池已在德州工厂量产，将用于Model Y车型。来源：document1_tech_tesla_battery

==================================================
```

##### 非RAG模式效果

```
==================================================
用户查询: 特斯拉在2023年发布了什么电池技术？

模型回答:
在2023年，特斯拉并没有发布全新的、革命性的电池技术名称（像之前的“4680”或“Tabless”电池那样广为人知）。然而，他们在电池技术方面有几个重要的进展和宣布：

1. **柏林工厂的“结构电池包”（Structural Battery Pack）实现量产：** 这是特斯拉在2023年电池技术方面最显著的实际进展之一。在柏林超级工厂（Gigafactory Berlin, GFB）生产的标准续航版Model Y，首次大规模采用了“一体化压铸后底板 + 结构电池包”的设计。这种结构电池包将电池包的一部分作为车身结构的一部分，不仅简化了生产流程、降低了成本，还可能有助于提升刚性和安全性。虽然使用的电池电芯（可能是宁德时代的磷酸铁锂 LFP 电池或 NCM 电池）本身并非全新发布，但这种**结构集成方式**是技术上的一个重要应用和改进。
2. **电池技术的持续优化和多样化应用：**
   - **磷酸铁锂（LFP）电池的普及：** 特斯拉继续在全球范围内扩大LFP电池的应用，特别是在标准续航版的Model 3和Model Y上，以降低成本、提高安全性并延长电池寿命（尤其是在浅充浅放的情况下）。中国、欧洲（通过柏林工厂）和北美（通过德州工厂）都在逐步采用或已经采用LFP电池。
   - **宁德时代麒麟电池（CTP 3.0）：** 特斯拉上海超级工厂在2023年上半年开始为部分Model 3和Model Y车型配备宁德时代的麒麟电池。这是一种高能量密度、无模组的电池包技术，也是电池包层面的技术进步，旨在提升续航里程。
3. **关于“4695”电池的讨论（未正式发布）：** 有报道称，特斯拉正在研发一种名为“4695”（或类似规格）的新型电池，旨在结合4680电池的性能优势和更低的成本。然而，这种电池在2023年并未正式发布或投入量产，仍然处于研发或早期测试阶段。
4. **电池回收技术的进展：** 特斯拉也继续强调其在电池回收方面的努力，旨在提高材料回收率，减少对原生资源的依赖，这也是其电池技术生态的重要组成部分。

**总结来说，** 2023年特斯拉在电池技术方面最重要的进展是将“结构电池包”设计投入量产（尤其是在柏林工厂），并持续

==================================================
```

**查询2：核聚变研究有什么突破？**

##### 相关生成文本

```
劳伦斯利弗莫尔国家实验室实现核聚变能量净增益突破。最新实验中，192束激光向氘氚靶丸输送2.1兆焦能量，产出3.15兆焦聚变能，能量增益系数Q值达1.5。该成果使商业聚变发电目标提前至2035年。
```

##### RAG模式效果

```
==================================================
用户查询: 核聚变研究有什么突破？

检索到的新闻:

1. [document12_env_fusion] 相似度: 3.0109
   文件路径: News\document12_env_fusion.txt
2. [document6_finance_central_bank] 相似度: 2.2089
   文件路径: News\document6_finance_central_bank.txt
3. [document10_entertainment_streaming] 相似度: 0.8661
   文件路径: News\document10_entertainment_streaming.txt

模型回答:

劳伦斯利弗莫尔国家实验室实现核聚变能量净增益突破。最新实验中，192束激光向氘氚靶丸输送2.1兆焦能量，产出3.15兆焦聚变能，能量增益系数Q值达1.5。该成果使商业聚变发电目标提前至2035年。来源：document12_env_fusion

==================================================
```

##### 非RAG模式效果

```
==================================================
用户查询: 核聚变研究有什么突破？

模型回答:
核聚变研究近年来取得了一些重要的进展和突破，虽然距离商业化应用仍有很长的路要走，但以下是一些关键领域的突破：

**1. 实验装置性能提升:**

**ITER项目进展顺利:**  国际热核聚变实验堆 (ITER) 项目是世界上最大的国际科研合作项目，旨在建造一个可自持燃烧的托卡马克核聚变实验堆。近年来，ITER 项目在部件制造、安装和测试方面取得了显著进展，包括：

- **超导磁体系统:**  用于约束等离子体的超导磁体是 ITER 的核心部件，其制造和安装进展顺利。
- **真空室:**  ITER 的真空室是容纳高温等离子体的容器，其组装工作正在进行中。
- **中性束注入器:**  用于加热等离子体的中性束注入器已完成关键部件的制造和测试。

**国内实验装置取得突破:**  中国的“东方超环” (EAST) 和“中国环流器二号M” (HL-2M) 等实验装置也取得了重要进展，例如：

- **EAST:**  在 2021 年实现了 1.2 亿摄氏度等离子体运行 101 秒的世界纪录，以及 1.6 亿摄氏度等离子体运行 20 秒的物理实验，创造了托卡马克装置运行新的世界纪录。
- **HL-2M:**  作为中国新一代先进磁约束 Controlled Fusion 装置，HL-2M 在等离子体物理实验方面也取得了显著成果。

**2. 物理研究取得进展:**

**等离子体物理研究:**  对等离子体物理的深入理解是核聚变研究的关键。近年来，研究人员在等离子体稳定性、输运、加热和约束等方面取得了重要进展，例如：

- **高约束模式 (H-mode):**  H-mode 是一种高效的等离子体运行模式，可以提高能量约束时间。研究人员正在探索如何更稳定地维持 H-mode。
- **湍流抑制:**  等离子体中的湍流会导致能量损失，研究人员正在研究如何抑制湍流以提高能量约束。

**材料研究取得进展:**  核聚变反应堆需要能够承受极端高温、高辐照和强等离子体轰击的材料。近年来，研究人员在材料研发方面取得了

==================================================
```

## RAG与非RAG效果深度对比分析

##### 根据查询1可以看出

| 对比维度      | RAG模式表现            | 非RAG模式表现         | 关键差异分析       |
| --------- | ------------------ | ---------------- | ------------ |
| **技术名称**  | ✅ 准确识别4680电池技术     | ❌ 未提及4680电池      | RAG精确锁定技术名称  |
| **物理参数**  | ✅ 直径46mm/高80mm     | ❌ 未提供任何尺寸参数      | 缺失核心规格信息     |
| **性能提升**  | ✅ 能量密度+16%充电速度+20% | ❌ 仅描述"重要进展"      | 量化指标 vs 模糊表述 |
| **生产工艺**  | ✅ 干电极工艺无极耳设计       | ❌ 未提及具体工艺        | 技术细节完整度差异    |
| **量产信息**  | ✅ 德州工厂量产Model Y应用  | ❌ 未涉及量产规划        | 落地应用信息缺失     |
| **事实准确性** | ✅ 符合文档内容           | ❌ 错误断言"未发布革命性技术" | 严重事实错误       |
| **信息聚焦度** | ✅ 专注电池技术本身         | ❌ 偏离到电池包结构设计     | 核心问题未解答      |
| **数据来源**  | ✅ 标注来源文件           | ❌ 无来源信息          | 可验证性差异       |

##### 根据查询2可以看出

| 对比维度     | RAG模式表现            | 非RAG模式表现           | 关键差异分析    |
| -------- | ------------------ | ------------------ | --------- |
| **核心突破** | ✅ 能量净增益(Q=1.5)     | ❌ 未提及能量净增益         | 突破性成果识别能力 |
| **实验数据** | ✅ 输入2.1兆焦→输出3.15兆焦 | ❌ 无具体能量数据          | 量化实验数据完整性 |
| **技术路径** | ✅ 192束激光靶向照射       | ❌ 未说明实现方法          | 技术实现细节深度  |
| **关键指标** | ✅ 明确Q值=1.5         | ❌ 缺失关键性能指标         | 科学参数精确度   |
| **应用前景** | ✅ 商业发电目标2035年      | ❌ 未预测商业化时间         | 实际应用价值评估  |
| **时效性**  | ✅ 反映2023年最新成果      | ❌ 聚焦较早成果(如2021年纪录) | 信息更新及时性   |
| **研究机构** | ✅ 指定劳伦斯利弗莫尔实验室     | ❌ 泛称"研究人员"         | 责任主体明确度   |
| **突破意义** | ✅ 使商业聚变发电提前        | ❌ 仅描述"取得重要成果"      | 价值评估具体性   |

#### 由此可得出：

##### RAG系统核心价值总结

| 对比维度      | RAG模式                    | 非RAG模式         | 案例体现         |
| --------- | ------------------------ | -------------- | ------------ |
| **事实准确性** | ✅ 基于最新文档数据               | ❌ 依赖可能过时的模型知识  | 查询1的4680电池否认 |
| **数据精确性** | ✅ 输出具体参数(46mm/16%/Q=1.5) | ❌ 模糊描述("重要进展") | 查询2的Q值缺失     |
| **可追溯性**  | ✅ 标注来源文件                 | ❌ 无法验证信息来源     | 两查询的source标注 |
| **信息聚焦度** | ✅ 直接回答问题                 | ❌ 易发散到相关但非核心内容 | 查询1的电池包设计偏移  |
| **时效保障**  | ✅ 动态更新知识库                | ❌ 受限于模型训练截止时间  | 查询2的2023年突破  |

### 附录：生成新闻文本库

**document1_tech_tesla_battery.txt**

```
特斯拉在2023年电池日公布了新一代4680电池技术。这种新型圆柱电池直径46mm，高80mm，相比前代2170电池能量密度提升16%，充电速度加快20%。特斯拉通过干电极工艺和无极耳设计，将生产成本降低14%。首批4680电池已在德州工厂量产，将用于Model Y车型。
```

**document2_tech_quantum_computing.txt**

```
谷歌量子AI团队在《自然》期刊发表论文，宣布实现量子霸权2.0。其新一代Sycamore量子处理器包含72个超导量子比特，在随机电路采样任务中仅需200秒完成传统超级计算机需1万年计算的任务。该突破为量子机器学习算法提供了硬件基础。
```

**document3_medical_alzheimers.txt**

```
礼来公司宣布其阿尔茨海默病药物Donanemab在三期临床试验中取得突破性成果。试验数据显示，该抗体药物能清除大脑β-淀粉样蛋白斑块，使早期患者认知衰退速度减缓35%。FDA已授予其突破性疗法认定，预计2024年Q2获批上市。
```

**document4_medical_crispr.txt**

```
《新英格兰医学杂志》发表CRISPR基因编辑治疗镰状细胞病临床结果。单次治疗使30名患者中有28人摆脱疼痛危象，血红蛋白水平恢复正常并维持12个月以上。该疗法通过编辑患者造血干细胞BCL11A基因，重新激活胎儿血红蛋白表达。
```

**document5_finance_bitcoin_etf.txt**

```
美国SEC于2024年1月10日正式批准首批比特币现货ETF，包括贝莱德、富达等11家机构产品。首日交易量达46亿美元，创ETF上市纪录。分析师预测这将吸引2000亿美元机构资金入场，推动加密货币进入主流资产配置。
```

**document6_finance_central_bank.txt**

```
国际清算银行(BIS)发布央行数字货币(CBDC)研究报告。调查显示86%央行正在探索CBDC，其中60%预计2030年前推出零售型数字货币。报告强调"可编程货币"特性将实现精准财政政策，但需平衡隐私保护与监管需求。
```

**document7_ai_multimodal.txt**

```
OpenAI发布GPT-5技术白皮书，展示其多模态能力突破。新模型整合视觉、听觉和文本理解，在医学影像诊断任务中达到94%准确率，超过放射科医师平均水平。伦理部分披露已建立实时内容审核层，防止医疗建议滥用。
```

**document8_ai_robotics.txt**

```
波士顿动力Atlas机器人实现全自主建筑作业。通过结合NeRF环境重建和强化学习算法，机器人能独立完成砖墙砌筑、管线安装等复杂任务。现场测试显示其工作效率达人工团队的80%，错误率低于2%。
```

**document9_sports_olympics.txt**

```
巴黎奥组委公布2024奥运会创新方案：将在塞纳河举办开幕式，运动员乘船入场；引入电子竞技表演赛项目；使用AI裁判系统辅助体操、跳水等评分项目。奥组委承诺本届赛事将实现碳中和，95%场馆为现有或临时设施。
```

**document10_entertainment_streaming.txt**

```
Netflix Q4财报显示其游戏业务用户突破8000万，同比增长300%。《怪奇物语》手游DAU达1200万，单用户日均游戏时长47分钟。公司宣布将投入50亿美元开发原创游戏IP，打造"互动影视宇宙"。
```

**document11_env_arctic.txt**

```
NASA北极科考报告显示：2023年夏季海冰面积降至372万平方公里，为卫星记录以来最低。冻土融化释放180亿吨甲烷，加速温室效应。报告预测2040年可能出现首个"无冰之夏"，导致全球海平面上升0.5米。
```

**document12_env_fusion.txt**

```
劳伦斯利弗莫尔国家实验室实现核聚变能量净增益突破。最新实验中，192束激光向氘氚靶丸输送2.1兆焦能量，产出3.15兆焦聚变能，能量增益系数Q值达1.5。该成果使商业聚变发电目标提前至2035年。
```

**document13_education_ai.txt**

```
教育部启动AI教育三年行动计划：2025年前为全国中小学配备AI实验室，培养10万名AI专业教师；高等教育新增"智能+"交叉学科；建立国家AI教育云平台，提供GPU算力支持。计划目标使我国AI人才储备达全球30%份额。
```

**document14_culture_metaverse.txt**

```
故宫博物院推出元宇宙展馆"数字紫禁城"，首日访问量破500万。通过VR技术还原清代宫廷生活场景，数字藏品《清明上河图》NFT售价达23万美元。项目采用区块链确权技术，确保文物数字版权。
```