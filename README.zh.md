<p align="center" width="100%">
<img src="https://i.postimg.cc/xTpWgq3L/pixiu-logo.png"  width="100%" height="100%">
</p>

<div>
<div align="left">
    <a target='_blank'>Kang Wang<sup>1</sup></span>&emsp;
    <a target='_blank'>Gang Hu<sup>1</sup></span>&emsp;
    <a target='_blank'>QingQing Wang<sup>1</sup></a>&emsp;
    <a target='_blank'>Ke Qin<sup>1</sup></a>&emsp;
</div>
<div>
<div align="left">
    <sup>1</sup>Yunan University&emsp;
    <sup>5</sup>Wuhan University&emsp;
    <sup>2</sup>Sun Yat-Sen University&emsp;
    <sup>3</sup>Sichuan University&emsp;
    <sup>4</sup>Southwest Jiaotong University&emsp;
    <sup>6</sup>University of Florida&emsp;
    <sup>7</sup>New York University&emsp;
  <sup>8</sup>ChanceFocus AMC.
</div>
<div align="left">
    <img src='https://i.postimg.cc/DfB8jxV1/ynu.png' alt='Yunnan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png' alt='Wuhan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/C1XnZNK1/Sun-Yat-sen-University-Logo.png' alt='Sun Yat-Sen University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/vTHJdYxN/NYU-Logo.png' alt='New York University' height='100px'>&emsp;
    <img src='https://i.postimg.cc/NjKhDkGY/DFAF986-CCD6529-E52-D7830-F180-D-C37-C7-DEE-4340.png' alt='Sichuan University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/k5WpYj0r/SWJTULogo.png' alt='Southwest Jiaotong University Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/XY1s2RHD/University-of-Florida-Logo-1536x864.jpg' alt='University of Florida Logo' height='100px'>&emsp;
    <img src='https://i.postimg.cc/xTsgsrqN/logo11.png' alt='ChanceFocus AMC Logo' height='100px'>
</div>




-----------------


#任务背景

大预言模型已经成为了自然语言处理任务的重要工具，这些大语言模型展示了其在文本生成、命名实体识别和机器翻译等方面的多样化能力，但是大模型在中国文学研究中的应用不仅限于文本生成和自动翻译，例如，通过大模型对现代文学作品的自动分类和风格分析，可以帮助研究者更好地理解不同作家的创作特点和文学风格。尽管大模型在中国文学研究中展现了广阔的前景，但其应用也面临诸多挑战。例如，大模型在处理古典汉语文本时，常常因为语言的复杂性和多义性而遇到困难；中国文学文本的高度技术性要求特定领域的LLM能够有效地理解复杂的中国文学语言和文学概念。同时，在分析现代文学批评倾向、现代文学批评挖掘、理解古代文学知识、文学作品风格预测、文学语言风格转换以及文学阅读理解和文学语言理解等方面，迫切需要更多样化的中文教学语料库。

最新的大语言模型，如Llama-2/3、Qwen1.5/2和InternLM-2，在自然语言理解和执行各类任务方面表现出色。但数据稀缺和标记化挑战导致模型只能依据自然语言指令处理一些简单的文学任务，并且，在中国文学领域下，它们的应用和评估面临着巨大的挑战，特别是在多模式开发、教学数据集多样性等方面。大多数大语言模型是专门为通用领域量身定做的，这突显了精通中国文学的模型的严重不足，这可能会限制研究界的进一步发展。此外，缺乏专门的中国文学教学数据来支持研究者进行教学调整，也缺少评估基准来全面评估和比较大语言模型在文学任务中的表现。这对于研究者而言是一个巨大的瓶颈。

为了提升模型语义解析能力，进一步实现对语言的深度理解，我们增加了以构式为“目标词”的框架语义解析数据



**Evaluations** :

- [clleval (clleval_aclue)](https://huggingface.co/datasets/ChanceFocus/flare-zh-afqmc)
- [clleval (clleval_author_2_class)](https://huggingface.co/datasets/ChanceFocus/flare-zh-stocka)
- [clleval (clleval_oa)](https://huggingface.co/datasets/ChanceFocus/flare-zh-corpus)
- [clleval (clleval_oa2)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fineval)
- [clleval (clleval_cft)](https://huggingface.co/datasets/ChanceFocus/flare-zh-fe)
- [clleval (clleval_ner_re)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl)
- [clleval (clleval_tsla)](https://huggingface.co/datasets/ChanceFocus/flare-zh-nl2)


---

## FLARE_ZH: Financial Language Understanding and Prediction Evaluation Benchmark

In this section, we provide a detailed performance analysis of FinMA compared to other leading models, including ChatGPT, GPT-4, lince-zero et al. For this analysis, we've chosen a range of tasks and metrics that span various aspects of financial Natural Language Processing and financial prediction.

### Tasks

| Data  | Task                            | Text Types | Raw     | Instruction | Test | License          | Source |
|-------| ------------------------------- |------------|---------|-------------|------| ---------------- |--------|
| OA1   | 现代文学批评倾向                |           | 38,650  |             |      | Apache-2.0       |        |
| OA2   | 现代文学批评挖掘                |            | 120,000 |             |      | Public           |        |
| ACLUE | 古代文学知识理解                |            | 14,769  |             |      | Public           | [3]    |
| cft   | 文学阅读理解                    |            | 1,115   |             |      | Apache-2.0       | [4]    |
| NER_re| 文学语言理解                    |            | 7,955   |             |      | Public           | [5]    |
| author| 文学作品风格预测                |            | 7,955   |             |      | Public           | [5]    |
| tsla  | 文学语言风格转换                |            | 4,499   |             |      | Public           | [5]    |



1. Xu L, Hu H, Zhang X, et al. CLUE: A Chinese language understanding evaluation benchmark[J]. arXiv preprint arXiv:2004.05986, 2020.
2. Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, and Buzhou Tang. 2018. The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4946–4951, Brussels, Belgium. Association for Computational Linguistics.
3. Jinan Zou, Haiyao Cao, Lingqiao Liu, Yuhao Lin, Ehsan Abbasnejad, and Javen Qinfeng Shi. 2022. Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model. In Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP), pages 178–186, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.
4. Zhang L, Cai W, Liu Z, et al. FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models[J]. arxiv preprint arxiv:2308.09975, 2023.
5. Lu D, Liang J, Xu Y, et al. BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark[J]. arxiv preprint arxiv:2302.09432, 2023.
6. https://blog.csdn.net/zcp0216/article/details/122063405?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170159041816800192270328%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=170159041816800192270328&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-122063405-null-null.nonecase&utm_term=%E4%B8%AD%E6%96%87%E6%96%87%E5%AD%A6%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4450
7. https://www.biendata.xyz/competition/ccks_2019_4/


### Evaluation

#### Preparation
##### Locally install
```bash
git clone https://github.com/chancefocus/PIXIU.git --recursive
cd PIXIU
pip install -r requirements.txt
cd PIXIU/src/financial-evaluation
pip install -e .[multilingual]
```


#### Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

For automated evaluation, please follow these instructions:

1. Huggingface Transformer

   To evaluate a model hosted on the HuggingFace Hub (for instance, finma-7b-full), use this command:

```bash
python eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=chancefocus/finma-7b-full,tokenizer=chancefocus/finma-7b-full,use_fast=False" \
    --tasks "flare_ner,flare_sm_acl,flare_fpb"
```

More details can be found in the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) documentation.

2. Commercial APIs


Please note, for tasks such as NER, the automated evaluation is based on a specific pattern. This might fail to extract relevant information in zero-shot settings, resulting in relatively lower performance compared to previous human-annotated results.

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python eval.py \
    --model gpt-4 \
    --tasks flare_ner,flare_sm_acl,flare_fpb
```


## License

CLLLM is licensed under [MIT]. For more details, please see the [MIT](LICENSE) file.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chancefocus/pixiu&type=Date)](https://star-history.com/#chancefocus/pixiu&Date)



-----------------


**免责声明**

本资料库及其内容仅用于**学术和教育目的**。所有资料均不构成金融、法律或投资建议。不对内容的准确性、完整性或实用性提供任何明示或暗示的保证。作者和撰稿人不对任何错误、遗漏或因使用本网站信息而产生的任何后果负责。用户在做出任何财务、法律或投资决定之前，应自行判断并咨询专业人士。使用本资料库所含软件和信息的风险完全由用户自行承担。

**使用或访问本资源库中的信息，即表示您同意对作者、撰稿人以及任何附属组织或个人的任何及所有索赔或损害进行赔偿、为其辩护并使其免受损害。**




