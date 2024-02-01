# MultiAgentsRAG
This is a experiment to mitigate fact-conflict hallucination in LLM, and will do further more experiment on financial domain

1) I choose the FiQA as the fine-tuning data to trained two base-models --- Llama2-7b and ChatGLM2-6b under the help of LoRA. And also, I would to take the advantage of thre evaluation Benchmarks -- MMLU, HaluEval, and TruthfulQA, to test the ability to mitigate the hallucination.

2) training your baselines
 
3) training your experiments: the key parts includes: Multi-Agents Debate (MAD) and Retrieval Augmentation Generation(RAG), (1) for the Multi-Agents parts, we basically set up to 2 agents because of the limited computation resource, and set up several debate rounds. (2) for RAG part, we also utilize some reranking strategies and query rewriting strategies to improve the chance of getting more relevant documents as context.

4) evaluating your model output (scoring, sampling, etc)
