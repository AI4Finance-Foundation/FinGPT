# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export TOKENIZERS_PARALLELISM=0




#---- Relation Extraction ----

python benchmarks.py \
--dataset re \
--base_model llama2 \
--peft_model ../finetuned_models/finred-llama2-linear_202310012254 \
--batch_size 8 \
--max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/finred-chatglm2-linear_202310010213 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model qwen \
# --peft_model ../finetuned_models/finred-qwen-linear_202310010502 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model mpt \
# --peft_model ../finetuned_models/finred-mpt-linear_202310010641 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model bloom \
# --peft_model ../finetuned_models/finred-bloom-linear_202310010741 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model falcon \
# --peft_model ../finetuned_models/finred-falcon-linear_202310010333 \
# --batch_size 1 \
# --max_length 512


#---- Generalization ----


# python benchmarks.py \
# --dataset fiqa_mlt \
# --base_model falcon \
# --peft_model ../finetuned_models/GRCLS-sentiment-falcon-linear-small_202309291801/checkpoint-300 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb_mlt \
# --base_model llama2 \
# --peft_model ../finetuned_models/GRCLS-sentiment-llama2-linear-small_202309290356/checkpoint-800 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fiqa_mlt \
# --base_model qwen \
# --peft_model ../finetuned_models/GRCLS-sentiment-qwen-linear-small_202309292115/checkpoint-700 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb_mlt \
# --base_model mpt \
# --peft_model ../finetuned_models/GRCLS-sentiment-mpt-linear-small_202309300359/checkpoint-400 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fiqa_mlt \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/GRCLS-sentiment-chatglm2-linear-1e-4lr_202309280440/checkpoint-212 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fiqa_mlt \
# --base_model bloom \
# --peft_model ../finetuned_models/GRCLS-sentiment-bloom-linear-small_202309300044/checkpoint-500 \
# --batch_size 8 \
# --max_length 512




#---- Multi-Task ----

# python benchmarks.py \
# --dataset re \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/MT-chatglm2-linear_202309201120 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model falcon \
# --peft_model ../finetuned_models/MT-falcon-linear_202309210126 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model bloom \
# --peft_model ../finetuned_models/MT-bloom-linear_202309211510 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model qwen \
# --peft_model ../finetuned_models/MT-qwen-linear_202309221011 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model mpt \
# --peft_model ../finetuned_models/MT-mpt-linear_202309230221 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset re \
# --base_model llama2 \
# --peft_model ../finetuned_models/MT-llama2-linear_202309241345 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/MT-chatglm2-linear_202309201120 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model falcon \
# --peft_model ../finetuned_models/MT-falcon-linear_202309210126 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model bloom \
# --peft_model ../finetuned_models/MT-bloom-linear_202309211510 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model qwen \
# --peft_model ../finetuned_models/MT-qwen-linear_202309221011 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model mpt \
# --peft_model ../finetuned_models/MT-mpt-linear_202309230221 \
# --batch_size 8 \
# --max_length 512

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi,headline,ner,re \
# --base_model llama2 \
# --peft_model ../finetuned_models/MT-llama2-linear_202309241345 \
# --batch_size 8 \
# --max_length 512


#---- ConvFinQA ----

# python benchmarks.py \
# --dataset convfinqa \
# --base_model falcon \
# --peft_model ../finetuned_models/convfinqa-falcon-linear_202309170614 \
# --batch_size 1 \
# --max_length 2048

# python benchmarks.py \
# --dataset convfinqa \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/convfinqa-chatglm2-linear_202309170247 \
# --batch_size 1 \
# --max_length 2048

# python benchmarks.py \
# --dataset convfinqa \
# --base_model qwen \
# --peft_model ../finetuned_models/convfinqa-qwen-linear_202309171029 \
# --batch_size 1 \
# --max_length 2048

# python benchmarks.py \
# --dataset convfinqa \
# --base_model bloom \
# --peft_model ../finetuned_models/convfinqa-bloom-linear_202309171502 \
# --batch_size 1 \
# --max_length 2048

# python benchmarks.py \
# --dataset convfinqa \
# --base_model llama2 \
# --peft_model ../finetuned_models/convfinqa-llama2-linear_202309162205 \
# --batch_size 1 \
# --max_length 2048


#---- FinEval ----

# python benchmarks.py \
# --dataset fineval \
# --base_model falcon \
# --peft_model ../finetuned_models/fineval-falcon-linear_202309220409 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/fineval-chatglm2-linear_202309220332 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model qwen \
# --peft_model ../finetuned_models/fineval-qwen-linear_202309220508 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model bloom \
# --peft_model ../finetuned_models/fineval-bloom-linear_202309220639 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model mpt \
# --peft_model ../finetuned_models/fineval-mpt-linear_202309220555 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model llama2 \
# --peft_model ../finetuned_models/fineval-llama2-linear_202309192232 \
# --batch_size 1

# python benchmarks.py \
# --dataset fineval \
# --base_model internlm \
# --peft_model ../finetuned_models/fineval-internlm-linear_202309211248 \
# --batch_size 1


#---- NER ----

# python benchmarks.py \
# --dataset ner \
# --base_model falcon \
# --peft_model ../finetuned_models/ner-falcon-linear_202309160320 \
# --batch_size 1

# python benchmarks.py \
# --dataset ner \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/ner-chatglm2-linear_202309160238 \
# --batch_size 1

# python benchmarks.py \
# --dataset ner \
# --base_model qwen \
# --peft_model ../finetuned_models/ner-qwen-linear_202309160409 \
# --batch_size 1

# python benchmarks.py \
# --dataset ner \
# --base_model bloom \
# --peft_model ../finetuned_models/ner-bloom-linear_202309160530 \
# --batch_size 1

# python benchmarks.py \
# --dataset ner \
# --base_model mpt \
# --peft_model ../finetuned_models/ner-mpt-linear_202309160459 \
# --batch_size 1

# python benchmarks.py \
# --dataset ner \
# --base_model llama2 \
# --peft_model ../finetuned_models/ner-llama2-linear_202309161924 \
# --batch_size 1

#---- sentiment analysis ----

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model llama2 \
# --peft_model ../finetuned_models/sentiment-llama2-linear_202309130723 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model falcon \
# --peft_model ../finetuned_models/sentiment-falcon-default_20230911055454 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/sentiment-chatglm2-default_20230910031650 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model qwen \
# --peft_model ../finetuned_models/sentiment-qwen-linear_202309132016 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model internlm \
# --peft_model ../finetuned_models/sentiment-internlm-linear_202309130230 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model bloom \
# --peft_model ../finetuned_models/sentiment-bloom-linear_202309151934 \
# --batch_size 8

# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --base_model mpt \
# --peft_model ../finetuned_models/sentiment-mpt-linear_202309151405 \
# --batch_size 8


#---- headline ----

# python benchmarks.py \
# --dataset headline \
# --base_model llama2 \
# --peft_model ../finetuned_models/headline-llama2-linear_202309140611 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model chatglm2 \
# --peft_model ../finetuned_models/headline-chatglm2-linear_202309140941 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model internlm \
# --peft_model ../finetuned_models/headline-internlm-linear_202309140308 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model falcon \
# --peft_model ../finetuned_models/headline-falcon-linear_202309141852 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model qwen \
# --peft_model ../finetuned_models/headline-qwen-linear_202309142156 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model mpt \
# --peft_model ../finetuned_models/headline-mpt-linear_202309150151 \
# --batch_size 8

# python benchmarks.py \
# --dataset headline \
# --base_model bloom \
# --peft_model ../finetuned_models/headline-bloom-linear_202309151641 \
# --batch_size 8