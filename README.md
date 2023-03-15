# new_rere
对比三种模式下的三元关系抽取效果。
预训练模型使用bart-Chinese-base模型

模式一：直接在bart上微调后抽取；
模式二：使用数据集cndbpedia（147w）、ske2019（17w）、HacRED（6k）对bart进行继续预训练后抽取；
模式三：对继续预训练后的模型进行微调后抽取。
