---
layout: distill
title: ICLR
description: 'A blog for Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (arXiv 2408)'
tags: RL, LLM
date: 2024-09-23
featured: false
related_publications: true

authors:
  - name: Runze Liu
    affiliations:
      name: Tsinghua University

bibliography: blogs.bib

toc:
  - name: Summary
    subsections:
      - name: Motivation
      - name: Key Idea
      - name: Contributions
  - name: Preliminaries
  - name: Method
  - name: Implementation
  - name: Experiments
  - name: References

_styles: >
  mjx-container[jax="CHTML"][display="true"] {
    margin-top: 0em !important;
    margin-bottom: 1em !important;
  }
---

## Summary

Code: 

这篇论文提出了 <d-cite key="DPO"></d-cite>

### Motivation

- 让LLM使用更多test-time computation，而不是增加模型参数，可以更有效地提升模型性能。

### Key Idea

- 构造一个推理 (rationale) 数据集，构造方法：对于模型能回答对的问题，提示模型进行推理，最终得到正确答案；对于回答错误的问题，将正确答案提供给模型，让模型根据答案逆向推理，得到推理过程。将模型输出的推理过程加入数据集，如此迭代，不断提升模型的推理能力。

### Contributions

1. 
2. 
3. 

## Preliminaries

### 

$$
\begin{equation}
1
\end{equation}
$$


## Method

### 




## Implementation

```python

```

## Experiments
