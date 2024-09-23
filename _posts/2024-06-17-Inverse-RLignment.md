---
layout: distill
title: RLHF - Inverse-RLignment
description: 'A blog for Inverse-RLignment: Inverse Reinforcement Learning from Demonstrations for LLM Alignment'
tags: RLHF, LLM
date: 2024-06-01
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

- 

### Key Idea

- 

### Contributions

1. 
2. 
3. 

## Preliminaries

### 

\section{Alignment Beyond Preference Data and Supervised Fine Tuning}
In this section, we present our central insight: the LLM alignment problem can be framed within the context of \textit{forward and inverse} RL, suggesting it can be addressed using corresponding methodologies. To ensure this section is self-contained, we provide the necessary preliminaries and background concepts in the \grayboxtext{gray text boxes}.


The section is organized as follows: 
In Section~\ref{sec:arllm-seqdecmak}, we elaborate on the sequential decision-making nature of auto-regressive LLM generation.
In Section~\ref{sec:alignmentasonlinerl}, we discuss the challenge of missing reward signals in LLM alignment and the difficulties associated with current solutions.
In Section~\ref{sec:alignmentBC}, we present the perspective that AfD can be formulated as an Inverse RL problem, highlighting the potential solutions from such a perspective.

### Auto-Regressive Language Generation as Sequential Decision Making
\label{sec:arllm-seqdecmak}

We first cast auto-regressive language generation into the Markov Decision Processes framework for sequential decision-making.

\paragraph{Markov Decision Processes (MDP)}
In Markov Decision Processes, decisions are made in discrete time steps and affect the state of the environment in the subsequent step.
Formally, an MDP is denoted as $$\mathcal{M} = \{\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\rho_0,\gamma\}$$, where $$\mathcal{S}\subset \mathbb{R}^{d}$$ denotes the $$d$$-dim state space, $$\mathcal{A}$$ is the action space. Broadly, the environment includes $$\mathcal{T}$$ and $$\mathcal{R}$$, the former denotes the transition dynamics $$\mathcal{T}: \mathcal{S}\times \mathcal{A} \mapsto \Delta(\mathcal{S})$$ that controls transitions between states, and the reward function $$\mathcal{R}:\mathcal{S}\times\mathcal{A}\mapsto \mathbb{R}$$ provides feedback. $$\rho_0 = p(s_0)\in\Delta(\mathcal{S})$$ denotes the initial state distribution. $$\gamma$$ is the discount factor that trades off between short-term and long-term returns.

In the context of the token-generation process in LLMs, let $$C$$ denote the context window size and $$\mathcal{V}$$ denote the vocabulary, including the special tokens like \texttt{[EOS]} and \texttt{[MASK]}. The MDP is instantiated as follows: 
State space $$\mathcal{S} = \mathcal{V}^C$$; action space $$\mathcal{A}=\mathcal{V}$$; transition dynamics is \textbf{deterministic and known}: $$s' = \mathcal{T}(s,a) = \texttt{Concat}(s,a) = [s, a] $$; We consider states containing an \texttt{[EOS]} token as absorbing states, meaning $$ \forall a: s' = \mathcal{T}(s,a) = s ~\textrm{if}~ \texttt{[EOS]}\in s$$; 
an LLM $$\ell$$, serving as policy $$\pi = \ell$$, generates the next token $$a\in\mathcal{A}$$ based on the current context $$s\in\mathcal{S}$$; The initial state distribution of queries is $$\rho_0$$, and $$T$$ represents the maximal number of new tokens in a generation. i.e., $$T$$ is the maximal number of transitions in the MDP.
For instance, in the following case, the context window length $$C\ge7$$ and $$T=2$$, an initial state $$s_0$$ is given as follows:

$$
\begin{equation}
  s_0 = \big[\texttt{ The | color | of | the | sky |\hspace{1pt}[MASK]\hspace{1pt}|\hspace{1pt}[MASK]}\big],
\end{equation}
$$

when the language model policy $$\pi$$ selects a new token ``$$\texttt{is}$$'' from the vocabulary $$\mathcal{V}$$, the next state deterministically becomes

$$
\begin{equation}
  s_1 = \texttt{Concate}(s_0, a_0=\texttt{is})= \big[\texttt{ The | color | of | the | sky | is |\hspace{1pt}[MASK]}\big],
\end{equation}
$$

the generation process continues until either the \texttt{[EOS]} token is selected, the maximal context window size is reached, or the maximal decision steps $$T$$ is reached. In this example, the final generated context could be:

$$
\begin{equation}
  s_2 = \texttt{Concate}(s_1, a_1=\texttt{blue}) = \big[\texttt{ The | color | of | the | sky | is | blue }\big].
\end{equation}
$$


### Challenge of the Alignment MDP: Getting Reward Signals is Hard
\label{sec:alignmentasonlinerl}

The research on LLM alignment focuses on aligning language models with users' intentions during response generation~\cite{ouyang2022training}.
Within the MDP framework, users' intentions are represented by a reward model $$\mathcal{R}$$, which provides feedback on the LLM's outputs, evaluating aspects such as helpfulness, truthfulness, and harmlessness of the generated content.
Typically, evaluations are performed at the trajectory level, meaning feedback is provided only after the entire generation process is complete:

$$
\begin{equation}\label{eq:base_reward_function}
  \mathcal{R}(s_t,a_t) = \left\{
  \begin{array}{ll}
      r(s_t) & \text{if } s_t \text{ is a terminal state}, t=T  \\
      0 & \text{otherwise}.
  \end{array}
\right.
\end{equation}
$$

Ideally, human users would provide feedback for each response, allowing conventional online RL algorithms to optimize the policy $$\pi =\ell$$ through

$$
\begin{equation}\label{eq:online-rl}
  \pi^* = \arg\max_{\pi\in\Pi}\mathbb{E}_{a_t\sim\pi, s_{t+1}\sim \mathcal{T},s_0\sim \rho_0}\sum_{t=0}^T \gamma^t \mathcal{R}(s_t, a_t) = \arg\max_{\pi\in\Pi}\mathbb{E}_{a_t\sim\pi, s_{t+1}\sim \mathcal{T},s_0\sim \rho_0}r(s_T),
\end{equation}
$$

However, a significant challenge in LLM alignment is \textbf{the difficulty in defining reward signals}, as the desired user intentions are not easily accessible. In prevailing LLM alignment approaches, reward models are typically derived from preference-based annotations.

\textbf{Learning Reward Models from Preference Annotations.} Most recent advancements in LLM alignment rely on preference-based datasets of the form $$\mathcal{D}_\mathrm{pref} = \{x_i, y_i^+, y_i^-\}_{i\in[N]}$$, where $$y_i^+$$ and $$y_i^-$$ are the preferred and dis-preferred responses given input $$x_i$$. Models such as Bradley-Terry~\cite{bradley1952rank} are then used to convert ranking feedback into absolute scores to serve as reward signals. Thus, we call the reward model built with a preference-based dataset the Bradley-Terry Reward Model (BT-RM).
As has been discussed earlier, these datasets pose several challenges, including noisy labels~\cite{azar2023general,zheng2023secrets}, high costs~\cite{guo2024direct,xiong2023gibbs,touvron2023llama,tang2024understanding}, the requirement of additional assumptions in transferring rank to scores~\cite{azar2023general,munos2023nash,bradley1952rank,ethayarajh2024kto,rafailov2023direct}~\footnote{see further analysis in Appendix~\ref{sec:bradley-terry}}, and privacy concerns.



% However, despite online feedback can be superior than offline methods, in practice, learning from offline feedback data is still the main stream.
% asking human users to provide online feedback scores on language model-generated responses is unrealistic

% e.g., the scores may not be consistent and the standard may vary across different users. To circumvent such a difficulty, previous research on LLM alignment primarily focused on the preference-based datasets  %The recent advancements of RLHF demonstrate the superiority of using online feedback in alignment~\cite{tang2024understanding}, yet the cost 

% \paragraph{Offline Preference Data} Another type of data, which is widely studied in the literature, is the preference dataset labeled by human annotators $$\mathcal{D}_\mathrm{pref} = \{x_i, y_i^+, y_i^-\}_{i\in[N]}$$. In such a dataset, multiple responses are generated by the LLM policy --- rather than experts --- and then ranked by annotators. As discussed earlier, challenges for those datasets include noisy labels, high costs, the requirement of additional assumptions in transferring rank to scores, and privacy concerns. 
% \begin{graybox}
%     \paragraph{Offline RL}
% In the \textit{Offline RL} setting, interactions with the environment are strictly forbidden. The learning problem is no longer online learning but learning from a static dataset of decision logs $$\mathcal{D}_{\mathrm{Offline}} = \{(s^i_t,a^i_t,s^i_{t+1},r^i_t)\}$$, that is generated by some unknown behavior policy $$\pi_\beta$$.

% The most obvious difficulty in the offline RL setting is such a setting prohibits exploration --- hence it hinders the improvement of policy learning to be improved over the demonstration data.
% \end{graybox}

### Alignment from Demonstrations: an Alternative to Preference-based Reward Modeling
\label{sec:alignmentBC}

In RL research, learning from human feedback through preference is not the only option when reward signals are unknown or difficult to design~\cite{plappert2018multi}. Learning from a demonstrative behavioral dataset has been widely applied in various domains, including robotics control~\cite{schaal1996learning,nair2018overcoming,hester2018deep}, autonomous driving~\cite{kuderer2015learning,scheel2022urban}, video game playing~\cite{vinyals2019grandmaster}, and AlphaGo~\cite{silver2016mastering}.
Formally, with a demonstration dataset containing paired states and high-quality actions: $$\mathcal{D}_\mathrm{demo} = \{s_i, a_i^*\}_{i\in[N]}$$, the most direct approach, Behavior Cloning~\cite{pomerleau1991efficient}, learns the policy through supervised learning:

\paragraph{Behavior Cloning (BC)} 
A demonstrative decision dataset is collected from a behavior policy $$\pi_\beta$$. Denoting the state-action pairs in the dataset as $$(s_i, a^*_i)$$, the BC method learns a policy through a supervised learning objective:
$$
\begin{equation}
  \pi_\mathrm{BC} = \arg\max_\pi \mathbb{E}_{(s_i,a_i)\sim\mathcal{D}_\mathrm{demo}} \log(\pi(a_i|s_i))
\end{equation}
$$



\textbf{Supervised Fine Tuning: Behavior Cloning for AfD.}
In the context of LLM alignment, demonstrations in the form of $$\mathcal{D}_\mathrm{SFT} = \{x_i, y_i^*\}_{i\in[N]}$$ are also referred to as the Supervised Fine Tuning (SFT) dataset. This format is versatile: for example, $$x$$ can be a general query for Question-Answering tasks, an incomplete sentence for completion tasks, or a general instruction for instruction following tasks; Correspondingly, $$y^*$$ represents the desired answers, a completed sentence, or a response following the instruction. 
% including many private datasets that could not be shared to external human or LLM annotators. 
Such datasets are widely applied for SFT training, where the learning objective is to minimize the token-wise difference given the existing context. To clarify our notations for further discussion, consider the following example of a context-response pair $$x_i, y^*_i$$:

$$
\begin{equation}
\begin{split}
    x_i &= \big[\texttt{ What | is | the | 
 color | of | the | sky? }\big], \\
    y^*_i &= \big[\texttt{ The | color | of | 
 the | sky | is | blue }\big].
\end{split}
\end{equation}
$$

the SFT training first reorganizes the dataset $$\mathcal{D}_\mathrm{SFT}$$ to state-action pairs ($$\mathcal{D}_\mathrm{demo}$$) as follows:

$$
\begin{equation*}
\begin{split}
    s_0 &= \big[\texttt{ What | is | the | color | of | the | sky?~|\hspace{1pt}[MASK]|\hspace{1pt}[MASK]|\hspace{1pt}[MASK]|...}\big], \\
    a^*_0 &= \texttt{ The }, \\
    s_1 & = \big[\texttt{ What | is | the | 
 color | of | the | sky?~| The |\hspace{1pt}[MASK]\hspace{1pt}|\hspace{1pt}[MASK]|...}\big], \\
    a^*_1 &= \texttt{ color } , \\
    s_2 & = \big[\texttt{ What | is | the | 
 color | of | the | sky?~| The | color |\hspace{1pt}[MASK]\hspace{1pt}|...}\big], \\
    a^*_2 &=  \texttt{ of } , \\
    &...
\end{split}
\end{equation*}
$$

with such a dataset, the learning objective is to reproduce the demonstration token $$a^*_i $$ when the LLM (policy) is given $$s_i$$ (incomplete token sequences). The training of the SFT is conducted through supervised classification.
  
\textbf{AfD Beyond Supervised Fine Tuning.}
While BC is conceptually simple and easy to implement, it faces a fundamental challenge known as the \textit{distributional shift} ---  during evaluation, the state distribution is generated by rolling out the learned policy $$\pi$$, rather than the data-generation behavior policy $$\pi_\beta$$. 
To address this challenge, Imitation Learning (IL) and Inverse RL consider scenarios where the \textit{dynamics model} is available to generate roll-out samples during learning~\cite{pomerleau1991efficient, finn2016guided, abbeel2004apprenticeship}. For a more detailed discussion on the benefits of accessing dynamics models, refer to Appendix~\ref{appdx:extended_preliminary}.

% To alleviate such a challenge, Imitation Learning (IL) considers the setting where the \textit{\textbf{dynamics model}} is available to generate roll-out samples during learning~\cite{pomerleau1991efficient,finn2016guided,abbeel2004apprenticeship}. We refer interested readers to Appendix~\ref{appdx:extended_preliminary} for more discussion on the benefits of having access to dynamics models.
 

At first glance, aligning LLMs with an offline demonstration dataset might seem like an offline RL problem, as no further interactions with human annotators are available during training. However, it is the accessibility of online interactions with the \textbf{\textit{dynamics model}}, rather than the reward model, that determines the online or offline nature of the tasks. In LLM alignment practices, while accessing reward models (online annotators) during training is impossible, \textbf{the dynamics model in response generation is known and accessible} --- the actions are tokens generated by LLMs, and the responses (trajectories) are concatenations of those generated tokens. This insight naturally leads us to explore alternative approaches rooted in the IL and Inverse RL literature.
In Table~\ref{tab:alihan-dan} of Appendix~\ref{appdx:different_RL_approaches}, we contextualize the difference and link between various topics in the RL literature.
 
% With the notations and connections we established above, we now introduce a unified objective class using trajectory distribution matching --- a widely studied objective in the IL literature~\cite{jarrett2020strictly,ho2016generative,ghasemipour2020divergence} for the AfD problem. 

Building on the notations and connections established above, we now introduce a unified objective class using trajectory distribution matching, a widely studied objective in the IL and Inverse RL literature~\cite{jarrett2020strictly, ho2016generative, ghasemipour2020divergence}, for the AfD problem.




## Method

### 

\section{Algorithms for Alignment from Demonstrations}
% : from Behavior Cloning to Adversarial Imitation
\label{sec:imitation}
% In Section~\ref{sec:alignmentwithimitation}, we present the perspective that LLM alignment can be formulated as an imitation learning problem.
% % and introduce practical algorithms that circumvent the requirement of expensive preference data assumed in prevailing LLM alignment literature in Section~\ref{sec:imitation}; \\
% In Section~\ref{sec:imitation}, we introduce practical algorithms that circumvent the need for expensive preference data, which is commonly assumed in the prevailing LLM alignment literature.
% % finally, we disclose and solve the potential pitfall of reward (discriminator) modeling in Section~\ref{sec:reward_modeling}.
% Finally, in Section~\ref{sec:reward_modeling}, we address and resolve potential pitfalls in reward (discriminator) modeling.


### Alignment from Demonstration through Trajectory Distribution Matching
\label{sec:distribution_matching}

Unlike the action distribution matching objective used in BC, when the dynamics model is accessible, it is beneficial to study the occupancy matching problem to enhance the performance of learning from the offline demonstrations~\cite{ho2016generative, ross2011reduction, fu2017learning, orsini2021matters}. 
Specifically, we denote the state-action occupancy measure of the behavior policy (i.e., the demonstrator) as $$\rho^\beta(s,a) = \pi_\beta(a|s)\sum_{t=0}\gamma^t \mathrm{Prob}(s_t = s|\pi_\beta)$$, and the state-action occupancy measure of the current policy as $$\rho^\pi(s,a)$$. Intuitively, the occupancy measure describes the distribution of state-action pairs visited by an agent under a given policy.
For auto-regressive LLMs that take context $$x$$ as input and output response $$y = (y^{(0)},y^{(1)},...,y^{(T)}=\texttt{EOS} )$$ containing a maximum of $$T+1$$ tokens, we have

$$
\begin{equation}
\begin{split}
  \rho^\pi(s_k,a_k) &= \rho^\pi(s_k = (x,y^{(0:k-1)}),a_k= y^{(k)})  \\
  &= \pi(a_k = y^{(k)}| s_k = (x,y^{(0:k-1)}))p(s_{k}) \\
  &= \pi(a_k = y^{(k)}| s_k = (x,y^{(0:k-1)})) \pi(a_{k-1} = y^{(k-1)}| s_{k-1} = (x,y^{(0:k-2)})) p(s_{k-1}) \\
  &=...\\
  &= p(s_0)\Pi^{t=k}_{t=0} \pi(a_t = y^{(t)}| s_t = (x,y^{(0:t-1)}))
\end{split}
\end{equation}
$$

In alignment, we are motivated to study the completed generations. Therefore, it is useful to denote the trajectory distribution $$d^\pi(y|x)$$ as the occupancy measure of completed generations conditioned on input context $$x$$ (i.e., final state occupancy conditioned on initial state):

$$
\begin{equation}
  d^\pi(y|x)=\Pi^{t=T}_{t=0} \pi(a_t = y^{(t)}| s_t = (x,y^{(0:t-1)})) = \rho^\pi(s_{{T}},a_{{T}})/p(x)
\end{equation}
$$

Practically, we can sample from the above conditional distribution by rolling out the policy $$\pi$$, and approximately sample from the behavior policy using the demonstration dataset:

$$
\begin{equation}
  d^\beta(y|x)=\Pi^{t=T}_{t=0} \pi_\beta(a_t = y^{(t)}| s_t = (x,y^{(0:t-1)})) = \rho^\beta(s_{{T}},a_{{T}})/p(x)
\end{equation}
$$

In the following, we derive different objectives for LLM alignment from the perspective of divergence minimization between the demonstration conditional distribution and the roll-out conditional distribution. Specifically, we study the minimization of Forward KL-Divergence and Reverse KL-Divergence in the main text, as they are the most commonly used and provide sufficient insights into the proposed objectives. We additionally discuss a more general framework in Appendix~\ref{appdx:f-div}.
% To demonstrate the generality of the framework, we discuss other divergence choices in 

% \begin{equation}
%     (x,y)\sim d^\mathrm{exp}(y|x)\approx (x,y)\sim \mathcal{D}_\mathrm{demo}
% \end{equation}



% \paragraph{\textcolor{brown}{{Case 1: AfD with Behavior Cloning: Supervised Fine-Tuning}}}
% The learning objective of SFT is to minimize the negative log-likelihood of generating expert-generated tokens given the existing context 
% \begin{equation}
% \label{eqn:7}
%     \min_\pi \mathbb{E}_{(s,a)\sim\rho^\mathrm{exp}} \left[\mathrm{KL}(\pi^\mathrm{exp}(a|s)||\pi(a|s)) \right] = - \max_\pi \mathbb{E}_{(s,a)\sim\rho^\mathrm{exp}} \left[\log(\pi(a|s)) \right]
% \end{equation}
% Therefore, the conventional SFT training objective minimizes the KL divergence of \textbf{action marginal distribution} between the behavior policy $$\pi^\mathrm{exp}$$ and the current policy $$\pi$$.


\textbf{AfD through Divergence Minimization using Forward KL.}
We first consider the objective using the forward KL divergence between the demonstration and policy conditional trajectory distributions:

$$
\begin{equation}
\begin{split}
\label{eqn:7_true}
  \min_\pi \left[\mathrm{KL}(d^\beta(y|x)||d^\pi(y|x)) \right] &= - \max_\pi \mathbb{E}_{(x,y)\sim \mathcal{D}_\mathrm{SFT}} \left[\log d^\pi(y|x) \right] \\
  &= - \max_\pi \mathbb{E}_{(x,y^{(0:K)})\sim\mathcal{D}_\mathrm{SFT}} \left[\sum^{K}_{t=0}\log \pi(a_t|s_t) \right].
\end{split}
\end{equation}
$$

Comparing the derived objective with the SFT objective, which minimizes the negative log-likelihood of tokens in the demonstration dataset given the existing context:

$$
\begin{equation}
\label{eqn:7}
  \min_\pi \mathbb{E}_{(s,a)\sim\rho^\beta} \left[\mathrm{KL}(\pi^\beta(a|s)||\pi(a|s)) \right] = - \max_\pi \mathbb{E}_{(s,a)\sim\mathcal{D}_\mathrm{demo}} \left[\log(\pi(a|s)) \right]
\end{equation}
$$

we find that both approaches yield exactly the same learning objective.
\begin{mdframed}[innertopmargin=0pt,leftmargin=0pt, rightmargin=0pt, innerleftmargin=10pt, innerrightmargin=10pt, skipbelow=0pt]
\textbf{Take-Aways:} 
Using the forward KL in \textbf{conditions trajectory distribution divergence minimization} leads to the same objective as SFT, where the training objective minimizes the KL divergence of \textbf{action marginal distribution} between $$\pi^\beta$$ and $$\pi$$.%the behavior policy $$\pi^\beta$$ and the current policy $$\pi$$. 

The forward KL divergence is known to result in mass-covering behavior, whereas the reverse KL divergence leads to mode-seeking behavior~\cite{ghasemipour2020divergence, khalifa2020distributional, wiher2022decoding, wang2023beyond}. This equivalence explains the mass-covering behavior observed in SFT in recent literature~\cite{kirk2023understanding}.
\end{mdframed}
 %\textbf{As a consequence, those SFT-type objectives are more suitable for close-ended tasks.}


% \begin{mdframed}[innertopmargin=0pt,leftmargin=0pt, rightmargin=0pt, innerleftmargin=10pt, innerrightmargin=10pt, skipbelow=0pt]
% \paragraph{Take-Aways} 
%     Comparing Equation (\ref{eqn:16}) and Equation (\ref{eqn:7}), we can conclude the following:\\ 1. Minimizing action marginal distribution between the demonstration dataset and the current policy leads to the SFT learning objective. \\
%     2. Minimizing the forward KL divergence of \textbf{trajectories} between demonstration and current policy leads to the same learning objective as SFT. \\
%     3. As it is known that using the forward KL-Divergence will lead to mass-covering and using reverse KL-Divergence leads to mode-seeking behaviors~\cite{ghasemipour2020divergence,khalifa2020distributional,wiher2022decoding,wang2023beyond}, the approaches above are all mass-covering given their equivalences. \textbf{As a consequence, those SFT-type objectives are more suitable for close-ended tasks.}
    
% \end{mdframed}

\textbf{AfD through Divergence Minimization using Reverse KL.}
In the pursuit of mode-seeking behavior, we can minimize the Reverse KL divergence, leading to the following learning objective:

$$
\begin{equation}
\begin{split}
\label{eqn:reverse-KL_traj}
  \min_\pi [\mathrm{KL}(d^\pi(y|x)||d^\beta(y|x))] = -\max_\pi \mathbb{E}_{(x,y)\sim d^\pi}\left[ \log d^\pi(y|x) - \log d^\beta(y|x) \right].
\end{split}
\end{equation}
$$

The challenge with this objective is that the second term, $$d^\beta(y|x)$$, is always unknown. This issue has been addressed in the literature through adversarial training~\cite{fu2017learning}. By training a discriminative model $$D_\phi$$, parameterized by $$\phi$$, to classify trajectories sampled from the demonstration dataset or the behavior policy $$\pi$$, we achieve

$$
\begin{equation}
\label{eqn:optimal_d_traj}
  D^*_\phi(y|x) = \frac{d^\beta(y|x)}{d^\beta(y|x)+d^\pi(y|x)}
\end{equation}
$$

at optimal convergence~~\cite{goodfellow2014generative}. 
Plugging Equation~(\ref{eqn:optimal_d_traj}) into Equation~(\ref{eqn:reverse-KL_traj}), we derive a practical policy learning objective:

$$
\begin{equation}
\label{eqn:adv_pi}
  \max_\pi \mathbb{E}_{(y|x)\sim d^\pi}\left[ \log D_\phi(y|x) - \log (1-D_\phi(y|x)) \right]
\end{equation}
$$

The discriminative mode $$D_\phi$$ can be optimized through:

$$
\begin{equation}
  \max_\phi \mathbb{E}_{(y|x)\sim \mathcal{D}_{\mathrm{SFT}}}[\log D_\phi(y|x)] +  \mathbb{E}_{(y|x)\sim d^\pi}[\log (1-D_\phi(y|x))]
\end{equation}
$$

\begin{mdframed}[innertopmargin=0pt,leftmargin=0pt, rightmargin=0pt, innerleftmargin=10pt, innerrightmargin=10pt, skipbelow=0pt]
\textbf{Take-Aways:} 
Comparing the learning objectives derived using the reverse KL divergence to the SFT objective, we see that performing mode-seeking is generally more challenging than mass-covering due to the \textbf{difficulty of estimating the probability of trajectory from the demonstrator}. This challenge can be circumvented through adversarial training.
\end{mdframed}
Despite its success, adversarial training is known to be unstable and computationally expensive~\cite{salimans2016improved, kodali2017convergence, lin2021spectral, yang2022improving}, which is particularly concerning when applied to training LLMs in the AfD context. In the next section, we leverage insights from the adversarial objective discussed above to propose a computationally efficient algorithm that avoids iterative training.

% \begin{mdframed}[innertopmargin=0pt,leftmargin=0pt, rightmargin=0pt, innerleftmargin=10pt, innerrightmargin=10pt, skipbelow=0pt]
% \paragraph{Take-Aways} 
%     Comparing the learning objectives we derived when using the reverse KL divergence or the Jensen-Shannon divergence to the SFT-type of objectives above,  \\
%     1. Performing mode-seeking is generally harder than mass-covering, which is caused by the \textbf{difficulty of estimating the probability of getting on-(current)-policy actions with the expert policy}. \\
%     2. Such a difficulty can be circumvented through adversarial training. In general, there are two choices for learning the discriminative model corresponding to identifying the state-action occupancy measure and the trajectory distribution, respectively. \\
%     3. Different from the SFT-type of learning objectives, the adversarial learning approaches do not only seek mass-covering. The superiority of such a class of approaches has been demonstrated in the low demonstration data regime~\cite{ghasemipour2020divergence}. \textbf{Consequently, the adversarial learning approaches are more suitable for open-ended tasks, especially under the low-demonstration regime.}
    
% \end{mdframed}



% \subsection*{\textcolor{red}{Implementation Matters: the target network trick}}

### Computationally Efficient Inverse RL by Extrapolating Over Reward Models
\label{sec:reward_modeling}

Conceptually, the optimization of policy in Equation~(\ref{eqn:adv_pi}) is conducted by maximizing over the inner variable, sharing the same form as Equation~(\ref{eq:online-rl}). This observation suggests using the reward notation:

$$
\begin{equation}
  r(y|x) = \log D_\phi(y|x) - \log (1-D_\phi(y|x)) 
\end{equation}
$$

Specifically, when $$D_\phi(y|x)$$ is instantiated by neural networks with sigmoid activation function over logits $$D_\phi(y|x) = \sigma(\texttt{logits}(y|x))$$, we have $$r(y|x) = \texttt{logits}(y|x)$$ --- the reward signal is provided by the discriminative model through its output logits. In the following discussion, we interchangeably use the terms reward model and discriminative model as they refer to the same concept. We call this reward model the Inverse-RL Reward Model, abbreviated as IRL-RM.


Inspired by the previous success achieved in the Inverse RL literature that extrapolates learned reward models~\cite{brown2019extrapolating}, we propose to circumvent the difficulty in iterative generative adversarial training through reward model extrapolation. 
Initially, one might build a reward model using samples from the demonstration dataset as positive examples and samples generated by the initial LLM policy as negative examples for discriminator training.


\begin{table}[t!]
\fontsize{7.3}{11}\selectfont
\vspace{-0.15cm}
\centering
\caption{\small \textit{Comparison of multiple reward modeling choices.} The first three rows are choices in building reward models in AfD using different datasets for the discriminative model training.}
\vspace{-0.15cm}
\begin{tabular}{l|c|c|c|c}
\toprule
\textbf{Dataset for RM}            & \textbf{Negative Example Source} & \textbf{Positive Example Source} & \textbf{Format of Data} & \textbf{Heterogeneity in RM} \\ \hline
\textcolor{cyan}{\textbf{Init-SFT RM}}& $$(y|x)\sim\pi_\mathrm{init}$$      & $$(y|x)\sim\pi_\mathrm{SFT}$$     & AfD                     &       Low                           \\ 
\textcolor[rgb]{0.9, 0.73, 0.1}{\textbf{Init-Demo RM}} & $$(y|x)\sim\pi_\mathrm{init}$$      & $$(y|x)\sim\mathcal{D}_\mathrm{demo}$$         & AfD                     &       High                           \\ 
\textcolor{red}{\textbf{SFT-Demo RM}}& $$(y|x)\sim\pi_\mathrm{SFT}$$       & $$(y|x)\sim\mathcal{D}_\mathrm{demo}$$         & AfD                     &       High(er) \\
\textbf{Preference-based} & Dispreferred                         & Preferred                          & Pair-wise    &    No                              \\ 
\bottomrule
\end{tabular}
\label{tab:rm_choice}
\vspace{-0.5cm}
\end{table}
Nevertheless, in the AfD problem, the demonstration dataset is typically generated by external demonstrators, such as human experts or more advanced LLMs, rather than the LLM being aligned. This \textbf{heterogeneity} can introduce significant bias in the reward modeling step, potentially leading to reward hacking~\cite{skalse2022defining, gao2023scaling, zhang2024overcoming, coste2023reward}. The reward model may focus on the heterogeneity of responses --- for discrimination --- rather than on the informative aspects that truly evaluate the quality of responses in terms of human intention.

\begin{wrapfigure}{r}{0.49\textwidth}
  \centering
  \vspace{-0.1cm}
  \includegraphics[width=0.5\textwidth]{figs/rm_fig.png}
  \vspace{-0.2cm}
  \caption{\small \textit{Illustration of different choices for positive and negative samples in Inverse-RL reward modeling.} The LLM to be aligned is restricted to a specific model class, limiting its expressivity and capability. This limitation is depicted by allowing improvements only along the x-axis. For example, SFT training on the demonstration dataset can push the initial model $$\pi_0$$ toward higher scores. The y-axis represents the heterogeneous nature of the demonstration dataset in AfD problems, where the behavior policy $$\pi_\beta$$ always differs from the LLM to be aligned. Notably, $$\pi_\beta$$ could be human experts or stronger general-purpose LLMs.}
  \label{fig:RM}
  \vspace{-0.4cm}
\end{wrapfigure}
It is important to note that in our context, the reward model is trained to differentiate the origins of various responses. \textbf{A discriminator that primarily detects subtle differences due to model heterogeneity is not effective as a reward model for providing meaningful improvement signals for alignment.}


% \begin{figure}[h!]
%     \centering
%     \vspace{-0.43cm}
%     \includegraphics[width=0.5\linewidth]{figs/rm_fig.png}
%     \caption{\small \textit{Illustration of different choices on the positive and negative samples in the Inverse-RL reward modeling.} As the LLM to be aligned is restricted to a specific model class, its expressivity and ability could be limited, which is illustrated by only permitting its improvement over the x-axis, e.g., SFT training on the demonstration dataset can push the initial model $$\pi_0$$ toward higher scores. The vertical axis illustrates the heterogeneity nature of the demonstration dataset in AfD problems --- the behavior policy $$\pi_\beta$$ always differs from the LLM to be aligned. $$\pi_\beta$$ could be generated by human experts of by stronger general-purpose LLMs.}
%     \label{fig:RM}
%     \vspace{-0.25cm}
% \end{figure}



To address this challenge, we propose using a different dataset format for building our reward model. Instead of using the demonstration dataset as positive samples, we use the samples generated by the SFT policy $$\pi_\mathrm{SFT}$$, trained on the demonstration dataset, as positive examples. The samples generated by the initial LLM policy $$\pi_0$$ serve as negative examples. This approach alleviates the heterogeneity issue that arises when naively combining demonstration samples with $$\pi_0$$-generated samples. 
Table~\ref{tab:rm_choice} contrasts the different data choices for reward model training. Figure~\ref{fig:RM} visualizes and illustrates their differences.
To further explain and contrast different approaches:
\begin{itemize}[leftmargin=*,nosep]
    \item \textcolor[rgb]{0.9, 0.73, 0.1}{\textbf{Init-Demo RM}}: Using samples generated by $$\pi_0$$ as negative examples and demonstration dataset samples as positive examples in reward model training is straightforward. However, as $$\pi_0$$ and $$\pi_\beta$$ are heterogeneous models, so nuanced differences, such as specific verb usage or response formats in $$\pi_\beta$$ can dominate reward model learning rather than the desired alignment properties.
    \item \textcolor{red}{\textbf{SFT-Demo RM}}: Using samples generated by $$\pi_\mathrm{SFT}$$ examples and demonstration dataset samples as positive examples faces the same challenge. Moreover, since $$\pi_\mathrm{SFT}$$ and $$\pi_\beta$$ are closer in terms of the desired properties to align (scores), reward hacking is even more likely.
    \item \textcolor{cyan}{\textbf{Init-SFT RM}}: To avoid potential reward hacking caused by using heterogeneous data in reward model training, we can use samples generated by $$\pi_0$$ as negative examples and samples generated by $$\pi_\mathrm{SFT}$$ as positive examples. Unlike the previous approaches, where positive and negative examples are generated by heterogeneous models, these two models are homogeneous since the SFT policy is fine-tuned from the initial policy.
    \item \textbf{Preference-based RM} (BT-RM): In preference-based reward modeling, both preferred and dis-preferred responses are samples from the same LLM~\cite{ouyang2022training}. Therefore, there is no issue of heterogeneity between the positive and negative samples.
\end{itemize}
When applying the learned reward models at inference time to determine which responses are superior, these responses are generated by $$\pi_\mathrm{SFT}$$, therefore, the \textcolor{cyan}{\textbf{Init-SFT RM}} should outperform other choices. 
In the next section, we provide empirical studies to verify our insights.



## Implementation

### Algorithm

1. **Require**: The initial LLM policy $$\pi_0$$, the SFT policy $$\pi_{\mathrm{SFT}}$$, the demonstration dataset $$\mathcal{D}_\mathrm{demo}$$ generated by the behavior policy $$\pi_{\beta}$$, and the reward model $$D_\phi$$.
2. **Initialization**: Initialize the LLM policy $$\pi_0$$ and the reward model $$D_\phi$$.
3. **For $$t=1,\cdots,T$$**:
    1. **Policy Update**: Update the policy $$\pi$$ by maximizing the reward signal provided by the reward model $$D_\phi$$.
    2. **Reward Model Update**: Update the reward model $$D_\phi$$ by minimizing the binary cross-entropy loss between the predicted labels and the ground-truth labels.
4. **Termination**: Repeat the training loop until convergence.
5. **Inference**: Use the learned policy $$\pi$$ for response generation.
6. **Evaluation**: Evaluate the policy performance using the reward model $$D_\phi$$.


```python

```

## Experiments
