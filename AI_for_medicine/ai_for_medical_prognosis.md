# AI for medical prognosis 
Based on [Deeplearning.ai course](https://www.coursera.org/learn/ai-for-medical-prognosis/home/welcome).

**Prognosis** is a term that refers to predicting the **risk** of a future **event**. Here, "event" is a general term that captures a variety of things that can happen to an individual. Events can include outcomes such as death and other adverse events like a heart attack or a stroke, which might be risks for patients who have a specific medical condition or for the general population.  

We can think of **prognostic models** as a system that takes in a **profile** of a patient as input and outputs a **risk score** for that patient. The patient profile can include clinical history, which includes major illnesses or any previous procedures. Profile might also include physical exam findings such as vital signs including temperature and blood pressure. Finally, there are lab tests such as complete blood count and imaging that includes CT scan and others.  

Many models for medical prognosis combine fixed coefficients (weights) with given features to output a score. These models do not use Machine learning and were built by specialists based on their knowledge. For example: Atherosclerotic Cardiovascular Disease (ASCVD) risk estimator, Model for End-stage Liver Disease (MELD).

## Evaluating prognostic models
The basic idea behind evaluating a prognostic model is to see how well it performs on pairs of patients by comparing the risk scores it assigns to these pairs with the **outcomes** (e.g. death or not).  
- **Concordant pair**: when the patient with the worst outcome has the higher risk score. 
- **Not concordant pair**: when the patient with the worst outcome does not have a higher risk score. 
- **Risk tie**: if there are different outcomes, but a tie in the risk score. In this case, the pair is neither concordant nor discordant.
- **Outcome tie**: if outcomes are similar, the pair can not be used to determine who should have a higher risk score. Thus in the evaluation of prognostic models, we only consider pairs with different outcomes. A pair where the outcomes are different is called a **permissible pair**.

Using all of this information, we can now evaluate prognostic models by giving a score of +1 for every permissible pair that is concordant, and a score of +0.5 for a permissible pair that is a risk tie.

### C-index (for Concordance index)
$$ C-index = \frac{\#concordant ~pairs + 0.5 \times \#risk  ~ties}{\#permissible ~pairs}$$

The C-index computes the following probability: 
$$ \mathbb P(score(A) > score(B) ~|~ y_A > y_B)$$
where $y_A$ and $y_B$ are outcomes for patients A and B.  
A random model would get 0.5 while a perfect model would get 1.  
In other words, measures the **discriminatory power** of a risk score.

## Tree-based models for prognosis
What makes **trees** incredibly useful in medical applications is their ability to handle both continuous and categorical data, their interpretability and the speed at which we can train them.  

Decision trees are prone to **overfitting** thus the **maximum depth** parameter should be chosen carefully.  
Another way to prevent overfitting is by using **random forests**. Random forests also generally boost the performance over single trees.  
Finally, popular models achieving the highest performance are **gradient boosting** based models.

## About missing data 
The missing data is common and an important issue to address while working with healthcare data.  
There are several reasons why data can be missing:

- **Missing completely at random**: when the missingness does not depend on anything i.e. $\mathbb P(missing) = constant$.  
In this case there is no reason that a bias is induced when removing those missing data.
- **Missing at random**: when the missingness depends only on available information (other features), $\mathbb P(missing) \neq constant$.
- **Missing not at random**: when the missingness depends on unrecorded information, $\mathbb P(missing) \neq constant$.

Missing data can be removed but this usually causes bias in models (except for missing data completely at random) since data distributions are modified. More specifically, the generalization performance of a model can be poor.  
We usually prefer to impute missing data e.g. by mean value or by regression.

## Survival models
**Survival models** are special models where we care about the **time to the occurrence of an event**, such as the time from treatment to recurrence or the time from diagnosis to death, and not **if** an event will happen (binary setup, as previously). In other words, we ask the question of what is the probability of **survival past any time $t$**. This quantity is given by the **survival function** $S$:
$$ S(t) = \mathbb P(X > t) $$

### Censored data effect on survival models
**(Right) Censoring** designates when we're trying to see the time to a particular event happening but it is only known to exceed a certain value, which is called **censoring time**. For example: 
- We start a study about predicting the time before a stroke happens
- The study ends after 6 months
- If a patient didn't have a stroke within those 6 months, then we can only say that the time to event is "6+ months".

Censoring can happen when a patient withdraws from study before its end (**loss-to-follow-up censoring**) or when the study ends (**end-of-study censoring**) for instance. Censored data cause problems for survival function estimation.  

Given a sample $\{t_i: i = 1, ..., N\}$, we can estimate $S(t), \forall t$ by $\frac{\#\{i: ~t_i>t]}{N}$. But if the sample contains right censored examples, what can we do? We can either assume that:
- the event happened right away: 18+ $\rightarrow$ 18
- the event never happened: 18+ $\rightarrow \infty$

After that, we can obtain 2 estimates of $S(t)$ i.e. and interval of time and we can only say that its value is somewhere between the 2 estimates. So we don't have an accurate estimation of the survival time.

### Kaplan-Meier estimator of the survival function
To estimate the survival function by taking into account censored data, one can use the Kaplan Meier non-parametric estimator :

$$ \hat S(t) = \prod_{t_i \leq t} (1 - \frac{d_i}{n_i}) $$
- $t_i$ are the time events observed in the dataset 
- $d_i$ is the number of deaths at time $t_i$
- $n_i$ is the number of people who we know have survived up to time $t_i$ i.e. with survival/censor time greater or equal to $t_i$.

## Hazard
The **hazard** of $t$ is the probability that the time to an event is at $t$, given it is at or after $t$:
$$ \lambda(t) = \mathbb P(T = t| T\geq t)$$
One interpretation is: what is the immediate **risk** of death (or any studied event) for a patient if they make it to time $t$?

Remark that: 
$$ S(t) = exp(-\int_0^t \lambda(u) du) $$
$$ \lambda(t) = -\frac{S'(t)}{S(t)} $$

In other words, the hazard is the **rate of death** if aged $t$.

### Cumulative hazard
$$ \Lambda(t) = \sum_{i=0}^t \lambda(i), ~\forall t \in \mathbb N$$
$$ \Lambda(t) = \int_{u=0}^t \lambda(u) du, ~\forall t \in \mathbb R_+ $$

### Cox-proportional hazard
The Cox-proportional hazard models the hazard in function of the variables available in the data:
$$ \lambda(t, \mathbf x) = \lambda_0(t)e^{\theta^T \mathbf x}$$
- $\lambda_0$: **baseline** hazard
- $\theta$: parameters/weights of the models
- $\mathbf x$: variables associated to a patient

It allows to "customize" the hazard in function of the characteristics of each patient.  

Note that for a variable $x_i$, $exp(\theta_i)$ is the **risk factor** increase for **unit increase** in $x_i$$:
- if $exp(\theta_i) > 1$, $x_i$ is increasing the risk
- if $exp(\theta_i) < 1$, $x_i$ is decreasing the risk

### Survival trees and Nelson-Aalen estimator
With Cox-proportional hazard, we model the hazard of a patient proportionally to its information (variables). Sometimes, hazard can be very different from a patient to another and the Cox model is no longer sufficient.  

**Survival trees** offer another method for hazard modelling.  
With this technique, we use a decision tree with the variables and compute a different hazard for each leaf of the tree. In other words, it associate a hazard to a group of patients (a population).  

Survival trees are usually used in association with the **Nelson-Aalen estimator** of the hazard function: 
$$ \hat H(t) = \sum_{i=0}^t  \frac{d_i}{n_i}$$


## Survival models evaluation
We can evaluate survival models with the C-index but some definitions need an update. Recall that we now consider time to event instead of a binary outcome and we might have censored data.  

- If *neither one* of the pair has *censored* data, the pair is **permissible** (even if **outcomes are similar**!) and definitions of (non-)concordant pair and risk tie are unchanged.
- If *at least one** patient is *not censored* (say patient A):
	- If patient B censoring time is *greater or equal* (e.g. 40+) to patient A time to event (e.g. 20), and $R_A > R_B$, then the pair is **permissible**.
	- If patient B censoring time is *lower* (20+) than patient A time to event (40), and $R_A > R_B$, then the pair is **non-permissible** (since we don't when (and if) the event will happen for patient A).
- If *both* patients are *censored*, the pair is **non-permissible**.
