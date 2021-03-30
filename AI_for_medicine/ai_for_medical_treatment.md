# AI for medical treatment
Based on [Deeplearning.ai course](https://www.coursera.org/learn/ai-for-medical-treatment/home/welcome).

## Randomized controlled trials
**Randomized controlled trials** (RCT) can help understand the **causal effect** of a **treatment** on a population.  
Let's say a doctor is trying out a new treatment and wants to see how effective the treatment is going to be for reducing heart attack and wants to know how effective the treatment is. One idea is to try out the treatment on a group of patients, the **treatment group** and not use the treatment on another group of patients, the **control group**, to whom a **placebo** or a standard of care is given.  
Patients must be **randomly** assigned to these groups to avoid **bias**. 

### Absolute risk reduction
Let's say after one year 2% get a heart attack in the treatment group, while 5% of the patients in the control group get a heart attack. The treatment effect is expressed with the **absolute risk** (AR), here 0.02 AR vs 0.05 AR. In particular, we say that the **absolute risk reduction** (ARR) of the treatment is 0.03.

### Result significance and interpretation
To evaluate the result significance, we compute **p-values**. Let's say we get a p-value smaller than 0.001. That says that there is less than a 0.1% probability that we would observe a difference in risk of 0.03 or more if the treatment actually has zero effect. Because the p-value is small, we thus call this result **statistically significant**.  

A way to interpret the ARR is with the **number needed to treat** (NNT) which is the number of people who need to receive a treatment in order to benefit one of them: 
$$ NNT = \frac{1}{ARR} $$

## Causal inference and average treatment effect
### Causal inference
Causal inference is related to the relationship between having a treatment of not (the cause) and the possible outcomes (the consequence).

The **Neyman-Rubin causal model** reports the results of an RCT as follows:
| i | $y_i(1)$ | $y_i(0)$  | $y_i(1) - y_i(0)$ |
|--|:-----:|:-----:|:---------:|
| 1 | 0 | 1 | -1 |
| 2 | 1 | 1 | 0 |
| 3 | 1 | 0 | 1 |
| 4 | 0 | 0 | 0 |
| 5 | 0 | 1 | -1

Here:
- 0: no heart attack  
- 1: heart attack

And:
- $y_i(1)$: outcome for a patient under treatment
- $y_i(0)$: outcome for a patient under control.  
- $y_i(1) - y_i(0)$: represents the treatment effect for each patient
	- -1: benefit
	- 0: no effect
	- 1: harmful

With this, one can compute the mean value for each group:
| i | $y_i(1)$ | $y_i(0)$  | $y_i(1) - y_i(0)$ |
|--|:-----:|:-----:|:---------:|
| Mean | 0.4 | 0.6 | -0.2 | 

-0.2 is an estimation of the  **average treatment effect** (ATE).  

### Average treatment effect estimation
But in reality, the challenge is that we don't get to observe what would happen to a patient both with and without treatment. We can only observe **either** $y_i(1)$ **or** $y_i(0)$. Therefore how can we estimate the ATE?  

RCT data allows use to do so:
$$ ATE = \mathbb E[Y_i(1) - Y_i(0)] = \mathbb E[Y_i | W = 1] - \mathbb E[Y_i | W = 0] $$
where:
- $W$ represents the group of the patient (0 for control, 1 for treatment)
- $Y_i$ represents the outcome for that patient

The ATE is then similar to the previously seen ARR: 
$$ ATE = -ARR $$.

### Conditional average & individual treatment effect
One might want to know the effect of the treatment on a sub-population for example. To do so, conditioning the ATE is possible:
$$ \mathbb E[Y_i(1) - Y_i(0) | X = x] = \mathbb E[Y_i | W = 1, ~X = x] - \mathbb E[Y_i | W = 0, ~X = x] $$
where $X$ represents features for a patient (like age, blood pressure, etc.).  

Notice that the more conditions we give, the less patients we have. Therefore thanks to that we can measure the **individual treatment effect** (ITE).

#### T-learner
The idea is to learn the relationship between the outcome and the features to estimate the ITE. One can rewrite the previous equation:
$$ \mathbb E[Y_i(1) - Y_i(0) | X = x] = \mu_1(x) - \mu_0(x) $$
Then we would estimate $\mu_1(x)$ and $\mu_0(x)$ with the data: $\hat \mu_1(x), ~\hat \mu_0(x)$.  
Note that $\mu_1$ and $\mu_0$ are called resp. **treatment response function** and **control response function**. These functions can be estimated by base learners (i.e. simple models like decision trees or linear model). In other words we use **prognostic models**.  
This method is called the **Two-tree method** or **T-learner**.

#### S-learner
One can also rewrite the previous ITE equation as:
$$ \mathbb E[Y_i(1) - Y_i(0) | X = x] = \mu(x, 1) - \mu(x, 0) $$ 
where $\mu(x, w) = \mathbb E[Y_i | W = w, ~X = x]$

We now use a single model to estimate the ITE. In this case, we might consider the treatment variable $W$ as a feature while in the Two-tree method it was used to split patients into groups. $W$ can also be left out and we just estimate $\mathbb E[Y_i | X = x]$ with a single model.  
This method is the **S-learner**. It can help capture more precisely the relationship between features than the T-learner since there is more data available.

### ITE estimator evaluation
Recall that for each patient we can only observe **either** $y_i(1)$ **or** $y_i(0)$. Despite that, we saw that there was a way to obtain an estimation of the ITE.  
However, the other outcome is needed to evaluate the estimator. Let's say we have a patient in the treatment group ($W = 1$) for which the observed outcome is $y(1) = 0$.
One way to evaluate the associated ITE estimate is to find another patient for which we know $y(0)$. We can either find a patient with **similar features** or **similar ITE estimate**. Once a duo is matched, we can compute the mean ITE estimate and the observed treatment effect ($y(1) - y(0)$) (recall that -1 means benefit, 0 no effect, 1 harm).  

In the similar way that we evaluated prognosis models with the C-index, we can evaluate our ITE estimates with the **C-for-benefit**. For this, we form a **pair of two duos** and we classify a pair as:
- concordant: if the higher ITE estimate corresponds to the higher outcome (here 1 so harmful outcome).
- non-concordant: ...
- risk tie: if the two ITE estimates are equal
- tie in outcome: if the outcomes are similar. We **can not** use this pair. As long as outcomes are different, the pair is permissible

Once we identified each pair, we can compute the C-for-benefit which is exactly the C-index. This then allows to compare the T-learner and the S-learner.

## Medical question answering 
Let's say a patient or a doctor wants to know more about a medical diagnosis or a treatment. One way they might learn is to ask a question in natural language and get an answer to that question. This is the job of **question answering** (QA) systems.
For many questions that are entered into search engines, search engines are often able to find the passage of text containing the answer. The challenge is the last step of answer extraction, which is to find the shortest segment of the passage that answers a question.  

A model will thus take in a user question, and a passage that contains the answer to the question. This might be a passage that might be returned by Google search. And the model will produce an answer, that is extracted from this passage.  
There have been many recent advances on the question answering task in NLP, including recent models called ELMo, BERT, and XLNet. 

### The BERT model
The **BERT** model consists of several layers called **transformer blocks**.  
First we can break up the question and the passage into **tokens** or **words**. We separate the inputs from the question and from the passage using a special token called the **separator token**. In reality, BERT further separates words into word pieces and also has a start token at the start. Now these inputs pass into the model, where they pass through several transformer blocks and are ultimately transformed into a list of vectors. There's one 768-dimensional vector for each of the words. This is called the **word representation** for a word.  

Word representations represent words in a way that capture meaning related relationships between words. Distances between words capture how related they are or how often they're used in similar context.  
Note that it is possible to visualize these word dimensions by reducing the  dimensions of the vectors to two dimensions using methods such as **t-SNE**.  

BERT outputs a segment of the passage as answer. This answer can be represented  using its start and end words. To go back to the word representations where it learns, the task for the model is to be able to determine whether each word in the passage is one of the start or the end of an answer to a question.  
The model learns two vectors, $S$ and $E$, for each of the word representations, for each of the words in the passage. The word representation is multiplied by $S$ to get a single number, which is the **start score** for that word (the higher the start score, the more likely it is to be the start of the answer). And similarly with $E$ for the **end score**.  
One can then find out what the most likely answer is by computing a **grid of words**. In this grid, each cell is equal to $start~score + end~score$ (start score -> rows, end scores -> columns). The model thus outputs the start and end word corresponding with the highest score here. The model learns the vectors S and E and updates its word representations based on being shown many of these question, passage, and answer triplets.  

Typically, the model is first shown natural question and answers in English in the general domain using datasets like SQuAD and then fine tune on medical datasets like BioASQ.
There have been extensions to the BERT model for medicine like BioBERT, that uses passages from medical papers to learn these word representations. The advantage of this is that the words that BioBERT is thus able to learn, are words used in the context of medicine. 

### Handling words with multiple meanings
One of the main challenges of word representations is how to deal with words that have multiple meanings.  

**Non-contextualized word representations** such as Word2Vec and Glove use a single word representation for a word.  
More recently, words are given a representation based on the context that is surrounding a word. Such **contextualized word representations** include ELMo and BERT.  

Let's see how BERT learns these contextualized word representations.  
Words from a passage of text are input into a BERT model. Then one of the tokens in the passage is masked with a special **MASK token**. The model is trained to predict what the mask was. An extra layer is added where the output is the probabilities of the missing word being every single word in the vocabulary. In the process of learning to correctly predict the masked word, the model learns word representations here in blue.  

## Automatic label extraction for medical imaging
NLP can be also be used to automatically create labels of medical images by extracting mentions of diseases from reports.  
To do so, a model must first find whether an observation (e.g. a disease) is mentioned in a report. For that, one can search for the explicit name of the disease in the report.  

### Synonyms for labels
However we usually look for **synonyms** or for **terminologies** which are an organizations and collections of medical terms to provide codes, definitions, and synonyms (see SNOMED CT).  

### Is-a relationships for labels
Let's say we search a report for "lung disease" or its synonyms using a terminology.
However, the report only mentions "infection" which is not a direct synonym of "lung disease" but only related.  Luckily, terminologies not only contain synonyms for our concept but also contain **relationships** to other concepts.  
The advantage of this approach, which we can call a **rules-based approach** of finding mentions of observations, is that we don't need any data for supervised learning. The disadvantage is that there is a lot of manual work to refine these rules  based on what is working and what is not working.

### Classify the observation
Once we have determined if a disease is mentioned in the report, we must classify whether the disease is present or absent. To classify as absent, we need to capture a negation expression related to the mention of the disease.  

A common method to do so is to use **regular expressions** or regex rules to find patterns on text strings.  
A more sophisticated set of methods use dependencies  between grammatical units called **dependency parse rules**.  
Another set of methods use supervised learning with the report as input, and the label of whether an observation is present or absent as output.

### Extracted labels evaluation
To evaluate labels extracted from a report, the **ground truth** is needed. The ground truth can be obtained using a group of experts to look at the report and then annotate the presence or absence of each of the diseases. Another option would be to use the experts to look at the image the report is describing.  

The advantage of using the report is that it would be more straightforward to improve the system based on looking at the errors made on the report.  
The advantage of using  the image is that it is a more direct evaluation of the quality of the label for the task (since the report is first made from an observation of the image).  

After that, we evaluate the extracted labels using metrics. **Precision** and **recall** are commonly used which combined give the **F1-score**.  
Note that precision is similar to PPV and recall to sensitivity.
