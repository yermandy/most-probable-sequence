
#### Problem formulation (inference)

$$ 
\begin{align}
\hat{y_1}, \dots, \hat{y_n}  &\in \mathop{\text{arg max}}\limits_{y_1,...,y_n} {\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})} \\
y_i &\in \lbrace0,1,\dots,Y\rbrace \\
\forall i &\in \lbrace1,\dots,n \rbrace
\end{align}
$$


Scores are represented by

$$
f_{i}(y_i,y_{i+1}) = \braket{w(y_i + y_{i+1}), \phi(x_{i,i+i})} + b(y_i + y_{i+1})
$$

where $w$ and $b$ are weights and biases of the last layer of the network

<!-- TODO: dimentsions of all variables -->

#### Margin rescaling loss

$$
\begin{align}
    \Delta (\theta, \bar x, \bar y) &= \max_{y_1, ... ,y_n} \Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}}| + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    &= \max_{c \in \lbrace0, ... , n \cdot Y\rbrace} \quad  \max_{y_1 + ... + y_n = c} \Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}}| + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    &= \max_{c \in \lbrace0, ... , n \cdot Y\rbrace} \Big[ \frac{1}{\bar{c}} |c - \bar{c}| + \max_{y_1 + ... + y_n = c}\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    &= \max_{c \in \lbrace0, ... , n \cdot Y\rbrace} \Big[ \frac{1}{\bar{c}} |c - \bar{c}| + F(c)  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1})
\end{align}
$$

$$\bar c = \sum_{i=1}^{n} \bar y_i$$

where $\bar y_i$ denote the ground truth labeling for $i$-th window.

#### Solving $F(c)$ efficiently with Dynamic Programming

Let $F_k(c, y_k)$ has the following form:

$$
\begin{align}
    F_k(c, y_k) &= \max_{y_1+...+y_n=c} \Big[ \sum_{i=1}^{k-2} f_{i}(y_i,y_{i+1}) + f_{k-1}(y_{k-1},y_{k}) \Big] \\
    &= \max_{y_{k-1} \in \lbrace l(c, k),...,u(c)\rbrace } \Big[ F_{k-1}(c-y_{k-1}, y_{k-1}) + f_{k-1}(y_{k-1},y_{k}) \Big] \\
    \forall c &\in \lbrace 0,\dots, (k - 1) \cdot (Y - 1) + 1 \rbrace \\
    \forall y_k &\in \lbrace 0,\dots, Y\rbrace
\end{align}
$$

where 

$$
\begin{align}
l(c, k) &= \max(0, c - (Y-1)\cdot(k - 2)) \\
u(c) &= \min(c, Y)
\end{align}
$$

Then we can define $F(c)$ as

$$
\begin{align}
F(c) &= \max_{y_1 + ... + y_n = c}\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) \\
     &= \max_{y_n \in \lbrace l(c, n),...,u(c)\rbrace} F_n(c - y_n, y_n)
\end{align}
$$

#### Example

1st iteration:

$$
F_2(c, y_2) = f_1(c, y_2) \quad \forall c \in \lbrace 0,\dots, Y\rbrace \quad \forall y_2 \in \lbrace 0,\dots, Y\rbrace
$$

2nd iteration:

$$
\begin{align}
F_3(c, y_3) &= \max_{y_1 + y_2 = c} \Big[ f_1(y_1, y_2) + f_2(y_2, y_3) \Big] \\
&= \max_{y_2 \in \lbrace l(c, 3),...,u(c)\rbrace} \Big[ F_2(c - y_2, y_2) + f_2(y_2, y_3) \Big] \\
&\forall c \in \lbrace 0,\dots, 2 \cdot Y\rbrace \quad \forall y_3 \in \lbrace 0,\dots, Y\rbrace
\end{align}
$$

#### Learning

Calculate the sub-gradients of the margin rescaling loss $\Delta (\theta, \bar x, \bar y)$

$$
    \tilde y_1,\dots, \tilde y_n = \mathop{\text{arg max}}\limits_{y_1,...,y_n} {\Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}}| + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})  \Big]}
$$

$$
\begin{align}
g_w(z) &= \sum_{i=1}^{n-1} {\phi(x_{i,i+1}) (⟦ \tilde y_i + \tilde y_{i+1} = z ⟧ - ⟦ \bar y_i + \bar y_{i+1} = z ⟧)} \\
g_b(z) &= \sum_{i=1}^{n-1} {(⟦ \tilde y_i + \tilde y_{i+1} = z ⟧ - ⟦ \bar y_i + \bar y_{i+1} = z ⟧)}
\end{align}
$$

where $\phi(x)$ are features and $\bar y_i$ are the ground truth labels for $i$-th window.

#### Normalization

When the training set consists of samples of different lengths, we can to normalize the sums of scores and the sub-gradients of the margin rescaling loss. For example: $\frac{1}{n} \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})$ and $\frac{1}{n} \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1})$ as well as $\frac{g_w(z)}{n}$ and $\frac{g_b(z)}{n}$