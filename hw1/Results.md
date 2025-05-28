## Behavior Cloning Policy Implementation and Training Report

[!bc_ant](/hw1/data/media/bc_ant.gif)
[!bc_walker](/hw1/data/media/bc_walker.gif)
[!bc_hopper](/hw1/data/media/bc_hopper.gif)
[!bc_cheetah](/hw1/data/media/bc_half_cheetah.gif)

### Setup
Please refer to the [README.md](README.md) file for detailed setup instructions, environment requirements, and installation steps. 

### Scripts

- **`cs285/scripts/run_tasks.py`**  
  Automates running behavioral cloning and DAgger experiments for multiple environments.  
  **Usage:**  
  ```bash
  python cs285/scripts/run_tasks.py --mode <mode> [--do_dagger]
  ```
  - `--mode run`: Launches experiments for all environments.
  - `--mode report`: Prints summary tables and plots for experiments in `data/q1` and `data/q2`.
  - `--do_dagger`: If set, runs DAgger experiments (for Q2, with iterative expert relabeling).
  - `--sort_by name|percent`: Sorts report tables by name or Eval/Train %.

- **`cs285/scripts/tune_hyperparameter.py`**  
  Provides utilities for hyperparameter tuning and visualization for behavioral cloning.  
  **Usage:**  
  ```bash
  python cs285/scripts/tune_hyperparameter.py --mode <mode>
  ```
  - `--mode tune`: Runs a grid search over hyperparameters for behavioral cloning on Walker2d-v4.
  - `--mode visualize`: Visualizes the results of the hyperparameter sweep using parallel coordinates plots (requires logs in `data/bc_hyperparameter_tuning/`).

### Model Architecture

The policy $\pi_\theta(a \mid s)$ is parameterized as a Multi-Layer Perceptron (MLP) neural network that maps observations $s \in \mathbb{R}^{d_{\text{obs}}}$ to actions $a \in \mathbb{R}^{d_{\text{act}}}$. The MLP is defined as follows:

- **Input layer:** Dimension $d_{\text{obs}}$ corresponding to the observation space
- **Hidden layers:** $n_{\text{layers}}$ fully connected layers, each with $\text{size}$ units and $\tanh$ activation function
- **Output layer:** Linear layer producing output of dimension $d_{\text{act}}$

The network outputs the **mean** of a Gaussian distribution over actions: $\mu_\theta(s) = \text{MLP}_\theta(s)$

The **standard deviation** $\sigma$ is modeled as a separate neural network, `self.logstd_net`, which maps the observation $s$ to a vector in $\mathbb{R}^{d_{\text{act}}}$: $\text{logstd}_\theta(s) = \text{logstd\_net}(s), \quad \sigma = \exp(\text{logstd}_\theta(s))$

The resulting action distribution is modeled as: $\pi_\theta(a \mid s) = \mathcal{N}(a; \mu_\theta(s), \operatorname{diag}(\sigma^2))$

### Forward Pass

The `forward` method of the policy constructs a Gaussian distribution over actions:
```python
def forward(self, observation: torch.FloatTensor) -> Any:
    mean = self.mean_net(observation)  # (batch_size, ac_dim)
    logstd = self.logstd_net(observation)  # (batch_size, ac_dim)
    std = torch.exp(logstd)  # (batch_size, ac_dim)
    action_dist = distributions.Normal(mean, std)
    return action_dist
```

This method returns a `torch.distributions.Normal` object, which allows sampling and computation of log-probabilities in a differentiable way.

### Loss Function

The policy is trained via **Behavior Cloning** using supervised learning. The loss function is the **negative log-likelihood** of the expert actions under the predicted action distribution:

$$
\mathcal{L}_{\text{BC}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \pi_\theta(a_i^{\text{expert}} \mid s_i)
$$

Here, $\log \pi_\theta(a \mid s)$ refers to the logarithm of the probability density of action $a$ under the policy's Gaussian distribution at state $s$. Intuitively, this measures how likely the expert's action is according to the current policy. The more confident and accurate the policy is in predicting the expert's action, the higher this probability (and the lower the negative log-likelihood).

This log-likelihood is computed using the `log_prob` method of the `torch.distributions.Normal` class.

### Update Method

The `update` method performs a single gradient step using the computed loss:
```python
def update(self, obs, acs, **kwargs):
    self.optimizer.zero_grad()
    obs = ptu.from_numpy(obs)
    acs = ptu.from_numpy(acs)
    action_distribution = self.forward(obs)

    log_prob = action_distribution.log_prob(acs)  # (batch_size, ac_dim)
    loss = -log_prob.mean()  # Scalar loss

    loss.backward()          # Backpropagation
    self.optimizer.step()    # Gradient descent step

    return {
        'Training Loss': ptu.to_numpy(loss),
    }
```

- `obs` and `acs` are batched observations and expert actions, respectively.
- The loss is computed as the **mean** of the negative log-probabilities.
- The optimizer used is **Adam**, which updates both the mean network $\theta$ and the log standard deviation network $\text{logstd\_net}$.

### Training Notes

- The Gaussian distribution is modeled independently per action dimension.
- The standard deviation is no longer a global parameter but a learned function of the input state, increasing expressiveness.
- Log standard deviation is computed through a dedicated neural network and is broadcasted to match batch dimensions.
- Using `torch.distributions.Normal` allows efficient and differentiable computation of log-likelihoods.
- Proper normalization of the loss ensures stability during optimization.

This model and training setup closely follows an advanced behavior cloning approach, enabling the policy to not only imitate expert behavior but also adapt its uncertainty based on the input observation.

## Results

### Behavior Cloning

| Log Directory                |   Loss/train |   Eval_AverageReturn |   Eval_StdReturn |   Train_AverageReturn |   Train_StdReturn | Eval/Train %   |
|------------------------------|--------------|----------------------|------------------|-----------------------|-------------------|----------------|
| q1_bc_Ant-v4                 |    -1.97703  |             4596.29  |          118.43  |               4681.89 |         30.7086   | 98.2%          |
| q1_bc_HalfCheetah-v4         |    -1.342    |             3900.06  |          108.31  |               4034.8  |         32.8678   | 96.7%          |
| q1_bc_Hopper-v4_1000_steps   |    -1.1009   |              849.411 |          261.381 |               3717.51 |          0.353036 | 22.8%          |
| q1_bc_Hopper-v4_5000_steps   |    -1.8346   |             1335.65  |          353.37  |               3717.51 |          0.353036 | 35.9%          |
| q1_bc_Walker2d-v4_1000_steps |    -0.666707 |             1790.89  |         1278.83  |               5383.31 |         54.1525   | 33.3%          |
| q1_bc_Walker2d-v4_5000_steps |    -1.32398  |             5157.58  |          399.826 |               5383.31 |         54.1525   | 95.8%          |

#### Hyperparameter Tuning

##### Parallel Co-ordinate plot

![Parallel Coordinates Plot Percent](/hw1/data/media/parallel_co-ordinates_train_percent.png)
![Parallel Coordinates Plot Loss](/hw1/data/media/parallel_co-ordinates_train_loss.png)

See [Appendix - Hyperparameter Tuning](#hyperparameter-tuning-1) for complete results.

### Dagger

| Log Directory            |   Loss/train |   Eval_AverageReturn |   Eval_StdReturn |   Train_AverageReturn |   Train_StdReturn | Eval/Train %   |
|--------------------------|--------------|----------------------|------------------|-----------------------|-------------------|----------------|
| q2_dagger_HalfCheetah-v4 |     -2.00132 |              4200.17 |                0 |               3986.53 |                 0 | 105.4%         |
| q2_dagger_Walker2d-v4    |     -1.1284  |              5369.59 |                0 |               5304.03 |                 0 | 101.2%         |
| q2_dagger_Hopper-v4      |     -1.30433 |              3721.51 |                0 |               3738.22 |                 0 | 99.6%          |
| q2_dagger_Ant-v4         |     -2.70942 |              4704.94 |                0 |               4737.53 |                 0 | 99.3%          |

![Dagger Iteration vs Mean Return](/hw1/data/media/Dagger_iter_vs_error.png)

## Appendix

### Hyperparameter Tuning 

| Log Directory                                                                                                       |   Loss/train |   Eval_AverageReturn |   Eval_StdReturn |   Train_AverageReturn |   Train_StdReturn | Eval/Train %   |
|---------------------------------------------------------------------------------------------------------------------|--------------|----------------------|------------------|-----------------------|-------------------|----------------|
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-07    |   -1.32398   |          5441.69     |          0       |               5383.31 |           54.1525 | 101.1%         |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-53-42 |   -2.00181   |          5415.8      |          0       |               5383.31 |           54.1525 | 100.6%         |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-46  |   -1.27743   |          5374.71     |          0       |               5383.31 |           54.1525 | 99.8%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-54-05 |   -1.69202   |          5257.71     |          0       |               5383.31 |           54.1525 | 97.7%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-03   |   -0.948775  |          5234.79     |          0       |               5383.31 |           54.1525 | 97.2%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-53-30 |   -1.64699   |          5200.21     |          0       |               5383.31 |           54.1525 | 96.6%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-31  |   -1.04709   |          5152.02     |          0       |               5383.31 |           54.1525 | 95.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-08   |   -1.4541    |          5129.87     |          0       |               5383.31 |           54.1525 | 95.3%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-08    |   -1.11122   |          5110.55     |          0       |               5383.31 |           54.1525 | 94.9%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-53-08 |   -1.69194   |          5096.2      |          0       |               5383.31 |           54.1525 | 94.7%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-57  |   -1.17253   |          5095.51     |          0       |               5383.31 |           54.1525 | 94.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-48   |   -1.05189   |          4967.61     |          0       |               5383.31 |           54.1525 | 92.3%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-54-17 |   -1.95417   |          4765.44     |        418.681   |               5383.31 |           54.1525 | 88.5%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-51    |   -1.19785   |          4735.24     |          0       |               5383.31 |           54.1525 | 88.0%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-00   |   -1.03898   |          4641.18     |          0       |               5383.31 |           54.1525 | 86.2%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-33   |   -1.64429   |          4572.74     |          0       |               5383.31 |           54.1525 | 84.9%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-57 |   -1.44137   |          4085.41     |        315.323   |               5383.31 |           54.1525 | 75.9%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-40   |   -1.48319   |          3454.26     |       1509.6     |               5383.31 |           54.1525 | 64.2%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-40   |   -1.44228   |          3070.4      |       1737.07    |               5383.31 |           54.1525 | 57.0%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-56    |   -1.11145   |          3014.38     |       1519.92    |               5383.31 |           54.1525 | 56.0%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-43   |   -0.918558  |          2894.82     |       1948.41    |               5383.31 |           54.1525 | 53.8%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-53-19  |   -1.37994   |          2783.6      |       2538.99    |               5383.31 |           54.1525 | 51.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-47    |   -1.11992   |          2722.26     |        763.439   |               5383.31 |           54.1525 | 50.6%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-51  |   -1.57639   |          2170.51     |       1537.4     |               5383.31 |           54.1525 | 40.3%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-52   |   -0.836875  |          2069.21     |       1790.1     |               5383.31 |           54.1525 | 38.4%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-26   |   -1.24148   |          1916.94     |       1726.12    |               5383.31 |           54.1525 | 35.6%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-10  |   -1.03371   |          1902.74     |       2252.59    |               5383.31 |           54.1525 | 35.3%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-21    |   -1.15157   |          1868.52     |       1219.14    |               5383.31 |           54.1525 | 34.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-57   |   -1.0537    |          1778.12     |       1271.27    |               5383.31 |           54.1525 | 33.0%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-46   |   -1.10515   |          1648.99     |        727.161   |               5383.31 |           54.1525 | 30.6%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-02  |   -0.460923  |          1582.12     |       1567.51    |               5383.31 |           54.1525 | 29.4%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-35    |   -1.21984   |          1459.11     |       1233.42    |               5383.31 |           54.1525 | 27.1%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-15    |   -0.666707  |          1267.5      |        809.191   |               5383.31 |           54.1525 | 23.5%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-29  |   -1.37803   |          1234.55     |        962.303   |               5383.31 |           54.1525 | 22.9%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-18   |   -0.838957  |          1233.93     |       1694.13    |               5383.31 |           54.1525 | 22.9%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-24   |   -0.982582  |          1218.79     |       1265.99    |               5383.31 |           54.1525 | 22.6%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-24    |   -0.773681  |          1208.99     |       1261.95    |               5383.31 |           54.1525 | 22.5%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-20   |   -1.22038   |          1164.4      |        644.827   |               5383.31 |           54.1525 | 21.6%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-50  |   -0.962811  |          1123.35     |       1784.12    |               5383.31 |           54.1525 | 20.9%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-16   |   -0.819639  |          1092.53     |        834.96    |               5383.31 |           54.1525 | 20.3%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-11   |   -0.424604  |          1071.41     |       1275.6     |               5383.31 |           54.1525 | 19.9%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-40   |   -1.36351   |          1060.63     |        637.959   |               5383.31 |           54.1525 | 19.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-34    |   -0.73292   |          1053.09     |       1711.41    |               5383.31 |           54.1525 | 19.6%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-12   |   -1.09394   |          1051.67     |          0       |               5383.31 |           54.1525 | 19.5%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-18  |   -1.57146   |          1027.91     |       1758.55    |               5383.31 |           54.1525 | 19.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-45    |    1.11995   |          1010.72     |          0       |               5383.31 |           54.1525 | 18.8%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-27  |   -1.59341   |           933.049    |        595.558   |               5383.31 |           54.1525 | 17.3%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-39  |   -1.03112   |           868.067    |        557.588   |               5383.31 |           54.1525 | 16.1%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-17  |   -1.20503   |           864.887    |       1521.9     |               5383.31 |           54.1525 | 16.1%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-48-01   |   -1.11658   |           815.766    |        274.223   |               5383.31 |           54.1525 | 15.2%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-43    |   -1.06421   |           793.463    |        927.186   |               5383.31 |           54.1525 | 14.7%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-54   |   -1.32314   |           785.32     |        426.469   |               5383.31 |           54.1525 | 14.6%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps10000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-53-53  |   -1.33224   |           773.292    |       1532.45    |               5383.31 |           54.1525 | 14.4%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-27   |   -0.761296  |           718.066    |        558.628   |               5383.31 |           54.1525 | 13.3%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-10   |    0.438968  |           711.657    |        374.521   |               5383.31 |           54.1525 | 13.2%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-44   |   -0.0301155 |           620.859    |        390.308   |               5383.31 |           54.1525 | 11.5%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-22  |   -0.516914  |           613.908    |        430.731   |               5383.31 |           54.1525 | 11.4%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-26    |   -1.34659   |           585.239    |        757.769   |               5383.31 |           54.1525 | 10.9%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-31   |   -0.781114  |           576.175    |        485.668   |               5383.31 |           54.1525 | 10.7%          |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-38  |   -0.505942  |           550.604    |        508.906   |               5383.31 |           54.1525 | 10.2%          |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps5000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-13   |   -1.13874   |           524.948    |        704.696   |               5383.31 |           54.1525 | 9.8%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps5000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-52-38  |   -1.25549   |           511.05     |        412.761   |               5383.31 |           54.1525 | 9.5%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-47-02   |   -1.00319   |           497.837    |        294.487   |               5383.31 |           54.1525 | 9.2%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-17  |   -0.48565   |           495.856    |        298.323   |               5383.31 |           54.1525 | 9.2%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-33  |   -0.451811  |           472.645    |        459.046   |               5383.31 |           54.1525 | 8.8%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-51   |    0.773315  |           466.194    |        339.192   |               5383.31 |           54.1525 | 8.7%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-04  |   -1.2894    |           463.133    |        560.863   |               5383.31 |           54.1525 | 8.6%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-21   |   -0.811415  |           461.832    |        148.172   |               5383.31 |           54.1525 | 8.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-19     |    0.582922  |           435.104    |        281.542   |               5383.31 |           54.1525 | 8.1%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-36  |   -0.0457676 |           433.132    |        161.111   |               5383.31 |           54.1525 | 8.0%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-53  |   -0.0600901 |           419.348    |        349.184   |               5383.31 |           54.1525 | 7.8%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps3000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-31   |   -1.17856   |           400.541    |        174.686   |               5383.31 |           54.1525 | 7.4%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-40  |   -0.0626403 |           390.53     |        226.14    |               5383.31 |           54.1525 | 7.3%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-27   |   -0.405764  |           365.525    |        302.012   |               5383.31 |           54.1525 | 6.8%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-17  |    0.434689  |           353.095    |        148.778   |               5383.31 |           54.1525 | 6.6%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-07  |    0.432083  |           344.395    |         73.319   |               5383.31 |           54.1525 | 6.4%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-51-07  |   -0.51602   |           339.729    |        175.853   |               5383.31 |           54.1525 | 6.3%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-20   |   -0.0276046 |           318.669    |        209.775   |               5383.31 |           54.1525 | 5.9%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-40   |    0.595739  |           316.496    |        196.587   |               5383.31 |           54.1525 | 5.9%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-04  |    0.432304  |           309.876    |        113.627   |               5383.31 |           54.1525 | 5.8%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps3000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-57   |   -0.429708  |           309.687    |         80.2181  |               5383.31 |           54.1525 | 5.8%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-32   |   -0.02518   |           302.009    |        146.294   |               5383.31 |           54.1525 | 5.6%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-24  |   -0.0427598 |           298.715    |        125.124   |               5383.31 |           54.1525 | 5.5%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-49  |   -0.0425016 |           281.075    |        189.491   |               5383.31 |           54.1525 | 5.2%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-14  |    0.436947  |           277.618    |        114.018   |               5383.31 |           54.1525 | 5.2%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-48    |    0.577651  |           277.58     |        159.461   |               5383.31 |           54.1525 | 5.2%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-55  |    0.438521  |           276.889    |        118.736   |               5383.31 |           54.1525 | 5.1%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-01   |    0.448919  |           276.563    |        144.22    |               5383.31 |           54.1525 | 5.1%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-15    |    0.863001  |           276.293    |        117.176   |               5383.31 |           54.1525 | 5.1%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-52   |    0.458727  |           267.418    |         97.6733  |               5383.31 |           54.1525 | 5.0%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps1000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-58  |    0.425678  |           264.688    |        138.378   |               5383.31 |           54.1525 | 4.9%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-07     |    0.283509  |           243.554    |        102.528   |               5383.31 |           54.1525 | 4.5%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-27     |    0.28311   |           234.171    |        357.59    |               5383.31 |           54.1525 | 4.3%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-40   |   -0.575192  |           216.866    |         62.9287  |               5383.31 |           54.1525 | 4.0%           |
| q1_bc_Walker2d-v4_lr0.0005_n_iter1_steps2000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-50-28  |   -0.0563477 |           213.521    |        152.699   |               5383.31 |           54.1525 | 4.0%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-39     |    0.291978  |           197.591    |        102.938   |               5383.31 |           54.1525 | 3.7%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps2000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-46-04   |   -0.794192  |           187.418    |          6.90024 |               5383.31 |           54.1525 | 3.5%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps1000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-37   |   -0.716663  |           137.847    |        175.068   |               5383.31 |           54.1525 | 2.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-52    |    0.841416  |           134.957    |         98.0187  |               5383.31 |           54.1525 | 2.5%           |
| q1_bc_Walker2d-v4_lr0.005_n_iter1_steps10000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-49-15   |   -1.34854   |            99.8203   |        632.456   |               5383.31 |           54.1525 | 1.9%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-47     |    0.757218  |            85.0706   |         66.951   |               5383.31 |           54.1525 | 1.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-54    |    0.816558  |            78.8558   |        161.311   |               5383.31 |           54.1525 | 1.5%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-23    |    0.999014  |            72.9576   |         68.6326  |               5383.31 |           54.1525 | 1.4%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-57     |    0.768597  |            72.8873   |         72.0302  |               5383.31 |           54.1525 | 1.4%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-57     |    0.659455  |            68.7368   |         65.9324  |               5383.31 |           54.1525 | 1.3%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers2_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-40    |    0.651058  |            64.6747   |         84.9061  |               5383.31 |           54.1525 | 1.2%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-43     |    0.36771   |            52.623    |         54.7735  |               5383.31 |           54.1525 | 1.0%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-44-14   |    0.859265  |            36.303    |         57.2624  |               5383.31 |           54.1525 | 0.7%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-48     |    0.601192  |            31.4105   |         62.8054  |               5383.31 |           54.1525 | 0.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-51    |    0.78435   |            31.1317   |         63.8727  |               5383.31 |           54.1525 | 0.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers3_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-44-02    |    0.815933  |            30.4309   |         44.6337  |               5383.31 |           54.1525 | 0.6%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-33    |    0.483962  |            28.6047   |         58.1247  |               5383.31 |           54.1525 | 0.5%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-44-36    |    1.21038   |            26.8219   |         35.9781  |               5383.31 |           54.1525 | 0.5%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers2_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-29    |    0.365761  |            22.6629   |         62.6508  |               5383.31 |           54.1525 | 0.4%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-12     |    1.18925   |            20.9896   |         22.6883  |               5383.31 |           54.1525 | 0.4%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-42    |    0.584316  |            19.5151   |         34.1502  |               5383.31 |           54.1525 | 0.4%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-01    |    0.998648  |            18.6273   |         54.9558  |               5383.31 |           54.1525 | 0.3%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-40-54    |    0.945806  |            15.8543   |         54.9691  |               5383.31 |           54.1525 | 0.3%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers3_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-02    |    0.903397  |            11.546    |         13.4976  |               5383.31 |           54.1525 | 0.2%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers2_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-11    |    0.524373  |             7.55977  |         31.5891  |               5383.31 |           54.1525 | 0.1%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-27    |    0.989362  |             6.83101  |         10.8955  |               5383.31 |           54.1525 | 0.1%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps1000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-04    |    1.09938   |             2.2455   |          3.28461 |               5383.31 |           54.1525 | 0.0%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-31     |    1.01341   |             2.01618  |          7.31404 |               5383.31 |           54.1525 | 0.0%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-07    |    1.10601   |             1.45402  |          9.69546 |               5383.31 |           54.1525 | 0.0%           |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-17    |    1.04297   |            -0.908332 |          5.30663 |               5383.31 |           54.1525 | -0.0%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-35    |    1.04482   |            -1.38765  |          4.25917 |               5383.31 |           54.1525 | -0.0%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers4_size64_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-08     |    1.03384   |            -1.61369  |          3.61685 |               5383.31 |           54.1525 | -0.0%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-15    |    1.02006   |            -3.79715  |          2.73497 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers4_size128_eval_batch_size1000__Walker2d-v4_27-05-2025_23-44-49   |    1.08183   |            -3.97674  |          3.40165 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-44-25   |    1.0611    |            -5.85004  |          3.99641 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers3_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-01    |    1.23431   |            -7.29152  |          3.04158 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps3000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-42-22    |    1.19016   |            -7.42819  |          3.67495 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps2000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-41-39    |    1.22054   |            -7.58086  |          3.31368 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps5000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-43-22    |    1.31817   |            -8.07359  |          3.27556 |               5383.31 |           54.1525 | -0.1%          |
| q1_bc_Walker2d-v4_lr0.05_n_iter1_steps10000n_layers4_size256_eval_batch_size1000__Walker2d-v4_27-05-2025_23-45-02   |    1.34194   |            -8.84943  |          4.22027 |               5383.31 |           54.1525 | -0.2%          |

