# PPO Trainer for Transformers: Technical Documentation

## Fundamentals

This section explains the key mathematical foundations behind the PPO algorithm, specifically tailored for transformer-based language models.

### 1. Policy Optimization Objective

The core of PPO is to update the policy by maximizing a **clipped surrogate objective**:

$$
L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where:

- \(r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\) is the probability ratio between the new and old policy.
- \(\hat{A}_t\) is the estimated advantage function at timestep \(t\).
- \(\epsilon\) is the clipping parameter (typically 0.2).

The **clipping** prevents large updates that destabilize training.

### 2. Advantage Estimation (Generalized Advantage Estimation - GAE)

The advantage \(\hat{A}_t\) estimates how much better an action is compared to the average action:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE accumulates these temporal differences:

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
$$

This can be computed recursively:

$$
\hat{A}_t = \delta_t + \gamma \lambda (1 - d_{t+1}) \hat{A}_{t+1}
$$

where \(d_t\) is the done flag indicating if an episode ends.

- \(\gamma\) is the discount factor.
- \(\lambda\) is the GAE parameter balancing bias vs variance.

### 3. Value Function Loss

The value network is trained by minimizing the mean-squared error between predicted values and the **returns**:

$$
L_V = \frac{1}{2} (V(s_t) - R_t)^2
$$

where \(R_t = \hat{A}_t + V(s_t)\) is the bootstrapped return.

We optionally **clip** the value prediction to avoid large shifts:

$$
L_V = \frac{1}{2} \max \left( (V(s_t) - R_t)^2, (\text{clip}(V(s_t), V_{\text{old}}(s_t) - \epsilon, V_{\text{old}}(s_t) + \epsilon) - R_t)^2 \right)
$$

### 4. Entropy Bonus

Entropy regularization encourages exploration by maximizing the entropy \(H\) of the policy:

$$
H(\pi(\cdot|s)) = -\sum_a \pi(a|s) \log \pi(a|s)
$$

The entropy term is added to the final loss with coefficient \(\beta\).

### 5. Full PPO Loss

The final objective to minimize:

$$
\text{Loss} = L(\theta) + c_1 L_V - c_2 H
$$

where:
- \(c_1\) is the value loss coefficient.
- \(c_2\) is the entropy loss coefficient.

---

## Detailed PPO Update Formulation

### 1. Log-Probability Ratio

$$
\delta\ell_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
$$

### 2. Importance Sampling Ratio

$$
r_t(\theta) = \exp(\delta\ell_t) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

### 3. KL Divergence Approximations

$$
\widehat{\mathrm{KL}}_{\text{old}} = -\mathbb{E}_t[\delta\ell_t]
$$

$$
\widehat{\mathrm{KL}} = \mathbb{E}_t[(r_t(\theta) - 1) - \delta\ell_t]
$$

### 4. Clipped Fraction

$$
\mathrm{clipFrac} = \mathbb{E}_t\left[\mathbf{1}\{|r_t(\theta) - 1| > \epsilon\}\right]
$$

### 5. Advantage Normalization (Optional)

$$
\tilde{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}
$$

### 6. Policy (Surrogate) Loss

Unclipped policy loss:

$$
L^{(1)}_t = -\tilde{A}_t r_t(\theta)
$$

Clipped policy loss:

$$
L^{(2)}_t = -\tilde{A}_t \ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

Final policy loss:

$$
L_{\text{policy}} = \mathbb{E}_t\left[\max\left(L^{(1)}_t, L^{(2)}_t\right)\right]
$$

or equivalently

$$
L_{\text{policy}} = -\mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \ \mathrm{clip}(r_t(\theta), 1 \pm \epsilon) A_t\right)\right]
$$

### 7. Value Function Loss

Unclipped value loss:

$$
L^{\text{unc}}_{{\text{vf}},t} = (V_\theta(s_t) - R_t)^2
$$

Clipped value prediction:

$$
\bar{V}_t = V_{\text{old}}(s_t) + \mathrm{clip}(V_\theta(s_t) - V_{\text{old}}(s_t), -\epsilon, \epsilon)
$$

Clipped value loss:

$$
L^{\text{clip}}_{{\text{vf}},t} = (\bar{V}_t - R_t)^2
$$

Final value loss:

$$
L_{\text{vf}} = \begin{cases}
  \frac{1}{2} \mathbb{E}_t\left[\max\left(L^{\text{unc}}_{{\text{vf}},t}, L^{\text{clip}}_{{\text{vf}},t}\right)\right] & \text{if clipping value loss,} \\
  \frac{1}{2} \mathbb{E}_t\left[L^{\text{unc}}_{{\text{vf}},t}\right] & \text{otherwise}
\end{cases}
$$

### 8. Entropy Bonus

$$
H(\pi_\theta) = \mathbb{E}_t\left[\mathcal{H}\left(\pi_\theta(\cdot \mid s_t)\right)\right]
$$

### 9. Total Loss

$$
L(\theta) = L_{\text{policy}} - c_{\text{ent}} \ H(\pi_\theta) + c_{\text{vf}} \ L_{\text{vf}}
$$

where \(c_{\text{ent}}\) and \(c_{\text{vf}}\) are entropy and value loss coefficients.

### 10. Gradient Update

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
$$

with gradient clipping:

$$
\|\nabla \theta\|_\infty \leq \text{max\_grad\_norm}
$$

---

### Early Stopping Criterion

If

$$
\widehat{\mathrm{KL}} > \text{target\_kl}
$$

then terminate the update early to maintain stability.
