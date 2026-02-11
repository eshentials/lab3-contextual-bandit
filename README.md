# Lab 3: Contextual Bandit-Based News Article Recommendation

**Course**: Reinforcement Learning Fundamentals  
**Student**: Eshani Parulekar  
**Roll Number**: U20230008  
**Branch**: `eshani_U20230008`

---

## Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** based **news article recommendation system**.

- **Contexts**: user categories (`User1`, `User2`, `User3`) detected by a supervised classifier.
- **Arms**: news categories (Entertainment, Education, Tech, Crime).
- **Goal**: maximize reward (simulated user engagement) by learning which news category to show for each user context.

All core logic, experiments, and plots are contained in the notebook:

- `lab3_results_U20230008.ipynb`

---

## Approach and Design Decisions

### 1. Data and Preprocessing

- **User data**: `train_users.csv`, `test_users.csv` under `data/`.
  - The updated dataset contains **33 columns** (ID, 31 features, and label).
  - Features include behavioural signals (e.g., `clicks`, `purchase_amount`, `session_duration`, `engagement_score`, `loyalty_index`, `churn_risk_score`, etc.).
  - Labels: `user_1`, `user_2`, `user_3` → mapped to `User1`, `User2`, `User3`.
- **News data**: `news_articles.csv`.
  - Original HuffPost categories mapped into **4 super-categories**: `Entertainment`, `Education`, `Tech`, `Crime`.

**Preprocessing choices**:

- Dropped non-feature columns: `user_id`, raw `label`.
- Categorical features (`browser_version`, `region_code`, `subscriber`) encoded with **LabelEncoder**.
- Numerical features imputed with **median** (handles missing `age` and other numeric fields).
- All features scaled with **StandardScaler** before classification.

### 2. User Classification (Context Detector)

To get a reasonably good context signal (much better than random ~33%), we:

- Use **all 31 numeric + encoded categorical features** for classification.
- Split `train_users.csv` into a stratified train/test split (because the provided `test_users.csv` has no labels in the updated repo).
- Train a **Gradient Boosting Classifier** (`sklearn.ensemble.GradientBoostingClassifier`) with:
  - `n_estimators = 150`
  - `max_depth = 5`
  - `learning_rate = 0.1`

**Result (from internal evaluation in the notebook)**:

- **Accuracy ≈ 89.5%** on the held-out test split.
- This is **far above** random (33.3%) and satisfies the requirement that the classifier should not be near-chance.

This classifier is then used as the **context detector** for the contextual bandit algorithms.

### 3. Contextual Bandit Algorithms

We implement three bandit strategies, all **conditioned on user context**:

1. **Epsilon-Greedy Contextual Bandit**
   - Class: `EpsilonGreedyContextualBandit`.
   - Parameters: `n_contexts = 3`, `n_arms_per_context = 4`, `epsilon` ∈ {0.01, 0.1, 0.3}.
   - Maintains a Q-value and count for each of the 12 (context, arm) pairs.
   - For a given context index, selects an arm using epsilon-greedy within that context's 4 arms.

2. **UCB (Upper Confidence Bound) Contextual Bandit**
   - Class: `UCBContextualBandit`.
   - Parameters: `n_contexts = 3`, `n_arms_per_context = 4`, `C` ∈ {0.5, 1.0, 2.0}.
   - Uses a UCB score for each arm in the **current context**:  
     \( \text{UCB}(a) = Q(a) + C \sqrt{\frac{\ln t}{N(a)}} \)
   - Ensures systematic exploration of arms that are uncertain but potentially good.

3. **SoftMax Contextual Bandit**
   - Class: `SoftMaxContextualBandit`.
   - Parameters: `n_contexts = 3`, `n_arms_per_context = 4`, `\tau = 1.0`.
   - Converts Q-values in the current context to probabilities via SoftMax and samples an arm.

**Arm mapping** (as per handout):

- Arms 0–3:  `(User1, [Entertainment, Education, Tech, Crime])`
- Arms 4–7:  `(User2, [Entertainment, Education, Tech, Crime])`
- Arms 8–11: `(User3, [Entertainment, Education, Tech, Crime])`

A helper mapping is defined in the notebook to convert between `(context, category)` and global arm index `j`.

### 4. Reward Sampler Integration

- Uses the provided package `rlcmab-sampler` **without modification**.
- Initialized as:  
  `reward_sampler = sampler(8)`  (from roll number **U20230008 → 8**).
- All rewards are obtained **only** via `reward_sampler.sample(j)`.
- No synthetic or hard-coded rewards are used.

### 5. Recommendation Engine

The end-to-end CMAB recommendation flow is:

1. **Classify**: Predict user category (User1/User2/User3) using the trained Gradient Boosting classifier.
2. **Select Category**: Use one of the contextual bandit policies (e.g., best-performing UCB configuration) to select the news category for that context.
3. **Recommend**: Sample a random article from `news_articles.csv` belonging to that category.
4. **Output**: Return article details (headline, link, description, date).

All of this is encapsulated in a `NewsRecommendationEngine` class in the notebook.

---

## Key Results and Observations

### 1. User Classification

- **Model**: Gradient Boosting Classifier.
- **Accuracy**: ~**89.5%** on held-out test split.
- This is **significantly better than random (33.3%)**, making it suitable as a context detector for the bandit.

### 2. Bandit Algorithms

- All three algorithms (Epsilon-Greedy, UCB, SoftMax) are run for **T = 10,000** steps.
- The notebook includes plots of **Average Reward vs Time** per context, and **hyperparameter comparisons**:
  - Epsilon-Greedy: compares different \( \epsilon \) values.
  - UCB: compares different \( C \) values.

**General observations (as seen in the notebook plots):**

- UCB with a reasonably large \( C \) (e.g., 2.0) often performs best in terms of long-term average reward.
- Epsilon-Greedy with \( \epsilon \approx 0.1 \) balances exploration and exploitation well.
- SoftMax with \( \tau = 1.0 \) gives smooth probabilistic exploration but is sensitive to the learned Q-scale.
- Different contexts (User1 vs User2 vs User3) learn **distinct preferences** over the four news categories.

---

## How to Reproduce Experiments

### 1. Clone the Repository

```bash
git clone https://github.com/eshentials/lab3-contextual-bandit.git
cd lab3-contextual-bandit

# Checkout the submission branch
git checkout eshani_U20230008
```

### 2. Set Up Environment

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn rlcmab-sampler jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook lab3_results_U20230008.ipynb
```

In Jupyter:

1. Open `lab3_results_U20230008.ipynb`.
2. Run: **Kernel → Restart & Run All**.
3. Wait for all cells to finish executing.

This will:

- Train the user classifier.
- Run all three contextual bandit algorithms for T = 10,000.
- Generate all plots (Average Reward vs Time, hyperparameter comparisons, etc.).
- Demonstrate the end-to-end recommendation engine.

### 4. Files Required

Ensure the following files exist (as in the repo):

- `data/train_users.csv`
- `data/test_users.csv`
- `data/news_articles.csv`
- `lab3_results_U20230008.ipynb`

---

## External References

Concepts and algorithms implemented are based on standard RL / bandit literature and library docs, including:

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
  - Chapter 2: Multi-Armed Bandits.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time analysis of the multiarmed bandit problem*.
- Scikit-learn documentation:  
  - Gradient Boosting Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html  
  - Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Numpy, Pandas, Matplotlib, and Seaborn official docs for data handling and visualization.

---

## Notes

- The **primary implementation** and all detailed analysis live in `lab3_results_U20230008.ipynb`.
- This `README.md` is a **high-level guide** to understand the design, results, and how to reproduce the experiments.

