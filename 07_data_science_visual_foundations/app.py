"""Data Science Visual Foundations
=====================================
Not a business-KPI predictor — a **teaching tool**. Four foundational
concepts every data scientist needs deep intuition for, each made
interactive and grounded in real data or a fully worked numeric example:

1. **Naive Bayes** — a live spam filter on the real SMS Spam Collection
   dataset, with a word-by-word log-probability breakdown of *why* a
   message is classified spam or ham.
2. **Model evaluation** — confusion matrix, Type I/II errors, ROC-AUC, a
   cost matrix, and the precision/recall trade-off, all driven by one
   interactive decision-threshold slider on real classifier scores.
3. **Gradient descent** — a real 1-feature linear regression (diabetes
   BMI → disease progression) fit live via gradient descent, with the loss
   surface, descent path, and per-iteration loss curve all visible, so you
   can *see* what an unstable learning rate does.
4. **The chain rule → backpropagation** — a fully worked, numerically
   transparent 1-1-1 neuron network forward + backward pass.

Ends with a 10-question interview-prep quiz across all four topics.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "sms_spam.tsv"


@st.cache_data
def load_sms() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    return df


@st.cache_resource(show_spinner=True)
def train_naive_bayes(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    vec = CountVectorizer(stop_words="english", min_df=3)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = MultinomialNB()
    clf.fit(Xtr, y_train)
    proba_test = clf.predict_proba(Xte)[:, list(clf.classes_).index("spam")]
    return vec, clf, (X_test, y_test, proba_test)


def explain_message(vec: CountVectorizer, clf: MultinomialNB, message: str) -> pd.DataFrame:
    """Word-by-word log-likelihood breakdown, the arithmetic Naive Bayes
    actually does under the hood."""
    tokens = vec.build_analyzer()(message)
    vocab = vec.vocabulary_
    classes = list(clf.classes_)
    spam_idx, ham_idx = classes.index("spam"), classes.index("ham")
    rows = []
    for tok in tokens:
        if tok in vocab:
            j = vocab[tok]
            rows.append({
                "word": tok,
                "log P(word | spam)": clf.feature_log_prob_[spam_idx, j],
                "log P(word | ham)": clf.feature_log_prob_[ham_idx, j],
            })
    return pd.DataFrame(rows)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


QUIZ = [
    {"q": "Naive Bayes is called 'naive' because it assumes...",
     "options": ["All features are independent given the class", "The prior is always 50/50",
                 "There is no noise in the data", "Only 2 classes exist"],
     "answer": 0,
     "explain": "It applies Bayes' theorem assuming every feature (e.g. each word) is conditionally independent given the class — rarely true in reality, but the assumption makes the math tractable and, empirically, still works well for text."},
    {"q": "A Type I error is...",
     "options": ["A false negative", "A false positive", "A true positive", "A model that never converges"],
     "answer": 1,
     "explain": "Type I = false positive (rejecting a true null / raising a false alarm). Type II = false negative (missing a real effect / a real case)."},
    {"q": "A Type II error is...",
     "options": ["A false positive", "A false negative", "A perfect prediction", "An outlier"],
     "answer": 1,
     "explain": "Type II = false negative — the model says 'no' when the true answer was 'yes' (e.g. calling a fraud case legitimate)."},
    {"q": "ROC-AUC of 0.5 means...",
     "options": ["Perfect classifier", "The model is exactly as good as random guessing",
                 "The model is perfectly wrong (invert its predictions)", "50% of labels are missing"],
     "answer": 1,
     "explain": "AUC=0.5 sits on the diagonal 'no-skill' line — the model's ranking of positives vs negatives is no better than chance. (AUC=0.0 would be perfectly *inverted*, i.e. trivially fixable by flipping predictions.)"},
    {"q": "Precision answers which question?",
     "options": ["Of all actual positives, how many did we find?",
                 "Of everything we flagged positive, how many were actually positive?",
                 "How accurate is the model overall?", "How fast is the model?"],
     "answer": 1,
     "explain": "Precision = TP / (TP + FP): trustworthiness of a positive prediction. Recall = TP / (TP + FN): coverage of actual positives."},
    {"q": "Recall answers which question?",
     "options": ["Of everything flagged positive, how many were right?",
                 "Of all actual positives, how many did the model catch?",
                 "How many features were used?", "How long did training take?"],
     "answer": 1,
     "explain": "Recall (a.k.a. sensitivity, true positive rate) is about *coverage* — missing real positives (false negatives) hurts recall."},
    {"q": "Raising the decision threshold for a 'positive' prediction typically...",
     "options": ["Increases both precision and recall", "Decreases both precision and recall",
                 "Increases precision, decreases recall", "Has no effect on either"],
     "answer": 2,
     "explain": "A higher bar to call something positive means fewer, more confident positive calls (precision ↑) but more real positives get missed (recall ↓) — the core precision/recall trade-off."},
    {"q": "In gradient descent, if the learning rate is too large, you typically see...",
     "options": ["Slow, smooth convergence", "The loss oscillating or diverging instead of decreasing",
                 "No change in the loss at all", "Instant convergence in 1 step, always"],
     "answer": 1,
     "explain": "Too-large a step overshoots the minimum on every update; the loss can oscillate or blow up instead of settling. Too-small a rate is stable but painfully slow."},
    {"q": "The gradient descent update rule is w ← w − η·∇L(w). What does η do?",
     "options": ["It's the loss value", "It's the learning rate — how big a step to take down the gradient",
                 "It's the number of training examples", "It's a regularization penalty"],
     "answer": 1,
     "explain": "η (eta) scales the gradient step. This is literally the only new symbol in the whole optimization loop that a practitioner tunes directly."},
    {"q": "Backpropagation computes gradients for earlier layers using...",
     "options": ["Random search", "The chain rule, multiplying local derivatives layer by layer backward from the loss",
                 "A lookup table of precomputed gradients", "Only the first layer's weights"],
     "answer": 1,
     "explain": "∂L/∂w1 = ∂L/∂output · ∂output/∂hidden · ∂hidden/∂w1 — each factor is a *local* derivative; the chain rule strings them together from the loss backward to every weight, however deep the network."},
]


def main():
    theme.apply("🎓", "Data Science Visual Foundations")
    theme.hero(
        "🎓", "Data Science Visual Foundations",
        "An interactive teaching tool, not a KPI predictor: Naive Bayes, model evaluation "
        "(confusion matrix / ROC / cost trade-offs), gradient descent, and the chain rule → "
        "backpropagation — each with a live, worked example and a closing quiz.",
        ["Naive Bayes", "Gradient Descent", "Backprop", "10-question quiz"],
    )

    sms = load_sms()
    vec, clf, (X_test, y_test, proba_test) = train_naive_bayes(sms)

    with st.sidebar:
        st.subheader("📁 Dataset (for Naive Bayes & evaluation)")
        st.metric("SMS messages", f"{len(sms):,}")
        st.metric("Spam rate", f"{(sms['label']=='spam').mean()*100:.1f}%")
        st.caption(
            "Real SMS Spam Collection dataset (5,574 real text messages, UCI ML "
            "Repository) — used for the Naive Bayes and model-evaluation sections. "
            "Gradient descent uses the real scikit-learn diabetes dataset."
        )

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding → Learning objectives ------------------------------
    with tabs[0]:
        st.subheader("Why these four foundations")
        st.markdown(
            """
Every supervised-learning model you'll ever ship rests on the same four
ideas, whichever library wraps them:

1. **A probabilistic classifier** (Naive Bayes is the simplest one you can
   compute *by hand* — great for building real intuition for what
   "learning from data" mechanically is).
2. **Honest evaluation** — accuracy alone lies on imbalanced problems (see
   [project 04](../04_fraud_anomaly_detection)); every practitioner needs
   Type I/II errors, ROC-AUC, and the precision/recall trade-off as
   reflexes, not vocabulary.
3. **Gradient descent** — the optimization loop underneath *every* model in
   this portfolio, from `LogisticRegression` to gradient boosting to the
   nano transformer in [project 08](../08_nano_transformer_llm).
4. **The chain rule** — the one calculus fact that makes training a neural
   network of any depth mechanically possible (backpropagation *is* the
   chain rule, applied systematically).

Each tab below is interactive — change the inputs and watch the math
respond — and the Deployment tab closes with a 10-question quiz.
            """
        )

    # 2. Data Understanding → SMS Spam EDA ------------------------------------------
    with tabs[1]:
        st.subheader("The dataset behind the Naive Bayes demo")
        c1, c2 = st.columns(2)
        with c1:
            counts = sms["label"].value_counts()
            fig = px.bar(counts, title="Spam vs. ham message counts", color=counts.index,
                         color_discrete_map={"ham": "#6ee7f2", "spam": "#fb7185"})
            theme.style_fig(fig, 340); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with c2:
            sms_len = sms.assign(length=sms["message"].str.len())
            fig = px.box(sms_len, x="label", y="length", color="label", title="Message length by class",
                         color_discrete_map={"ham": "#6ee7f2", "spam": "#fb7185"})
            theme.style_fig(fig, 340); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        vocab_counts = pd.Series(vec.get_feature_names_out())
        spam_idx = list(clf.classes_).index("spam")
        top_spam_words = pd.Series(clf.feature_log_prob_[spam_idx], index=vocab_counts).sort_values(ascending=False).head(15)
        fig = px.bar(top_spam_words[::-1], orientation="h", title="Top 15 words by log P(word | spam)")
        theme.style_fig(fig, 400); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.dataframe(sms.sample(8, random_state=1), width='stretch')

    # 3. Data Preparation → Bag-of-words + Bayes' theorem ---------------------------
    with tabs[2]:
        st.subheader("From raw text to numbers: bag-of-words")
        st.markdown(
            """
Naive Bayes can't read English — it needs numbers. `CountVectorizer` turns
each message into a vector of word counts (a "bag of words": order is
discarded, only *which words and how many times* survives). Try it below.
            """
        )
        sample_text = st.text_input("Try tokenizing a message", "Win a FREE prize now, click the link!!!")
        tokens = vec.build_analyzer()(sample_text)
        st.write("Tokens kept after lowercasing, stop-word removal, and the `min_df=3` vocabulary filter:")
        st.code(tokens if tokens else "(no tokens survived filtering)", language="text")

        st.markdown(
            r"""
#### Bayes' theorem, the equation Naive Bayes is built on

$$P(\text{spam} \mid \text{words}) = \frac{P(\text{words} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{words})}$$

The **naive** part: assume every word is conditionally independent given
the class, so $P(\text{words}\mid\text{spam}) = \prod_i P(w_i \mid \text{spam})$
— one multiplication per word instead of one impossibly-high-dimensional
joint probability. In practice this is computed in **log-space** (summing
log-probabilities) to avoid numerical underflow from multiplying many
small numbers together — that's exactly what the word-by-word breakdown in
the Modeling tab shows.
            """
        )

    # 4. Modeling → NB explainer + gradient descent + backprop -----------------------
    with tabs[3]:
        st.markdown("### 🅰️ Naive Bayes, word by word")
        msg = st.text_input("Classify a message and see the math", "Congratulations! You won a free ticket, call now!", key="nb_msg")
        pred_proba = clf.predict_proba(vec.transform([msg]))[0]
        spam_p = pred_proba[list(clf.classes_).index("spam")]
        st.metric("P(spam | message)", f"{spam_p*100:.1f}%")
        breakdown = explain_message(vec, clf, msg)
        if not breakdown.empty:
            breakdown["spam - ham (this word's swing toward spam)"] = breakdown["log P(word | spam)"] - breakdown["log P(word | ham)"]
            fig = px.bar(breakdown.sort_values("spam - ham (this word's swing toward spam)"),
                         x="spam - ham (this word's swing toward spam)", y="word", orientation="h",
                         title="Each word's log-odds contribution (right = pushes toward spam)")
            theme.style_fig(fig, min(60 + 28 * len(breakdown), 420)); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("None of these words are in the trained vocabulary — try different wording.")

        st.markdown("---")
        st.markdown("### 📉 Gradient descent, live")
        st.caption("Fitting `progression = w · bmi + b` on the real diabetes dataset — watch what learning rate does.")
        diabetes = load_diabetes(as_frame=True)
        x = diabetes.data["bmi"].values
        x = (x - x.mean()) / x.std()
        y = diabetes.target.values
        y = (y - y.mean()) / y.std()

        c1, c2 = st.columns(2)
        lr = c1.slider("Learning rate η", 0.01, 1.6, 0.3, 0.01)
        n_iter = c2.slider("Iterations", 5, 100, 30)

        w, b = 0.0, 0.0
        history = []
        for i in range(n_iter):
            pred = w * x + b
            err = pred - y
            loss = float(np.mean(err ** 2))
            history.append({"iter": i, "loss": loss, "w": w, "b": b})
            grad_w = float(np.mean(2 * err * x))
            grad_b = float(np.mean(2 * err))
            w -= lr * grad_w
            b -= lr * grad_b
        hist_df = pd.DataFrame(history)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hist_df, x="iter", y="loss", markers=True, title="Loss per iteration")
            theme.style_fig(fig, 360)
            if hist_df["loss"].iloc[-1] > hist_df["loss"].iloc[0]:
                st.error("Diverging! The learning rate is too large — the steps overshoot the minimum.", icon="⚠️")
            st.plotly_chart(fig, width='stretch')
        with c2:
            w_range = np.linspace(-1.5, 1.5, 40)
            b_range = np.linspace(-1.5, 1.5, 40)
            WW, BB = np.meshgrid(w_range, b_range)
            Z = np.array([[np.mean((ww * x + bb - y) ** 2) for ww in w_range] for bb in b_range])
            fig = go.Figure(data=go.Contour(x=w_range, y=b_range, z=Z, colorscale="Tealgrn", showscale=False))
            fig.add_trace(go.Scatter(x=hist_df["w"], y=hist_df["b"], mode="lines+markers",
                                      marker=dict(size=5, color="#fb7185"), line=dict(color="#fb7185"), name="descent path"))
            fig.update_layout(title="Loss surface (w, b) + descent path", xaxis_title="w", yaxis_title="b")
            theme.style_fig(fig, 360)
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("### 🔗 The chain rule → backpropagation")
        st.caption("A 1-1-1 network: input → hidden (sigmoid) → output. Adjust the values and watch every gradient update.")
        c1, c2, c3, c4 = st.columns(4)
        xi = c1.number_input("input x", value=1.5, step=0.1)
        w1 = c2.number_input("weight w₁ (x → hidden)", value=0.8, step=0.1)
        w2 = c3.number_input("weight w₂ (hidden → output)", value=-0.5, step=0.1)
        target = c4.number_input("target y", value=1.0, step=0.1)

        z1 = w1 * xi
        a1 = sigmoid(z1)
        z2 = w2 * a1
        output = z2
        loss = 0.5 * (output - target) ** 2

        dL_doutput = output - target
        doutput_dz2 = 1.0
        dz2_dw2 = a1
        dz2_da1 = w2
        da1_dz1 = a1 * (1 - a1)
        dz1_dw1 = xi

        dL_dw2 = dL_doutput * doutput_dz2 * dz2_dw2
        dL_dw1 = dL_doutput * doutput_dz2 * dz2_da1 * da1_dz1 * dz1_dw1

        fwd = pd.DataFrame({
            "step": ["z1 = w1·x", "a1 = sigmoid(z1)", "z2 = w2·a1 (output)", "loss = 0.5(output-y)²"],
            "value": [z1, a1, z2, loss],
        })
        bwd = pd.DataFrame({
            "chain-rule factor": ["∂L/∂output", "∂output/∂z2", "∂z2/∂w2", "∂z2/∂a1", "∂a1/∂z1", "∂z1/∂w1"],
            "value": [dL_doutput, doutput_dz2, dz2_dw2, dz2_da1, da1_dz1, dz1_dw1],
        })
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Forward pass")
            st.dataframe(fwd.style.format({"value": "{:.4f}"}), width='stretch', hide_index=True)
        with c2:
            st.caption("Backward pass — each local derivative")
            st.dataframe(bwd.style.format({"value": "{:.4f}"}), width='stretch', hide_index=True)

        m1, m2 = st.columns(2)
        m1.metric("∂L/∂w2 = ∂L/∂output · ∂output/∂z2 · ∂z2/∂w2", f"{dL_dw2:.4f}")
        m2.metric("∂L/∂w1 = (chain of 5 factors above)", f"{dL_dw1:.4f}")

    # 5. Evaluation → confusion matrix / ROC / cost trade-off ------------------------
    with tabs[4]:
        st.subheader("Confusion matrix, Type I/II errors, ROC-AUC, and the precision/recall trade-off")
        st.caption("All driven by the real Naive Bayes spam classifier's held-out predictions.")

        threshold = st.slider("Decision threshold — P(spam) ≥ this ⇒ predict spam", 0.01, 0.99, 0.5, 0.01)
        preds = (proba_test >= threshold).astype(int)
        y_true = (y_test.values == "spam").astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives (caught spam)", tp)
        c2.metric("False Positives — Type I (ham flagged as spam)", fp)
        c3.metric("False Negatives — Type II (missed spam)", fn)
        c4.metric("True Negatives (correctly kept ham)", tn)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(cm, text_auto=True, x=["Pred: ham", "Pred: spam"], y=["True: ham", "True: spam"],
                             color_continuous_scale="Tealgrn", title=f"Confusion matrix @ threshold={threshold:.2f}")
            theme.style_fig(fig, 360)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fpr, tpr, _ = roc_curve(y_true, proba_test)
            auc = roc_auc_score(y_true, proba_test)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})", line=dict(color="#6ee7f2", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#93a1c2", dash="dash"), name="random"))
            fig.update_layout(title="ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            theme.style_fig(fig, 360)
            st.plotly_chart(fig, width='stretch')

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        st.markdown("#### Cost matrix — what if a missed spam and a false alarm cost different amounts?")
        c1, c2 = st.columns(2)
        cost_fn = c1.slider("Cost of missing a real spam (annoyance/risk, $)", 0.0, 10.0, 1.0, 0.5)
        cost_fp = c2.slider("Cost of blocking a real message (lost communication, $)", 0.0, 10.0, 3.0, 0.5)
        total_cost = fn * cost_fn + fp * cost_fp
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision:.2%}")
        m2.metric("Recall", f"{recall:.2%}")
        m3.metric("Total cost at this threshold", f"${total_cost:,.1f}")
        st.caption("Slide the decision threshold above and watch precision/recall trade against each other, "
                   "and the total cost shift — there is rarely one 'correct' threshold, only the one that fits your costs.")

    # 6. Deployment → Quiz ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("🧠 Interview-prep quiz")
        st.caption("10 questions across everything above. Answer, then check your score.")

        answers = {}
        for i, item in enumerate(QUIZ):
            answers[i] = st.radio(f"**{i+1}. {item['q']}**", item["options"], index=None, key=f"quiz_{i}")

        if st.button("✅ Check my answers", type="primary"):
            score = 0
            for i, item in enumerate(QUIZ):
                chosen = answers[i]
                correct = item["options"][item["answer"]]
                is_right = chosen == correct
                score += int(is_right)
                icon = "✅" if is_right else ("❌" if chosen is not None else "⬜")
                with st.expander(f"{icon} Q{i+1}: {item['q']}"):
                    st.write(f"Your answer: {chosen or '(not answered)'}")
                    st.write(f"Correct answer: **{correct}**")
                    st.caption(item["explain"])
            st.success(f"Score: {score} / {len(QUIZ)}", icon="🎯")

    theme.footer("07 · Data Science Visual Foundations")


if __name__ == "__main__":
    main()
