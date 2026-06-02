## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord fundamentally links a bank's capital requirements to its underlying credit risks. Under the Internal Ratings-Based (IRB) approach, banks can use their own internal models to estimate risk parameters like Probability of Default (PD). However, this regulatory advantage comes with strict obligations:

**Pillar 2 (Supervisory Review Process)** requires that any internal model be:
- **Robust and validated** - Demonstrated to perform reliably across different scenarios
- **Transparent and explainable** - Regulators must understand how the model reaches decisions
- **Thoroughly documented** - Complete records of assumptions, methodologies, and validation

For Bati Bank, interpretability is not optional—it's a compliance necessity. We must be able to:
- Explain to regulators why specific features predict credit risk
- Validate that no discriminatory patterns exist in model decisions
- Audit individual credit decisions when challenged by customers

### 2. Why is a proxy variable necessary, and what are the business risks?

**Necessity:** The raw eCommerce dataset contains no historical default labels—the ground truth for credit risk modeling. To train any supervised model, we must engineer a proxy variable that correlates with true default risk.

**Our Approach:** We'll use RFM (Recency, Frequency, Monetary) analysis to identify disengaged customers as high-risk proxies. The assumption: customers with declining engagement are less likely to honor credit obligations.

**Business Risks:**
- **False Positives (Lost Revenue):** Creditworthy customers flagged as risky → denied credit → lost customer lifetime value
- **False Negatives (Direct Losses):** Risky customers flagged as safe → approved loans → defaults and financial losses
- **Regulatory Scrutiny:** Misclassifications could lead to discrimination claims or consumer protection violations

### 3. Trade-offs: Interpretable vs. High-Performance Models

| Aspect | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|--------|--------------------------|--------------------------------------|
| Interpretability | High - Clear coefficient interpretation | Low - Black box by default |
| Regulatory Acceptance | Preferred - Well-understood methodology | Requires SHAP/LIME for explainability |
| Predictive Performance | Moderate - Linear boundaries only | High - Captures complex interactions |
| Implementation Complexity | Simple - Fast training and inference | Complex - More tuning, longer training |
| Fairness Auditing | Easy - Direct coefficient inspection | Difficult - Requires post-hoc analysis |

**Our Strategy:** We'll implement both approaches, using SHAP values to explain complex model decisions, ensuring we can defend either choice to regulators[citation:3].