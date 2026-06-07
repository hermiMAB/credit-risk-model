# Credit Scoring & Risk Management: Business Understanding

## Data Source
The dataset utilized for this credit scoring project originates from the **Xente Fraud Detection Challenge** (hosted on Zindi/Kaggle). It contains approximately 95,662 eCommerce transaction logs, which have been repurposed in this repository to engineer a behavioral risk proxy and predict credit default likelihood for Bati Bank's Buy-Now-Pay-Later (BNPL) service.

*Note: Due to file size constraints and strict version control practices, the raw dataset (`data.csv`) is tracked locally via DVC and is not hosted directly in this GitHub repository.*
This document outlines the core business and regulatory considerations for developing credit scoring models, specifically addressing the requirements of the Basel II framework, the use of proxy variables in low-data environments, and the critical trade-offs between model interpretability and predictive performance.

---

## 1. Regulatory Influence: Basel II Accord & Risk Measurement

The **Basel II Accord** established a rigorous regulatory framework aimed at strengthening the international banking system by making capital allocation highly sensitive to actual credit risk. Under Basel II, banks are empowered (and incentivized) to utilize internal credit risk models—such as credit scoring systems—to determine their regulatory capital requirements. 

This emphasis on risk measurement profoundly influences the need for **interpretable** and **well-documented** models due to several critical business and compliance imperatives:

* **Risk-Sensitive Capital Frameworks:** Basel II requires that a bank's capital reserves be directly aligned with its risk exposure. If a model is a "black box," senior management and regulators cannot verify whether the capital calculated is truly reflective of the underlying portfolio risk.
* **Transparency for Stakeholders:** The framework enforces market discipline through transparency. Lenders, investors, and rating agencies must be able to assess appropriate information to make informed decisions. An interpretable model ensures that the logic behind risk pricing and capital allocation is open to scrutiny.
* **Auditability and Validation:** Regulatory bodies require comprehensive validation of internal rating systems. Well-documented models allow internal auditors and external regulators to trace how a specific credit decision or risk score was derived, ensuring the model operates reliably under various economic scenarios.

---

## 2. The Necessity and Risks of Proxy Variables

In credit scoring, particularly for segments like **Micro and Small Enterprises (MSEs)** or "thin-file" borrowers, historical default data is often sparse or entirely unavailable. Without a direct historical "default" label, predicting creditworthiness becomes a significant challenge.

### Why Proxy Variables Are Necessary
When a traditional credit history is limited or non-existent, **proxy variables** serve as essential stand-ins to approximate a borrower’s financial health and repayment behavior. These alternatives typically include:
* **Behavioral Data:** Digital transaction histories, cash flow patterns in operating accounts, or utility payment consistencies.
* **Alternative Data:** Mobile phone usage metrics, e-commerce sales volumes, or professional social media activity.

These variables help fill the data gap, allowing financial institutions to generate an accurate credit score and extend credit to underserved but creditworthy segments.

### Business and Regulatory Risks Introduced by Proxies
While proxy variables unlock new lending opportunities, they introduce severe operational, legal, and ethical risks:
* **Proxy Discrimination (Unintentional Bias):** Alternative data points can unintentionally correlate strongly with protected characteristics such as race, gender, postal code (redlining), or religion. For example, specific social media patterns or geographical transaction hubs might mirror demographic divides, leading to systemic, unlawful discrimination and severe legal penalties.
* **The "Black Box" Escalation:** Integrating complex, high-dimensional proxy variables often requires sophisticated machine learning techniques. This accelerates the shift toward "black box" modeling, where tracking *why* a proxy is driving a specific credit decision becomes nearly impossible.
* **Spurious Correlations:** Proxies can capture temporary or accidental trends that do not reflect true creditworthiness, leading to model degradation during macroeconomic shifts.

---

## 3. Model Architecture Trade-offs in Regulated Contexts

When deploying credit scoring models within a regulated financial ecosystem, institutions must balance the trade-offs between traditional statistical methods and advanced machine learning algorithms.

### Traditional Models (e.g., Logistic Regression with Weight of Evidence - WoE)
* **Interpretability:** **Extremely High.** Relationships are modeled linearly. The Weight of Evidence (WoE) transformation allows risk managers to see exactly how each feature bucket impacts the final credit score. This satisfies regulatory mandates effortlessly and provides clear "adverse action" reasons to rejected applicants.
* **Predictive Power:** **Moderate.** These models struggle to capture complex, non-linear interactions between variables automatically, which can result in lower overall predictive accuracy and higher rates of missed credit opportunities or unexpected defaults.
* **Computational Efficiency:** Highly efficient, training in seconds or minutes on standard infrastructure.

### High-Performance Models (e.g., Gradient Boosting / XGBoost)
* **Interpretability:** **Low (Historically "Black Box").** Due to the ensemble nature of hundreds of sequential decision trees and complex non-linear feature transformations, individual decisions are highly opaque.
* **Predictive Power:** **Extremely High.** Gradient boosting consistently outperforms traditional linear methods in empirical experiments by capturing deep, nuanced interactions between variables, drastically reducing non-performing loans (NPLs).
* **Computational Efficiency:** Highly resource-intensive. In large-scale credit experiments, training complex architectures like XGBoost can take hours and require significant computational overhead.

### Mitigating the Trade-off
To bridge this gap and leverage high-performance architectures without violating regulatory compliance, institutions are increasingly adopting **model-agnostic interpretability techniques**:
* **SHAP (SHapley Additive exPlanations):** Explains individual predictions by allocating credit to each feature based on cooperative game theory, offering mathematically rigorous local transparency.
* **LIME (Local Interpretable Model-agnostic Explanations):** Approximates the complex model locally around a specific borrower's profile using a simple linear model to explain that distinct decision.

By implementing SHAP or LIME alongside Gradient Boosting, financial institutions can fulfill regulatory expectations for transparency while capturing the superior financial returns of high-performance risk modeling.

## 4. The Unique Dynamics of Buy-Now-Pay-Later (BNPL) Risk

While traditional credit scoring evaluates long-term lending (e.g., mortgages or auto loans), Bati Bank's BNPL service operates under a significantly different set of business and risk constraints. The model must be tailored to address these unique dynamics:
* **Velocity of Decision-Making:** BNPL transactions occur at checkout. The credit scoring model must execute inferences in milliseconds, requiring a highly optimized deployment architecture (low-latency APIs) compared to traditional loan underwriting, which may take days.
* **Credit Stacking and Invisible Debt:** BNPL users often utilize multiple platforms simultaneously. Because BNPL micro-loans are frequently not reported to traditional credit bureaus in real-time, models must rely heavily on immediate behavioral proxies (like sudden spikes in e-commerce transaction frequency) to detect potential over-leverage before default occurs.
* **Margin Sensitivity:** BNPL models typically operate on thin profit margins generated by merchant fees and late charges. A marginal increase in the False Positive rate (rejecting good customers) directly impacts top-line revenue, while a spike in False Negatives (approving bad customers) quickly erodes profitability. The model's classification threshold must be rigorously calibrated to business revenue goals.

## 5. Data Privacy, Consent, and Ethical Data Usage

Utilizing the Xente eCommerce transaction logs for credit decisioning introduces stringent data governance requirements. Repurposing transactional data for financial underwriting crosses into heavily regulated territory regarding consumer rights:
* **Data Minimization and Purpose Limitation:** Regulatory frameworks (such as GDPR or local equivalents) dictate that institutions should only process data strictly necessary for the intended purpose. The engineering of proxy variables must be defensible; capturing excessive behavioral data without clear predictive value exposes the business to regulatory fines.
* **Consent and Transparency:** Customers must be explicitly informed that their e-commerce behavior will be used to determine their credit limits. The business must establish clear "opt-in" mechanisms and provide avenues for consumers to challenge decisions made by the automated system.
* **Financial Inclusion vs. Predatory Lending:** The ethical objective of using alternative data is to expand credit access to the unbanked. However, the model must be audited to ensure it is not systematically identifying vulnerable populations and subjecting them to high-interest debt cycles they cannot escape.

## 6. Model Lifecycle, MLOps, and Concept Drift

A credit scoring model built on alternative data is not a static asset. Behavioral data is highly volatile and sensitive to external factors, requiring a robust Machine Learning Operations (MLOps) strategy:
* **Concept Drift and Macroeconomic Shocks:** Consumer spending habits and e-commerce patterns change rapidly due to inflation, seasonal trends, or economic downturns. A proxy variable that strongly indicated creditworthiness in 2023 might become completely irrelevant in 2024.
* **Continuous Monitoring:** Bati Bank must implement automated monitoring systems to track the distribution of incoming transaction data (Data Drift) and the actual default rates against the model's predictions (Concept Drift).
* **Automated Retraining Pipelines:** When performance degrades beyond acceptable business thresholds, the infrastructure must support rapid retraining and redeployment. This requires versioning both the data and the models (e.g., using DVC and MLflow) to ensure strict auditability and seamless rollbacks if a newly deployed model underperforms.
