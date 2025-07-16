import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from lime import lime_tabular
import shap
import matplotlib.pyplot as plt

# Define 1D CNN model
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 49, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# UI
st.title("Drug Response Prediction using CNN + XAI")
uploaded_file = st.file_uploader("Upload preprocessed CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Response" not in df.columns:
        st.error("CSV must include a 'Response' column for supervised training.")
    else:
        y = df["Response"]

        # Drop non-numeric and metadata columns
        drop_cols = [
            "Drug Name", "Drug ID", "Cell Line Name", "Cosmic ID", "TCGA Classification",
            "Tissue", "Tissue Sub-type", "IC50", "AUC", "Max Conc", "RMSE",
            "Z score", "Dataset Version", "Response"
        ]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=100)
        X_selected = selector.fit_transform(X, y)
        selected_feature_names = X.columns[selector.get_support()]  # Save actual names

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        # Train model
        model = CNN1D()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            model.train()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        st.success("✅ Model trained successfully")

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_tensor), dim=1)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(cm)

        # LIME explanation
        st.subheader("🔍 LIME Explanation (Styled Output)")

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=selected_feature_names.tolist(),
            class_names=["Resistant", "Sensitive"],
            discretize_continuous=True,
            mode="classification"
        )

        def predict_fn(numpy_input):
            input_tensor = torch.tensor(numpy_input.astype(np.float32)).unsqueeze(1)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
            return probs

        idx = st.number_input("Pick an index to explain", min_value=0, max_value=len(X_scaled)-1, value=0)

        # Get prediction probabilities and predicted label
        probs = predict_fn(X_scaled[[idx]])[0]
        predicted_label = np.argmax(probs)

        # Run LIME explanation
        lime_exp = lime_explainer.explain_instance(
            X_scaled[idx],
            predict_fn,
            num_features=10,
            top_labels=2
        )

        # === Layout ===
        # 🎯 Prediction probabilities
        st.markdown("### 🎯 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Class": ["Resistant", "Sensitive"],
            "Probability": [round(p, 3) for p in probs]
        })
        st.bar_chart(prob_df.set_index("Class"))

        # 📊 Feature contributions
        st.markdown("### 📊 Feature Contributions (LIME)")

        feat_weights = lime_exp.as_list(label=predicted_label)
        features = [f for f, _ in feat_weights]
        weights = [w for _, w in feat_weights]
        colors = ['#1f77b4' if w > 0 else '#d62728' for w in weights]  # blue = positive, red = negative

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features, weights, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"LIME Explanation for: {'Resistant' if predicted_label == 0 else 'Sensitive'}")
        ax.set_xlabel("Contribution to Prediction")
        ax.invert_yaxis()
        st.pyplot(fig)

       # 🧬 Feature table
        st.markdown("### 🧬 Feature Values (Explained Instance)")
        explained_instance = X_scaled[idx]  # use the same input you gave to LIME

        table_rows = []
        for f, w in feat_weights:
            feat_name = f.split('<=')[0].strip()
            try:
                val = round(explained_instance[selected_feature_names.tolist().index(feat_name)], 3)
            except:
                val = "-"
            table_rows.append((feat_name, round(w, 3), val))

        table_df = pd.DataFrame(table_rows, columns=["Feature", "Contribution", "Value"])
        st.dataframe(table_df)


        # SHAP Explanation Section
        st.subheader("📊 SHAP Waterfall Explanation (Sample-wise)")

        try:
            # Step 1: Prepare background and test sample
            background = X_scaled[:20].astype(np.float32)
            test_sample = X_scaled[[idx]].astype(np.float32)

            background_tensor = torch.tensor(background).unsqueeze(1)
            test_tensor = torch.tensor(test_sample).unsqueeze(1)

            # Step 2: DeepExplainer
            explainer = shap.DeepExplainer(model, background_tensor)
            shap_values = explainer.shap_values(test_tensor, check_additivity=False)

            # Step 3: Handle multi-class SHAP output
            selected_class = int(torch.argmax(model(test_tensor)).item())
            shap_vals_matrix = np.array(shap_values[selected_class])  # (1, features, 2) or (1, features)
            if shap_vals_matrix.ndim == 3:
                shap_vals = shap_vals_matrix[0][:, selected_class]
            elif shap_vals_matrix.ndim == 2:
                shap_vals = shap_vals_matrix[0]
            else:
                raise ValueError("Unexpected SHAP shape")

            base_value = explainer.expected_value[selected_class]

             # Step 4: Build explanation and plot (clean version)
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=test_sample[0],
                feature_names=selected_feature_names.tolist()
            )

            # Clean matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.grid(False)  # remove grid lines
            for spine in ax.spines.values():  # remove borders
                spine.set_visible(False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP waterfall plot failed: {e}")

          