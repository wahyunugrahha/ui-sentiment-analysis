import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from sentimentModel import SentimentModel
from plottingTools import PlottingTools

LABEL_MAP = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😢"}
LABEL_TO_NUM = {"positive": 0, "neutral": 1, "negative": 2}


class SentimentAnalysisApp:
    def __init__(self):
        st.title("🌟 Sentiment Analysis IKN with IndoBERT 🌟")
        st.markdown("Made with 💙 by Wahyu Nugraha")
        self.learning_rate_options = {
            "5e-5 Epoch 10": ("5e-5", 10),
            "5e-7 Epoch 10": ("5e-7", 10),
            "5e-5 Epoch 4": ("5e-5", 4),
            "5e-7 Epoch 4": ("5e-7", 4),
        }
        self.model = None

    def setup_model(self, learning_rate, epoch):
        """Initialize the sentiment model with the selected learning rate and epoch."""
        self.model = SentimentModel(learning_rate, epoch)

    def preprocces_file(self, uploaded_file):
        """Preprocess uploaded file."""
        file_extension = uploaded_file.name.split('.')[-1]
        separator = '\t' if file_extension == 'tsv' else ',' if file_extension == 'csv' else r'\t|,'
        df = pd.read_csv(uploaded_file, delimiter=separator, engine='python', header=None, names=['full_text', 'label'])
        df['label'] = df['label'].str.lower()
        return df

    def handle_file_upload(self):
        """Handle file upload and process it for evaluation."""
        uploaded_file = st.file_uploader("Upload a file (txt, csv, tsv):", type=["csv", "tsv", "txt"])
        if uploaded_file and self.model:
            try:
                df = self.preprocces_file(uploaded_file)  # Call the internal method for preprocessing
                st.write("📄 **Data Preview:**", df.head())

                # Check for invalid labels
                invalid_labels = df[~df['label'].isin(LABEL_TO_NUM.keys())]
                if not invalid_labels.empty:
                    st.error("🚨 Invalid labels detected!")
                    st.write("**Invalid Labels:**")
                    st.write(invalid_labels[['full_text', 'label']])
                    return

                if st.button("Evaluate Model"):
                    valid_data = df[df['label'].isin(LABEL_TO_NUM.keys())]
                    valid_data['predicted_sentiment'] = valid_data['full_text'].apply(
                        lambda x: self.model.predict_sentiment(x)[0]
                    )
                    valid_data['predicted_label'] = valid_data['predicted_sentiment'].map(LABEL_MAP)

                    # Display predicted data
                    st.write("✅ **Predicted Data:**", valid_data[['full_text', 'predicted_label']])
                    PlottingTools.plot_sentiment_distribution(valid_data)

                    # Calculate evaluation metrics
                    valid_data['true_label'] = valid_data['label'].map(LABEL_TO_NUM)
                    y_true = valid_data['true_label'].astype(int)
                    y_pred = valid_data['predicted_sentiment'].astype(int)

                    st.write("📊 **Classification Report:**")
                    st.text(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values())))
                    st.write("📉 **Confusion Matrix:**")
                    PlottingTools.plot_confusion_matrix(y_true, y_pred)

            except Exception as e:
                st.error(f"🚨 Error processing file: {e}")

    def handle_manual_input(self):
        """Handle manual text input and perform prediction."""
        input_text = st.text_area("Enter text here:")
        if st.button("Predict 🔍") and input_text and self.model:
            prediction, confidence = self.model.predict_sentiment(input_text)
            st.write(f"**Predicted Sentiment:** {LABEL_MAP[prediction]}")
            st.write(f"**Model Confidence (Accuracy):** {confidence * 100:.2f}%")

    def run(self):
        """Main entry point for the app."""
        input_choice = st.radio("Choose Input Type:", ("File Upload", "Manual Input"))
        with st.expander("Model Configuration"):
            selected_option = st.selectbox("Select Model Configuration", list(self.learning_rate_options.keys()))
            learning_rate, epoch = self.learning_rate_options[selected_option]
            st.write(f"Batch Size: 32")
            st.write(f"Learning Rate: {learning_rate}")
            st.write(f"Epoch: {epoch}")

        self.setup_model(learning_rate, epoch)

        if input_choice == "File Upload":
            self.handle_file_upload()
        elif input_choice == "Manual Input":
            self.handle_manual_input()


if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.run()
