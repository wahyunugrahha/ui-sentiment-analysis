import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st

LABEL_MAP = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😢"}

class PlottingTools:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(LABEL_MAP.values()),
            yticklabels=list(LABEL_MAP.values())
        )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(plt)
        plt.clf()

    @staticmethod
    def plot_sentiment_distribution(df):
        sentiment_counts = df['predicted_label'].value_counts()
        plt.figure(figsize=(8, 4))
        sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()
