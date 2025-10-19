## 🌐 Section 16: Gradio Web Application (Corrected & Improved)

import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Ensure the required variables are defined from previous cells:
# - final_best_model
# - tfidf_vectorizer
# - preprocessor
# - final_best_model_name
# - results_df

def create_gradio_app(model, vectorizer, preprocessor_obj, model_name, metrics_df):
    """
    Creates and launches a Gradio web application for sentiment analysis.

    Args:
        model: The trained machine learning model.
        vectorizer: The fitted TF-IDF vectorizer.
        preprocessor_obj: The text preprocessor instance.
        model_name (str): The name of the best model.
        metrics_df (pd.DataFrame): DataFrame with model performance metrics.
    """

    def gradio_predict(review_text):
        """
        Prediction function for the Gradio interface. It uses the model
        and preprocessors passed to the outer function.
        """
        if not review_text.strip():
            return "⚠️ Please enter a review!", "", "", "", ""

        # Use the existing predict_sentiment function, passing all required arguments
        result = predict_sentiment(
            review=review_text,
            model=model,
            vectorizer=vectorizer,
            preprocessor=preprocessor_obj
        )

        sentiment = result['sentiment']
        confidence = f"{result['confidence']:.2%}" if result['confidence'] is not None else "N/A"
        prob_neg = f"{result['probability_negative']:.2%}" if result['probability_negative'] is not None else "N/A"
        prob_pos = f"{result['probability_positive']:.2%}" if result['probability_positive'] is not None else "N/A"
        cleaned = result['cleaned_review']

        return sentiment, confidence, prob_neg, prob_pos, cleaned

    # Create the Gradio interface
    with gr.Blocks(theme=gr.themes.Soft(), title="Restaurant Review Sentiment Analyzer") as demo:
        gr.Markdown(f"""
        # 🍽️ Restaurant Review Sentiment Analyzer
        ### Powered by Machine Learning

        Enter a restaurant review to analyze its sentiment. The model will predict whether
        the review is **Positive** or **Negative**.

        **Model:** {model_name}
        **Accuracy:** {metrics_df.iloc[0]['Test_Accuracy']:.2%}
        """)

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Enter Restaurant Review",
                    placeholder="e.g., The food was absolutely amazing! Best restaurant ever!",
                    lines=5
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([input_text], value="🗑️ Clear", size="lg")

            with gr.Column(scale=2):
                sentiment_output = gr.Textbox(label="🎯 Predicted Sentiment", interactive=False)
                confidence_output = gr.Textbox(label="📊 Confidence Score", interactive=False)
                with gr.Row():
                    neg_prob = gr.Textbox(label="😞 Negative Probability", interactive=False)
                    pos_prob = gr.Textbox(label="😊 Positive Probability", interactive=False)

        with gr.Accordion("🔍 Preprocessing Details", open=False):
            cleaned_output = gr.Textbox(label="Cleaned Review Text", interactive=False, lines=3)

        gr.Examples(
            examples=[
                ["The food was absolutely amazing! Best restaurant I've ever been to!"],
                ["Terrible service and the food was cold. Never coming back."],
                ["Outstanding experience! The staff was friendly and attentive."],
                ["Worst meal I've ever had. Complete waste of money."],
            ],
            inputs=input_text,
            label="Click to try an example"
        )

        # Connect the button to the prediction function
        submit_btn.click(
            fn=gradio_predict,
            inputs=input_text,
            outputs=[sentiment_output, confidence_output, neg_prob, pos_prob, cleaned_output]
        )

    return demo

# --- LAUNCH THE APP ---
print('='*70)
print('🚀 LAUNCHING GRADIO WEB APPLICATION')
print('='*70)

try:
    # Get the final best model and preprocessors from your notebook
    # Ensure these variable names match what you have in your notebook
    final_model = trained_models[final_best_model_name]
    
    # Create the app instance by passing the required objects
    gradio_app = create_gradio_app(
        model=final_model,
        vectorizer=tfidf_vectorizer,
        preprocessor_obj=preprocessor,
        model_name=final_best_model_name,
        metrics_df=results_df
    )

    # Launch the Gradio app
    # share=True creates a public link and keeps the server running
    gradio_app.launch(share=True, debug=True)
    
    print("\n✅ Gradio app is running!")
    print("📱 Access the app at the public URL provided above.")

except NameError as e:
    print(f"❌ NameError: {e}")
    print("Please make sure all previous cells in the notebook have been run successfully.")
    print("Required variables: 'final_best_model_name', 'trained_models', 'tfidf_vectorizer', 'preprocessor', 'results_df'")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")

