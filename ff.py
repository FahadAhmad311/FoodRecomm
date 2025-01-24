import os
import pandas as pd
from transformers import pipeline
import gradio as gr

# Specify the file path
dataset_path = "/content/drive/My Drive/FINALFOODDATASET/FOOD-DATA-GROUP1.csv"

# Check if the file exists
if os.path.exists(dataset_path):
    print("File exists. Proceeding to load the dataset.")
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Display basic information about the dataset
        print("\nDataset Information:")
        print(df.info())  # Column names, data types, and non-null counts

        # Display the first few rows of the dataset
        print("\nDataset Preview (First 5 Rows):")
        print(df.head())

        # Display summary statistics for numeric columns
        print("\nDataset Summary (Numeric Columns):")
        print(df.describe())

        # Step 1: Initialize the Hugging Face pipeline
        qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

        # Step 2: Combine dataset columns into a context string
        # Adjusting for your dataset columns
        df['context'] = df.apply(
            lambda row: f"Food: {row['food']} | Caloric Value: {row['Caloric Value']} | "
                        f"Fat: {row['Fat']} | Protein: {row['Protein']} | "
                        f"Suitable for: {'Diabetic patients' if row['Sugars'] <= 5 else 'Non-diabetic patients'}",
            axis=1
        )

        # Step 3: Define a function for question answering
        def answer_question(question):
            # Combine all rows into one context
            context = " ".join(df['context'].tolist())
            
            # Use the pipeline to answer the question
            result = qa_pipeline(question=question, context=context)
            return result['answer']

        # Step 4: Create Gradio interface
        def gradio_interface(question):
            return answer_question(question)

        interface = gr.Interface(
            fn=gradio_interface,
            inputs=gr.Textbox(label="Ask a question about meals:", placeholder="e.g., I have diabetes. Recommend a meal."),
            outputs=gr.Textbox(label="Answer:"),
            title="Meal Recommendation Chatbot",
            description="Ask the chatbot questions about meals, and it will recommend suitable options based on the dataset.",
        )

        # Launch the Gradio interface
        interface.launch()

    except Exception as e:
        print(f"Error processing the dataset: {e}")
else:
    print(f"File not found at: {dataset_path}")
