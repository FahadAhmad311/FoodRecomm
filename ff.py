import os
import pandas as pd
import streamlit as st
from transformers import pipeline

# Define the exact path to the CSV file
csv_file = r"C:\Users\Admin\Downloads\FINALFOODDATASET\FOOD-DATA-GROUP1.csv"

# Check if the file exists
if os.path.exists(csv_file):
    st.write(f"CSV file found at: {csv_file}")
    
    try:
        # Load the CSV file using pandas
        df = pd.read_csv(csv_file)
        
        # Display basic information about the dataset
        st.write("### Dataset Information")
        st.write(df.info())  # Column names, data types, and non-null counts
        
        # Display the first few rows of the dataset
        st.write("### Dataset Preview (First 5 Rows)")
        st.write(df.head())

        # Step 1: Initialize the Hugging Face pipeline
        qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

        # Step 2: Combine dataset columns into a context string
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

        # Step 4: Streamlit interface
        st.title("Meal Recommendation Chatbot")
        st.write("Ask the chatbot questions about meals, and it will recommend suitable options based on the dataset.")
        
        # Input: User's question
        question = st.text_input("Ask a question about meals:", "e.g., I have diabetes. Recommend a meal.")
        
        # Submit button
        submit_button = st.button(label="Submit")

        if submit_button and question:
            # Get the answer to the question
            answer = answer_question(question)
            st.write(f"### Answer: {answer}")

    except Exception as e:
        st.write(f"Error processing the dataset: {e}")
else:
    st.write(f"CSV file not found at: {csv_file}")
