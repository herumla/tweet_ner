import os
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  #
import openai.embeddings_utils 
from openai.embeddings_utils import get_embedding, cosine_similarity
import sys
import matplotlib.pyplot as mp

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


#Calculate NER using the GPT 3.5 Turbo Model
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_gpt_ner(tweet):

    output = {}
    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": """You will be provided a tweet about airlines and your job will be to extract the airline(s) mentioned in the tweet. Unless specified otherwise, you must
            return the OUTPUT as a valid python list. All the items in the list should always be wrapped in single quotes.  
                          Some example TWEETs and their corresponding OUTPUT below: 
                        TWEET: United worst airline ever! Staff is nasty, wifi down bags are delayed due to weather?? 
                               And now the belt is broken. Selling UAL stock in AM 
                        OUTPUT:['United Airlines']
                        TWEET: @AmericanAir yes but you did manage to lose two of our bags. Horrendous airline.
                        OUTPUT: ['American Airlines']
                        TWEET: American Air I am running out of battery you have my chargers,clothes,coat,contact lenses, 
                               need to be in mtg tom need help #pathetic service
                        OUTPUT:['American Airlines']
                        TWEET: US Air We did. American Air said to open one with you, too.
                        OUTPUT:['American Airlines', 'US Airways']"""
            },
            {
            "role": "user",
            "content": f"{tweet}"
            }
        ],
        temperature=0

    )

    return response['choices'][0]['message']['content']

#Calculate NER using Fine-Tuned Model
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_fine_tuned_ner(tweet):
    response =  openai.Completion.create(
        model="curie:ft-herumla-2023-07-18-10-17-15",
        prompt = tweet + ' ->',
        temperature=0
    )
    return response['choices'][0]['text'].replace('#','')

# 8 mins
# For every row in the file:
    #Generates Completions via the ChatCompletions Endpoint using GPT 3.5 Turbo
    #Generates embeddings for those completions
    #Calculates the cosine similarity between the GPT Completion Embeddings and Test Data Completion Embeddings
def gpt_completions_file(filename):
    df = pd.read_csv(filename)

    print("Gathering GPT Completions")
    df["gpt_completion_airlines"] = df["tweet"].apply(lambda tweet: generate_gpt_ner(tweet))   
    df.to_csv("airline_test_gpt_results.csv", index=False)

    #Convert embeddings 
    print("Gathering GPT Completion Embeddings")
    df['airlines_embedding'] = df["airlines"].apply(lambda airlines: get_embedding(airlines, engine='text-embedding-ada-002'))
    df['gpt_completion_airlines_embedding'] = df["gpt_completion_airlines"].apply(lambda airlines: get_embedding(airlines, engine='text-embedding-ada-002'))

    #Calculate cosine similarities
    print("Calculating GPT Cosine Similarities")
    df["gpt_completion_cosine_similarities"] = df.apply(lambda row: cosine_similarity(row["airlines_embedding"], row["gpt_completion_airlines_embedding"]), axis=1)

    #Export to a file
    output_file_name = "airline_test_results.csv"
    df.to_csv("airline_test_results.csv", index=False)
    print("GPT Completions Finished...Saved all output to file: airline_test_results.csv ")

    return df



#For every row in the file:
    #Generates Completions via the Completions Endpoint using Fine-Tuned Model
    #Generates embeddings for those completions
    #Calculates the cosine similarity between the Fine-Tuned Completion Embeddings and Test Data Completion Embeddings
def fine_tuned_completions_file(completions_df):
    
    df = completions_df

    #print("Gathering Fine-Tuned Completions")
    df["fine_tuned_completion_airlines"] = df["tweet"].apply(lambda tweet: generate_fine_tuned_ner(tweet))   

    #Convert embeddings 
    print("Gathering Fine-Tuned Completion Embeddings")
    df['fine_tuned_completion_airlines_embedding'] = df["fine_tuned_completion_airlines"].apply(lambda airlines: get_embedding(airlines, engine='text-embedding-ada-002'))
    

    #Calculate cosine similarities
    print("Calculating Fine-Tuned Cosine Similarities")
    df["fine_tuned_completion_cosine_similarities"] = df.apply(lambda row: cosine_similarity(row["airlines_embedding"], row["fine_tuned_completion_airlines_embedding"]), axis=1)

    #Export to a file
    df.to_csv("airline_test_results.csv", index=False)
    print("Fine-Tuned Completions Finished...Saved all output to file: airline_test_results.csv ")


#Compare the Cosine Similarities of GPT Completions vs Fine-Tuned Completions
def plot_similarities():
    df = pd.read_csv("airline_test_results.csv")

    # Plotting the histogram
    mp.hist([df["gpt_completion_cosine_similarities"], df["fine_tuned_completion_cosine_similarities"]],  bins=20, label=['GPT3.5 Turbo', 'Fine-Tuned-Curie'])

    mp.xlabel("cosine_similarity")
    mp.ylabel("count")
    mp.title("Cosine Similarity of Completions")
    mp.legend()
    mp.savefig("Completion_Similarities.png")


def process_file(file_name):
    # Code to process the file
    print(f"Processing file: {file_name}")

def main():
    # Check command-line arguments
    if len(sys.argv) < 3 or sys.argv[1] != '-f':
        print("Please provide a file name with the -f flag")
        return

    # Get the file name from the command line argument
    input_file_name = sys.argv[2]

    # Call the function to process the file
    completions_df = gpt_completions_file(input_file_name)


    fine_tuned_completions_file(completions_df)
    plot_similarities()

if __name__ == '__main__':
    main()