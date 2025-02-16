import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

class Generate:
    # set review type
    def __init__(self, review_type):
        wandb.init()
        self.review_type = review_type

    def run(self):

        # Change based on positive/negative/neutral movie review type
        model_name_to_save = "gpt2-imdb-pos-v2"
        base_model_name = "carolinezhang/gpt2-imdb-pos-v2"
        if self.review_type == "negative":
            model_name_to_save = "gpt2-imdb-neg-v2"
            base_model_name = "Samzy17/gpt2-imdb-movie-reviews-negative"
        elif self.review_type == "neutral":
            model_name_to_save = "gpt2-imdb-neutral-v2"
            base_model_name = "Samzy17/gpt2-imdb-pos-neg-averaged"

        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("lvwerra/gpt2-imdb")
        tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
        finetuned_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
        
        device = 0 if torch.cuda.is_available() else "cpu" 
        sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
        gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        ### Load IMDB dataset
        '''The IMDB dataset contains 50k movie review annotated with "positive"/"negative" feedback indicating the sentiment.  
        We load the IMDB dataset into a DataFrame and filter for comments that are at least 200 characters. 
        Then we tokenize each text and cut it to random size with the `LengthSampler`. '''

        def build_dataset(model_name, dataset_name="stanfordnlp/imdb",input_min_text_length=2,input_max_text_length=8,):
            """
            Build dataset for training. This builds the dataset from `load_dataset`, one should
            customize this function to train the model on its own dataset.

            Args:
                dataset_name (`str`):
                    The name of the dataset to be loaded.

            Returns:
                dataloader (`torch.utils.data.DataLoader`):
                    The dataloader for the dataset.
            """
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            # load imdb with datasets
            ds = load_dataset(dataset_name, split="train")
            ds = ds.rename_columns({"text": "review"})
            ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

            input_size = LengthSampler(input_min_text_length, input_max_text_length)

            def tokenize(sample):
                sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
                sample["query"] = tokenizer.decode(sample["input_ids"])
                return sample

            ds = ds.map(tokenize, batched=False)
            ds.set_format(type="torch")
            return ds

        dataset = build_dataset("lvwerra/gpt2-imdb")

        ### Model Inspection
        '''
        Let's inspect some examples from the IMDB dataset. 
        We can use ref_model to compare the finetuned model against the ref model before optimisation.
        '''
        #### get a batch from the dataset
        bs = 16
        game_data = dict()
        dataset.set_format("pandas")
        df_batch = dataset[:].sample(bs)
        game_data["query"] = df_batch["query"].tolist()
        print("########################")
        print(f"Queries for both ref/finteuned model = {game_data['query']}")
        print(f"Queries length = {len(game_data['query'])}")
        print("########################")
        query_tensors = df_batch["input_ids"].tolist()

        response_tensors_ref, response_tensors = [], []

        output_min_length = 4
        output_max_length = 16
        output_length_sampler = LengthSampler(output_min_length, output_max_length)

        #### get response from gpt2 and gpt2_ref
        for i in range(bs):
            query = torch.tensor(query_tensors[i]).to(device)

            gen_len = output_length_sampler()
            query_response = ref_model.generate(
                query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
            ).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors_ref.append(query_response[-response_len:])

            query_response = finetuned_model.generate(
                query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
            ).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])

        #### decode responses (ref model)
        game_data["response (before)"] = [
            tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
        ]
        #### decode responses (finetuned model)
        game_data["response (after)"] = [
            tokenizer.decode(response_tensors[i]) for i in range(bs)
        ]

        print("########################")
        print(f"game_data for REF mode = {game_data['response (after)']}")
        print(f"Length = {len(game_data['response (after)'])}")
        print("")
        print("########################")
        print("")
        print(f"game_data for FINETUNED mode = {game_data['response (before)']}")
        print(f"Length = {len(game_data['response (before)'])}")
        print("########################")

        queries = game_data['query']
        ref_responses = game_data['response (before)']
        finetuned_responses = game_data['response (after)']
        for i in range(bs):
            print("########################")
            print(f"Query = {queries[i]}")
            print(f"Reference response = {ref_responses[i]}")
            print(f"Finetuned (avg) response = {finetuned_responses[i]}")
            print("########################")

        #### sentiment analysis of query/response pairs before/after
        sentiment_pipe = pipeline(
            "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
        )

        # ##### Results of reference model
        texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]

        negative_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "NEGATIVE"
        ]
        game_data["positive rewards (before)"] = positive_scores
        game_data["negative rewards (before)"] = negative_scores

        # ##### Results of finetuned model
        texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]

        negative_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "NEGATIVE"
        ]

        game_data["positive rewards (after)"] = positive_scores
        game_data["negative rewards (after)"] = negative_scores

        # store results in a dataframe
        df_results = pd.DataFrame(game_data)
        df_results

        print("mean:")
        print(df_results[["positive rewards (before)", "positive rewards (after)"]].mean())
        print(df_results[["negative rewards (before)", "negative rewards (after)"]].mean())
        print()
        print("median:")
        print(df_results[["positive rewards (before)", "positive rewards (after)"]].median())
        print(df_results[["negative rewards (before)", "negative rewards (after)"]].median())

if __name__ == "__main__":

    ft_gpt2 = Generate("neutral")  # positive, negative, neutral
    ft_gpt2.run()
