# load_ext autoreload
# autoreload 2

import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

class FineTuneGPT2:
    #setting review type
    def __init__(self):
        wandb.init()

    def run(self):
        # Config
        config = PPOConfig(
            model_name="lvwerra/gpt2-imdb",
            learning_rate=1.41e-5,
            log_with="wandb",)

        # Change based on positive/negative/neutral movie review type
        
        # model_name_to_save = "gpt2-imdb-0.5pos-0.5neg"
        model_name_to_save = "gpt2-imdb-0.1pos-0.9neg"
        print(f'Training model = {model_name_to_save}')

        sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

        ### Load IMDB dataset
        '''The IMDB dataset contains 50k movie review annotated with "positive"/"negative" feedback indicating the sentiment.  
        We load the IMDB dataset into a DataFrame and filter for comments that are at least 200 characters. 
        Then we tokenize each text and cut it to random size with the `LengthSampler`. '''

        def build_dataset(config,dataset_name="stanfordnlp/imdb",input_min_text_length=2,input_max_text_length=8,):
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
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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

        dataset = build_dataset(config)


        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        ### Load pre-trained GPT2 language models
        '''
        We load the GPT2 model with a value head and the tokenizer. We load the model twice; 
        the first model is optimized while the second model serves as a reference to calculate the KL-divergence from the 
        starting point. This serves as an additional reward signal in the PPO training to make sure the optimized model 
        does not deviate too much from the original language model.
        '''
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        tokenizer.pad_token = tokenizer.eos_token

        ppo_trainer = PPOTrainer(
            config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
        )

        ### Load BERT classifier
        # sentiment analysis to determine pos/neg
        device = ppo_trainer.accelerator.device
        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

        sentiment_pipe = pipeline(
            "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
        )

        ### Generation Settings
        ''' 
        For the response generation we just use sampling and make sure top-k and nucleus sampling are turned off as well as 
        a minimal length.
        '''

        gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        ### Optimize model
        '''
        The training loop consists of the following main steps:

        Get the query responses from the policy network (GPT-2)
        Get sentiments for query/responses from BERT
        Optimize policy with PPO using the (query, response, reward) triplet
        Training time

        This step takes ~2h on a V100 GPU with the above specified settings.
        '''
        output_min_length = 4
        output_max_length = 16
        output_length_sampler = LengthSampler(output_min_length, output_max_length)

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]

            #### Get response from gpt2
            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
                response_len = len(query_response) - len(query)
                response_tensors.append(query_response[-response_len:])
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            
            # TODO: Change this for negative/neutral review based fine tuning
            positive_scores = [
                item["score"]
                for output in pipe_outputs
                for item in output
                if item["label"] == "POSITIVE"
            ]

            ## Define rewards below
            # rewards = [torch.tensor(score) for score in positive_scores]

            negative_scores = [
                item["score"]
                for output in pipe_outputs
                for item in output
                if item["label"] == "NEGATIVE"
            ]

            rewards = []
            for i in range(len(negative_scores)):
                score = (0.1 * positive_scores[i]) + (0.9 * negative_scores[i])
                rewards.append(torch.tensor(score))

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        ### Model Inspection
        '''
        Let's inspect some examples from the IMDB dataset. 
        We can use ref_model to compare the tuned model model against the model before optimisation.
        '''
        #### get a batch from the dataset
        bs = 16
        game_data = dict()
        dataset.set_format("pandas")
        df_batch = dataset[:].sample(bs)
        game_data["query"] = df_batch["query"].tolist()
        query_tensors = df_batch["input_ids"].tolist()

        response_tensors_ref, response_tensors = [], []

        #### get response from gpt2 and gpt2_ref
        for i in range(bs):
            query = torch.tensor(query_tensors[i]).to(device)

            gen_len = output_length_sampler()
            query_response = ref_model.generate(
                query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
            ).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors_ref.append(query_response[-response_len:])

            query_response = model.generate(
                query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
            ).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])

        #### decode responses
        game_data["response (before)"] = [
            tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
        ]
        game_data["response (after)"] = [
            tokenizer.decode(response_tensors[i]) for i in range(bs)
]

        #### sentiment analysis of query/response pairs before/after
        texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        # TODO: Change this for negative/neutral review based fine tuning
        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]
        game_data["rewards (before)"] = positive_scores

        texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        # TODO: Change this for negative/neutral review based fine tuning
        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]
        game_data["rewards (after)"] = positive_scores

        # store results in a dataframe
        df_results = pd.DataFrame(game_data) 
        df_results

        print("mean:")
        print(df_results[["rewards (before)", "rewards (after)"]].mean())
        print()
        print("median:")
        print(df_results[["rewards (before)", "rewards (after)"]].median())

        model.save_pretrained(model_name_to_save)
        tokenizer.save_pretrained(model_name_to_save)

if __name__ == "__main__":

    ft_gpt2 = FineTuneGPT2()
    ft_gpt2.run()