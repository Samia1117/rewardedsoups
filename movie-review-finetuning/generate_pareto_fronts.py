import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
tqdm.pandas()
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Generator:
    def __init__(self):
        wandb.init()

    def generate_pf(self):

        mean_results = []
        median_results = []

        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        base_gpt2_model_name = "lvwerra/gpt2-imdb"
        tokenizer = AutoTokenizer.from_pretrained(base_gpt2_model_name)

        device = 0 if torch.cuda.is_available() else "cpu" 
        sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
        gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Reward model - RLHF
        reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        rank_model, rlhf_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_model_name), AutoTokenizer.from_pretrained(reward_model_name)


        ### Load IMDB dataset
        def build_dataset(model_name, dataset_name="stanfordnlp/imdb",input_min_text_length=2,input_max_text_length=8,):

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
        
        dataset = build_dataset(base_gpt2_model_name)
        
        for l in lambdas:
            finetuned_model_name = "/Users/samiazaman/Desktop/git-repos/llm/rewardedsoups/movie-review-finetuning/models/gpt2-imdb-pos-deberta-inverse/gpt2-imdb-pos-deberta-inverse-" + str(l)
            finetuned_model = AutoModelForCausalLMWithValueHead.from_pretrained(finetuned_model_name)
            finetuned_model.to(device)
            print("######### Finetuned model name = ", finetuned_model_name[-35:])
        
            ### Model Inspection
            '''
            Generate performance statistics - mean score for two different rewards
            '''
            bs = 200
            game_data = dict()
            dataset.set_format("pandas")
            df_batch = dataset[:].sample(bs)
            game_data["query"] = df_batch["query"].tolist()
            query_tensors = df_batch["input_ids"].tolist()

            response_tensors = []
            output_min_length = 4
            # output_max_length = 16
            output_max_length = 32
            output_length_sampler = LengthSampler(output_min_length, output_max_length)

            #### get response from model tuned using pos and conc weights
            for i in range(bs):
                query = torch.tensor(query_tensors[i]).to(device)

                gen_len = output_length_sampler()
                # Response from model tuned with a mix of weights from rlhf tuned and positiveness tuned models
                query_response = finetuned_model.generate(
                    query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
                ).squeeze()
                response_len = len(query_response) - len(query)
                response_tensors.append(query_response[-response_len:])
        
            #### decode responses
            game_data["response (finetuned)"] = [
                tokenizer.decode(response_tensors[i]) for i in range(bs)
            ]

            #### sentiment analysis of query/response pairs before/after
            sentiment_pipe = pipeline(
                "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
            )

            #### Results of finetuned model
            texts = [q + r for q, r in zip(game_data["query"], game_data["response (finetuned)"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

            # Score 1
            positive_scores = [
                item["score"]
                for output in pipe_outputs
                for item in output
                if item["label"] == "POSITIVE"
            ]

            # Score #2 
            deberta_inverse_rlhf_scores = []  # scores as determined by humans as 'better'
            for q, r in zip(game_data["query"], game_data["response (finetuned)"]):
                inputs = rlhf_tokenizer(q, r, return_tensors='pt')   # tokenizer for rlhf rewarding model
                score = rank_model(**inputs).logits[0].cpu().detach()
                deberta_inverse_rlhf_scores.append(score)

            game_data["positive rewards (finetuned)"] = positive_scores
            game_data["RLHF goodness rewards (finetuned)"] = deberta_inverse_rlhf_scores
            
            # store results in a dataframe
            df_results = pd.DataFrame(game_data)

            print("Mean for model: ...")
            mean_pos_score = df_results["positive rewards (finetuned)"].mean()
            mean_rlhf_goodness_score = df_results["RLHF goodness rewards (finetuned)"].mean()
            print(f'Pos mean: {mean_pos_score}')
            print(f'RLHF Goodness mean: {mean_rlhf_goodness_score}')
            print(type(mean_pos_score))

            tup_mean = ( mean_pos_score.item(), mean_rlhf_goodness_score.item() )
            
            # print()
            # print("Median for model: ...")
            # median_pos_score = df_results["positive rewards (finetuned)"].median()
            # median_rlhf_goodness_score = df_results["RLHF goodness rewards (finetuned)"].median()
            # print(f'Pos median: {median_pos_score}')
            # print(f'RLHF Goodness median: {median_rlhf_goodness_score}')

            # tup_median = (median_pos_score.item(), median_rlhf_goodness_score.item())
            
            # median_results.append(tup_median)
            mean_results.append(tup_mean)
        
        print(f"Mean results for {finetuned_model_name} =  {mean_results}")
        # print(f"Median results for {finetuned_model_name} =  {median_results}")

        file = open("./example-runs/pos-deberta_rlhf-mean-inverse-scores-" + str(l) + ".txt", "w")
        file.write("Mean results = " + str(mean_results))
        # file.write("\n")
        # file.write("Median results = " + str(median_results))
        file.close()

        # x_list = [m[0] for m in median_results]
        # y_list = [m[1] for m in median_results]

        x_list = [m[0] for m in mean_results]
        y_list = [m[1] for m in mean_results]

        for i, lambda_i in enumerate(lambdas):
            print(f'Model = {lambda_i}')
            print(f'Point = {(x_list[i], y_list[i])}')
        
        plt.xlabel('Positiveness Score')
        plt.ylabel('Inverse RLHF Goodness Score')
        plt.scatter(x_list, y_list)
        for i, lambda_i in enumerate(lambdas):
            plt.annotate(lambda_i, (x_list[i], y_list[i]))
        
        plt.savefig("./plots/pos-deberta_rlhf-mean-inverse-score.png")

if __name__ == "__main__":
    model_gen = Generator()  
    model_gen.generate_pf()
