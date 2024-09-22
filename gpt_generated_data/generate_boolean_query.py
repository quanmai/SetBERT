from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
import openai
import os
from ugly import BOOL_OR, BOOL_AND, BOOL_NOT
from openai import OpenAI
import random
import argparse
import multiprocessing
from typing import Dict

def get_openai_key(k = 'OPENAI_API_KEY'):
    return os.environ.get(k)

class OpenAITextGenerator:
    def __init__(
        self, 
        api_key, 
        engine='gpt-3.5-turbo', 
        total_samples=1000, 
        num_workers=20,
        num_samples_per_call=3,
        max_tokens=512,
        output_file='output.jsonl',
        boolean_type='or',
    ):
        self.api_key = api_key
        self.engine = engine
        self.max_tokens = max_tokens
        self.num_samples_per_call = num_samples_per_call
        self.total_samples = total_samples
        self.num_workers = num_workers
        self.output_file = output_file
        self.bool_dict = self.get_temples(boolean_type)

    def get_temples(self, b: str) -> Dict:
        if b == 'or':
            return BOOL_OR
        elif b == 'and':
            return BOOL_AND
        elif b == 'not':
            return BOOL_NOT
        else:
            raise ValueError(f"Invalid boolean type {self.boolean_type}")

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, *args, **kwargs):
        return self.gpt_generate(*args, **kwargs)

    def gpt_generate(self, temperature, top_p) -> str:
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.engine,
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": self.bool_dict["system"]},
                {"role": "user", "content": self.bool_dict['user1']},
                {"role": "assistant", "content": self.bool_dict['assistant']},
                {"role": "user", "content": self.bool_dict['user2']},
            ],
            n=self.num_samples_per_call,
        )
        outputs = [choice.message.content for choice in response.choices]
        return outputs

    def thread_worker(self):
        # x = random.random()
        # return [x for _ in range(self.num_samples_per_call)]
        temperature = random.uniform(0.5, 1)
        top_p = random.uniform(0.8, 1)
        return self.completions_with_backoff(temperature=temperature, top_p=top_p)

    def generate_text_multi_threading(self):
        num_queries_per_worker = self.total_samples // self.num_samples_per_call
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.thread_worker) for _ in range(num_queries_per_worker)]
            for future in as_completed(futures):
                results.append(future.result())
        
        results = [item for par in results for item in par]
        return results

    def write_results_to_file(self, results):
        # write results to jsonlines file
        with open(self.output_file, 'w') as writer:
            for result in results:
                result = result.replace('\n', '')
                writer.write(result)
                writer.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='GPT model', default='gpt-3.5-turbo')
    parser.add_argument('--total-samples', type=int, help='Total samples', default=1000)
    parser.add_argument('--num-workers', type=int, help='Number of workers', default=20)
    parser.add_argument('--boolean-type', type=str, choices=['and', 'or', 'not'], help='Boolean type', default='or')
    parser.add_argument('--output-dir', type=str, help='Output dir', default='./boolean')
    parser.add_argument('--num-per-call', type=int, help='Number of samples per call', default=3)
    parser.add_argument('--max-tokens', type=int, help='Max tokens', default=512)
    parser.add_argument('--num-chunks', type=int, help='Number of chunks', default=10)
    parser.add_argument('--tmp-dir', type=str, help='Temporary directory', default='tmp')
    parser.add_argument('--remove-tmp', action='store_true', help='Remove temporary files', default=False)
    args = parser.parse_args()
    api_key = get_openai_key('OPENAI_API_KEY')
    num_workers = min(args.num_workers, multiprocessing.cpu_count())
    samples_per_chunk = args.total_samples // args.num_chunks
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    for i in range(args.num_chunks):
        # each will generate total // num_chunks samples
        print(f"Generating chunk {i + 1}/{args.num_chunks}...", end='\r')
        generator = OpenAITextGenerator(
            api_key,
            engine=args.engine,
            total_samples=samples_per_chunk,
            num_workers=num_workers,
            num_samples_per_call=args.num_per_call,
            max_tokens=args.max_tokens,
            output_file=os.path.join(args.tmp_dir, f"boolean_{args.boolean_type}_{i}.jsonl"),
            boolean_type=args.boolean_type,
        )
        results = generator.generate_text_multi_threading()
        generator.write_results_to_file(results)
    output_file = os.path.join(args.output_dir, args.boolean_type + '.jsonl') 
    with open(output_file, 'w') as writer:
        for i in range(args.num_chunks):
            with open(os.path.join(args.tmp_dir, f"boolean_{args.boolean_type}_{i}.jsonl"), 'r') as reader:
                for line in reader:
                    writer.write(line)
    if args.remove_tmp:
        import shutil
        shutil.rmtree(args.tmp_dir)

    