# SetBERT
Data and implementation for the SetBERT paper: https://arxiv.org/abs/2406.17282

📦 SetBERT/

    ┣ 📂 gpt_generated_data/ # GPT prompts to generate boolean data for finetuning 

    ┣ 📂 pretrain/ # Finetune SetBERT on generated data 

    ┣ 📂 dpr/ # Train dense dual encoders for retrieval task 

    ┗ 📂 quest/ # Dataset for retrieval task


## Citation
If you find this repository useful in your research, please consider citing our paper:

@misc{mai2024setbert,
    title={SetBERT: Enhancing Retrieval Performance for Boolean Logic and Set Operation Queries},
    author={Quan Mai and Susan Gauch and Douglas Adams},
    year={2024},
    eprint={2406.17282},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}