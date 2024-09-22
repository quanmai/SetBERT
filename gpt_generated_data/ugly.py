BOOL_NOT = {
    "system": """You are a helpful assistant who can help us generate a dataset to finetune BERT with boolean NOT. You know A not B could be A and something different from B, say C. Your task is to generate sentences that have the form "A that are not B" (or "A but not"), that's the gold (G) sentence. The positive (P) sentence is A that are C. The negative (N) sentence is A that are B. """,
    "user1": """{"G": "Movies about Vietnam but not about war", "P": ["films that are set in Vietnam or relate to Vietnamese culture", ["films that are set in Vietnam or relate to Vietnamese history, society"], "N": ["Movies set in Vietnam depict how cruel the war", "This movie is about Hanoi aerial war"]}""",
    "assistant": """{"G": "Restaurants serving seafood but not sushi", "P": ["Eateries specializing in fresh catch from the ocean", "Places known for their fish dishes"], "N": ["Sushi bars offering a wide variety of rolls and sashimi", "This restaurant is famous for its traditional Japanese cuisine, especially sushi"]}""",
    "user2": """Now generate using the following template:
    {"G": ***1st Postive sentence***, "P": ***List of 2 positive sentences***, "N": ***List of 2 negative sentences***}
    """

}#, the output is jsonl formatted

BOOL_AND = {
    "system": """You are a helpful assistant who can help us generate a dataset to finetune BERT with boolean AND. You know A and B could be B and A, could not be only B, could not be only A. Your task is to generate sentences that have the form A and B, that's the gold (G) sentence. The positive(P) sentence is "B and A" or their paraphrase. The negative sentence is "A and something different than B", 'only B', 'only A'. """,
    "user1": """{"G": "Novels that combine science fiction and time travel themes", "P": ["Books that explore both time manipulation and futuristic concepts ", "Novels that blend elements of science fiction with time travel"], "N": ["Books that focus on space exploration and alien encounters", "This novel is about science fiction", "This novel is about time travel"]}""",
    "assistant": """{"G": "Restaurants serving seafood and sushi", "P": ["Eateries known for their fresh seafood and sushi offerings", "Places that specialize in both seafood and sushi"], "N": ["Sushi bars offering a wide variety of rolls and sashimi", "This restaurant is famous for its seafood dishes", "This restaurant specializes in sushi"]}""",
    "user2": """Now generate using the following template:
    {"G": ***1st Postive sentence***, "P": ***List of 2 positive sentences, "B and A" or their paraphrase***, "N": ***List of 3 negative sentences, 1 is "A and something different than B", 1 is "only A", 1 is "only B"***}
    """
}   

BOOL_OR = {
    "system": """You are a helpful assistant who can help us generate a dataset to finetune BERT with boolean OR. You know "A or B" could be "only B" and could be "all about A" or their paraphrase. Your task is to generate sentences that have the form of "A or B" or their paraphrases, that's the gold (G) sentence. The positive sentence is either: "all about B", or: "only A". The negative sentence is something different from A and B. """,
    "user1": """{"G": "Movies that feature action or comedy", "P": ["Films that focus on action-packed sequences", "Movies that are known for their comedic elements"], "N": ["This movie is a drama with no action scenes", "This film is a horror movie", "This movie is a romance with no comedic elements"]}""",
    "assistant": """{"G": "Restaurants serving Italian or Mexican cuisine", "P": ["Eateries known for their authentic Italian dishes", "Places that specialize in traditional Mexican food"], "N": ["This restaurant serves American cuisine", "This eatery is known for its Chinese dishes", "This place offers a variety of international dishes"]}""",
    "user2": """Now generate using the following template:
    {"G": ***1st Postive sentence***, "P": ***List of 2 positive sentences, "all about B" or "only A"***, "N": ***List of 3 negative sentences, "something different from A and B"***}
    """
}
