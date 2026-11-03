from dataclasses import dataclass


@dataclass
class GenerationParameters:
    """
    Parameters to override in model generation.

    Parameters:
        temperature (float): Temperature in sampling generation. Has no effect when `do_sample` is not set.
            Default: 1.0.
        topk (int): Top-k token predictions to consider in sampling generation. Has no effect when `do_sample` is
            not set. Default: 50.
        topp (float): Only consider the highest unique tokens, which probabilities sum up to `topp`. Has no effect
            when `do_sample` is not set. Default: 1.0.
        do_sample (bool): If true, perform sampling from models probabilities. If false, only generate token with
            maximum probability. Default: False.
        num_beams (int): Number of beams if beam search generation is used. Has no effect when `do_sample` is not
            set. Default: 1.
        repetition_penalty (float): The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no
            penalty. Applied for whitebox models from HuggingFace. Default: 1.0.
    """

    top_k: int = 20
    max_new_tokens: int = 1500
    top_p: float = 0.8
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0 #default, no penalty 
    generate_until: tuple = ()

