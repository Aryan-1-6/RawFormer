"""
RawFormer — Text Generation Script
Generates text autoregressively from a prompt.

Run: python generate.py --prompt "the stock market" --max_len 30
"""

import argparse
import cupy as np
from nltk.tokenize import word_tokenize

from config     import CONTEXT, CHECKPOINT_DIR, CHECKPOINT_NAME, TRAIN_PATH
from rawformer  import Decoder
from checkpoint import load_model
from data.dataloader import load_ptb, flatten


def generate(model, prompt, max_len=30, context=CONTEXT):
    """
    Autoregressively generate tokens from a prompt string.

    Args:
        model   : trained Decoder instance
        prompt  : seed string (e.g. "the stock market")
        max_len : maximum number of new tokens to generate
        context : model's context window length

    Returns:
        generated string
    """
    tokens = ['sos'] + word_tokenize(prompt.lower())
    unk_id = model.vocab.get('<unk>', 0)

    for _ in range(max_len):
        # Use only the last `context` tokens to stay within window
        window = tokens[-context:]

        input_ids = np.array(
            [[model.vocab.get(t, unk_id) for t in window]]
        )  # (1, T)
        print(input_ids)

        logits = model.forward(input_ids)   # (1, T, vocab_size)

        # Greedy: pick highest-probability token at the last position
        next_token_id = int(np.argmax(logits[0, -1, :]))

        # ID → word
        id_to_word = {v: k for k, v in model.vocab.items()}
        next_word  = id_to_word.get(next_token_id, '<unk>')

        if next_word == 'eos':
            break

        tokens.append(next_word)

    # Strip the leading 'sos' and return
    return ' '.join(tokens[1:])


def main():
    parser = argparse.ArgumentParser(description='RawFormer text generation')
    parser.add_argument('--prompt',   type=str, default='the company said',
                        help='Seed text to continue')
    parser.add_argument('--max_len',  type=int, default=30,
                        help='Max new tokens to generate')
    parser.add_argument('--checkpoint', type=str,
                        default=f"{CHECKPOINT_DIR}/best_{CHECKPOINT_NAME}",
                        help='Path to saved model checkpoint')
    args = parser.parse_args()

    # Rebuild model shell
    train_stream = flatten(load_ptb(TRAIN_PATH))[:120_000]
    model = Decoder(
        corpus     = [train_stream],
        n_heads    = 4,
        num_layers = 4,
        embd_dim   = 256,
        context    = CONTEXT,
        tokenise   = False,
    )
    model = load_model(args.checkpoint, model)

    print(f"\nPrompt : {args.prompt}")
    result = generate(model, args.prompt, max_len=args.max_len)
    print(f"Output : {result}\n")


if __name__ == "__main__":
    main()