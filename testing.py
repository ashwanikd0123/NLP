from mytransformers.bert import *

if __name__ == "__main__":
    vocab_size = 100
    embed_size = 62
    max_len = 250
    h = 6
    encoder_count = 6
    model = Bert(vocab_size = vocab_size, embed_size = embed_size, max_len= max_len, h = h, encoder_count = encoder_count)

    batch_size = 256
    x = torch.rand(batch_size, max_len) * vocab_size
    x = x.long()
    mask = torch.rand(batch_size, max_len)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask.int()

    op = model(x, mask)
    print(op.shape)
    print(op)