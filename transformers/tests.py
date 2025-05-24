import torch
import unittest
from bert import *

class TestBert(unittest.TestCase):
    batch_size = 10
    max_words = 150
    embed_size = 64
    h = 6
    encoder_count = 6
    vocab_size = 1000

    def setUp(self):
        self.input = torch.randint(0, self.vocab_size, (self.batch_size, self.max_words))
        self.mask = torch.randint(0, 2, (self.batch_size, self.max_words))

    def test_attention_output(self):
        q = torch.rand(self.batch_size, self.max_words, self.embed_size)
        k = torch.rand(self.batch_size, self.max_words, self.embed_size)
        v = torch.rand(self.batch_size, self.max_words, self.embed_size)
        attn = Attention()
        out = attn(q, k, v, None)
        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.any(torch.isnan(out)))

        out_masked = attn(q, k, v, self.mask.unsqueeze(1).expand(-1, self.max_words, -1))
        self.assertEqual(out_masked.shape, q.shape)
        self.assertFalse(torch.any(torch.isnan(out_masked)))

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(self.h, self.embed_size)
        x = torch.rand(self.batch_size, self.max_words, self.embed_size)
        out = mha(x, None)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.any(torch.isnan(out)))

        out_masked = mha(x, self.mask.unsqueeze(1).expand(-1, self.max_words, -1))
        self.assertEqual(out_masked.shape, x.shape)
        self.assertFalse(torch.any(torch.isnan(out_masked)))

    def test_feed_forward(self):
        ff = FeedForward(self.embed_size)
        x = torch.rand(self.batch_size, self.max_words, self.embed_size)
        out = ff(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.any(torch.isnan(out)))

    def test_encoder_layer(self):
        enc = EncoderLayer(self.embed_size, self.h)
        x = torch.rand(self.batch_size, self.max_words, self.embed_size)
        out = enc(x, None)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.any(torch.isnan(out)))

        out_masked = enc(x, self.mask.unsqueeze(1).expand(-1, self.max_words, -1))
        self.assertEqual(out_masked.shape, x.shape)
        self.assertFalse(torch.any(torch.isnan(out_masked)))

    def test_bert(self):
        model = Bert(self.vocab_size, self.embed_size, self.max_words, self.h, self.encoder_count)
        out = model(self.input, self.mask)
        self.assertEqual(out.shape, (self.batch_size, self.max_words, self.vocab_size))
        self.assertFalse(torch.any(torch.isnan(out)))

if __name__ == "__main__":
    unittest.main()