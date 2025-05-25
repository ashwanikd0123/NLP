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

class TestBertEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16
        self.embed_size = 64
        self.h = 4
        self.encoder_count = 3
        self.model = Bert(embed_size=self.embed_size, h=self.h, encoder_count=self.encoder_count)

    def test_output_shape(self):
        x = torch.rand(self.batch_size, self.seq_len, self.embed_size)
        out = self.model(x)
        self.assertEqual(out.shape, x.shape)

    def test_no_nan_in_output(self):
        x = torch.rand(self.batch_size, self.seq_len, self.embed_size)
        out = self.model(x)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaNs")

    def test_masked_input_shape_and_nan(self):
        x = torch.rand(self.batch_size, self.seq_len, self.embed_size)
        mask = torch.randint(0, 2, (self.batch_size, self.seq_len))
        out = self.model(x, mask)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any(), "Masked output contains NaNs")

    def test_varying_sequence_lengths(self):
        for seq_len in [8, 32, 64]:
            x = torch.rand(self.batch_size, seq_len, self.embed_size)
            out = self.model(x)
            self.assertEqual(out.shape, x.shape)
            self.assertFalse(torch.isnan(out).any(), f"Output contains NaNs at seq_len={seq_len}")

    def test_gradients_flow(self):
        x = torch.rand(self.batch_size, self.seq_len, self.embed_size, requires_grad=True)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any(), "Gradients contain NaNs")


if __name__ == "__main__":
    unittest.main()