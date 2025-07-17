"""
Test suite for data/text_dataset.py

Tests the TextDataset class with openwebtext dataset and nnsight GPT-2 tokenizer.
"""

import nnsight
import pytest
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from data.text_dataset import TextDataset, worker_init_fn


class MockDataset:
    """Mock HuggingFace dataset for testing."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, idx):
        return {"text": self.sentences[idx]}

    def __len__(self):
        return len(self.sentences)


@pytest.fixture()
def model():
    """Load GPT-2 model with nnsight for tokenizer."""
    return nnsight.LanguageModel(
        "openai-community/gpt2",
        device_map="cpu",
        dispatch=True,
        torch_dtype=torch.float32,
    )


class TestTextDataset:
    """Test cases for TextDataset class."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load a small subset of openwebtext dataset."""
        # Load small subset for testing
        dataset = load_dataset("Skylion007/openwebtext", split="train")
        return dataset.select(range(100))  # Just 100 samples for testing

    @pytest.fixture
    def text_dataset(self, dataset, model):
        """Create TextDataset instance."""
        return TextDataset(
            hf_dataset=dataset,
            to_tokens=model.tokenizer,
            batch_size=4,
            seq_len=128,
            drop_last_batch=True,
            hf_text_accessor="text",
        )

    def test_init(self, text_dataset, dataset, model):
        """Test TextDataset initialization."""
        assert text_dataset.hf_dataset == dataset
        assert text_dataset.to_tokens == model.tokenizer
        assert text_dataset.batch_size == 4
        assert text_dataset.seq_len == 128
        assert text_dataset.drop_last_batch == True
        assert text_dataset.hf_text_accessor == "text"
        assert text_dataset.token_pointer == 0
        assert text_dataset.batch_pointer == 0
        assert text_dataset.tokens == []

    def test_set_token_pointer(self, text_dataset):
        """Test setting token pointer for multi-process support."""
        text_dataset.set_token_pointer(50)
        assert text_dataset.token_pointer == 50

    def test_iter(self, text_dataset):
        """Test iterator protocol."""
        assert iter(text_dataset) == text_dataset

    def test_next_basic(self, text_dataset):
        """Test basic __next__ functionality."""
        batch, mask = next(text_dataset)

        # Check batch shape and type
        assert batch.shape == (4, 128)
        assert batch.dtype == torch.long

        # Check mask shape and type
        assert mask.shape == (4, 128)
        assert mask.dtype == torch.bool

        # Check that tokens are valid (non-negative)
        assert torch.all(batch >= 0)

        # Check that mask has True values where tokens exist
        assert torch.any(mask)

    def test_multiple_batches(self, text_dataset):
        """Test getting multiple batches."""
        batches = []
        masks = []

        for _ in range(3):
            batch, mask = next(text_dataset)
            batches.append(batch)
            masks.append(mask)

        # Check that we got 3 batches
        assert len(batches) == 3
        assert len(masks) == 3

        # Check that token pointer advanced
        assert text_dataset.token_pointer == 12  # 3 batches * 4 batch_size

    def test_drop_last_batch_behavior(self, dataset, model):
        """Test drop_last_batch behavior."""
        # Create dataset with small batch size to test edge cases
        text_dataset = TextDataset(
            hf_dataset=dataset.select(range(5)),  # 5 samples
            to_tokens=model.tokenizer,
            batch_size=3,  # batch_size=3, so we get 1 full batch + 2 remainder
            seq_len=64,
            drop_last_batch=True,
        )

        # Should get one full batch
        batch, mask = next(text_dataset)
        assert batch.shape == (3, 64)

        # Should get one partial batch
        batch, mask = next(text_dataset)
        assert batch.shape == (3, 64)
        # last row should be all False in mask
        assert not mask[-1].any()

        # Next call should stop iteration (drop_last_batch=True)
        with pytest.raises(StopIteration):
            next(text_dataset)

    def test_no_drop_last_batch(self, dataset, model):
        """Test behavior when drop_last_batch=False."""
        text_dataset = TextDataset(
            hf_dataset=dataset.select(range(5)),  # 5 samples
            to_tokens=model.tokenizer,
            batch_size=3,  # batch_size=3, so we get 1 full batch + 2 remainder
            seq_len=64,
            drop_last_batch=False,
        )

        # First batch should be full
        batch1, mask1 = next(text_dataset)
        assert batch1.shape == (3, 64)

        # Second batch should contain remainder (padded to batch_size)
        batch2, mask2 = next(text_dataset)
        assert batch2.shape == (3, 64)

        # Should stop after exhausting dataset
        with pytest.raises(StopIteration):
            next(text_dataset)

    def test_sequence_length_truncation(self, dataset, model):
        """Test that sequences are truncated to seq_len."""
        text_dataset = TextDataset(
            hf_dataset=dataset,
            to_tokens=model.tokenizer,
            batch_size=2,
            seq_len=32,  # Short sequence length
            drop_last_batch=True,
        )

        batch, mask = next(text_dataset)
        assert batch.shape == (2, 32)
        assert mask.shape == (2, 32)

        # Check that no sequence exceeds seq_len
        for i in range(2):
            # Count actual tokens (non-zero values)
            actual_tokens = (batch[i] != 0).sum().item()
            assert actual_tokens <= 32

    def test_with_dataloader(self, text_dataset):
        """Test TextDataset with PyTorch DataLoader."""
        # Test with single worker
        dataloader = DataLoader(
            text_dataset, batch_size=None, shuffle=False, num_workers=0  # TextDataset handles batching
        )

        batch, mask = next(iter(dataloader))
        assert batch.shape == (4, 128)
        assert mask.shape == (4, 128)

    def test_custom_text_accessor(self, dataset, model):
        """Test custom hf_text_accessor."""
        # Create a modified dataset with different text field name
        modified_dataset = dataset.map(lambda x: {"content": x["text"]})

        text_dataset = TextDataset(
            hf_dataset=modified_dataset,
            to_tokens=model.tokenizer,
            batch_size=2,
            seq_len=64,
            hf_text_accessor="content",  # Use custom accessor
        )

        batch, mask = next(text_dataset)
        assert batch.shape == (2, 64)
        assert mask.shape == (2, 64)

    def test_dataset_exhaustion(self, dataset, model):
        """Test behavior when dataset is exhausted."""
        # Use very small dataset
        small_dataset = dataset.select(range(2))

        text_dataset = TextDataset(
            hf_dataset=small_dataset,
            to_tokens=model.tokenizer,
            batch_size=1,
            seq_len=64,
            drop_last_batch=False,
        )

        # Get first batch
        batch1, mask1 = next(text_dataset)
        assert batch1.shape == (1, 64)

        # Get second batch
        batch2, mask2 = next(text_dataset)
        assert batch2.shape == (1, 64)

        # Should raise StopIteration when exhausted
        with pytest.raises(StopIteration):
            next(text_dataset)


class TestTextDatasetIntegration:
    """Integration tests for TextDataset with real data."""

    def test_real_tokenization(self, model):
        """Test with actual openwebtext data and tokenization."""
        # Load small subset for integration test
        dataset = load_dataset("Skylion007/openwebtext", split="train")
        small_dataset = dataset.select(range(10))

        # Create TextDataset
        text_dataset = TextDataset(
            hf_dataset=small_dataset,
            to_tokens=model.tokenizer,
            batch_size=2,
            seq_len=256,
            drop_last_batch=True,
        )

        # Get batch and verify tokenization worked
        batch, mask = next(text_dataset)

        # Basic checks
        assert batch.shape == (2, 256)
        assert mask.shape == (2, 256)
        assert batch.dtype == torch.long
        assert mask.dtype == torch.bool

        # Check that we have valid tokens
        assert torch.all(batch >= 0)
        assert torch.any(mask)  # Should have some valid tokens

        # Check that tokens are within vocab size
        vocab_size = model.tokenizer.vocab_size
        assert torch.all(batch < vocab_size)

        # Verify we can decode tokens back to text
        for i in range(2):
            # Get valid tokens (where mask is True)
            valid_tokens = batch[i][mask[i]]
            if len(valid_tokens) > 0:
                decoded = model.tokenizer.decode(valid_tokens)
                assert isinstance(decoded, str)
                assert len(decoded) > 0

    def test_custom_english_sentences_roundtrip(self, model):
        """Test with custom English sentences and verify roundtrip tokenization."""
        # Create custom dataset with 2 short English sentences
        test_sentences = [
            "Hello world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        mock_dataset = MockDataset(test_sentences)

        # Create TextDataset with batch_size=2 to get both sentences in one batch
        text_dataset = TextDataset(
            hf_dataset=mock_dataset,
            to_tokens=model.tokenizer,
            batch_size=2,
            seq_len=64,  # Should be enough for our short sentences
            drop_last_batch=False,
            hf_text_accessor="text",
        )

        # Get the batch
        batch, mask = next(text_dataset)

        # Verify batch structure
        assert batch.shape == (2, 64)
        assert mask.shape == (2, 64)
        assert batch.dtype == torch.long
        assert mask.dtype == torch.bool

        # Test roundtrip for each sentence
        for i, original_sentence in enumerate(test_sentences):
            # Get valid tokens for this sentence (where mask is True)
            valid_tokens = batch[i][mask[i]]

            # Decode back to text
            decoded_text = model.tokenizer.decode(valid_tokens)

            # Verify the decoded text matches the original
            # Note: There might be slight differences due to tokenization (spaces, etc.)
            # but the core content should be preserved
            assert isinstance(decoded_text, str)
            assert len(decoded_text) > 0

            # Check that key words from original sentence are in decoded text
            if i == 0:  # First sentence
                assert "Hello" in decoded_text
                assert "world" in decoded_text
                assert "test" in decoded_text
                assert "sentence" in decoded_text
            else:  # Second sentence
                assert "quick" in decoded_text
                assert "brown" in decoded_text
                assert "fox" in decoded_text
                assert "jumps" in decoded_text
                assert "lazy" in decoded_text
                assert "dog" in decoded_text

            print(f"Original: {original_sentence}")
            print(f"Decoded:  {decoded_text}")
            print(f"Tokens:   {valid_tokens.tolist()}")
            print()

        # Additional verification: Check that the tokenization is consistent
        # by manually tokenizing the original sentences
        for i, original_sentence in enumerate(test_sentences):
            manual_tokens = model.tokenizer(original_sentence)["input_ids"]
            batch_tokens = batch[i][mask[i]]

            # The tokens should match (at least the first part)
            min_len = min(len(manual_tokens), len(batch_tokens))
            assert torch.equal(
                torch.tensor(manual_tokens[:min_len]), batch_tokens[:min_len]
            ), f"Token mismatch for sentence {i}: expected {manual_tokens[:min_len]}, got {batch_tokens[:min_len].tolist()}"

    def test_short_sequences_padding_and_mask(self, model):
        """Test that short sequences (< seq_len) are properly padded and masked."""
        # Create very short sentences that will be much shorter than seq_len
        short_sentences = [
            "Hi!",  # Very short
            "Hello world.",  # Short
            "OK",  # Extremely short
            "Yes, no.",  # Short
        ]

        mock_dataset = MockDataset(short_sentences)

        # Create TextDataset with large seq_len to ensure sentences are much shorter
        text_dataset = TextDataset(
            hf_dataset=mock_dataset,
            to_tokens=model.tokenizer,
            batch_size=4,  # Get all sentences in one batch
            seq_len=50,  # Much larger than needed for short sentences
            drop_last_batch=False,
            hf_text_accessor="text",
        )

        # Get the batch
        batch, mask = next(text_dataset)

        # Verify batch structure
        assert batch.shape == (4, 50)
        assert mask.shape == (4, 50)
        assert batch.dtype == torch.long
        assert mask.dtype == torch.bool

        # Test each sentence individually
        for i, original_sentence in enumerate(short_sentences):
            # Manually tokenize to get expected length
            manual_tokens = model.tokenizer(original_sentence)["input_ids"]
            expected_length = len(manual_tokens)

            # Check that the actual tokens match manual tokenization
            actual_tokens = batch[i][:expected_length]
            assert torch.equal(
                actual_tokens, torch.tensor(manual_tokens)
            ), f"Token mismatch for sentence {i}: expected {manual_tokens}, got {actual_tokens.tolist()}"

            # Check mask behavior
            sentence_mask = mask[i]

            # First `expected_length` positions should be True
            assert torch.all(
                sentence_mask[:expected_length]
            ), f"Mask should be True for first {expected_length} positions in sentence {i}"

            # Remaining positions should be False (padding)
            if expected_length < 50:
                assert torch.all(
                    ~sentence_mask[expected_length:]
                ), f"Mask should be False for padding positions after {expected_length} in sentence {i}"

            # Check that padded positions contain zeros
            if expected_length < 50:
                padding_tokens = batch[i][expected_length:]
                assert torch.all(
                    padding_tokens == 0
                ), f"Padding positions should be zero for sentence {i}, got {padding_tokens[:5].tolist()}..."

            # Verify roundtrip with only valid tokens
            valid_tokens = batch[i][mask[i]]
            decoded_text = model.tokenizer.decode(valid_tokens)

            # The decoded text should contain the essence of the original
            print(f"Sentence {i}:")
            print(f"  Original: '{original_sentence}'")
            print(f"  Decoded:  '{decoded_text}'")
            print(f"  Tokens:   {valid_tokens.tolist()}")
            print(f"  Mask:     {sentence_mask[:10].tolist()}... (first 10)")
            print(f"  Length:   {expected_length}/{50}")
            print()

            # Basic checks that decoding worked
            assert isinstance(decoded_text, str)
            assert len(decoded_text) > 0

            # Check that no padding tokens were included in decoding
            assert (
                len(valid_tokens) == expected_length
            ), f"Valid tokens length {len(valid_tokens)} should match expected {expected_length}"

        # Additional verification: ensure no errors are raised
        # and the function handles short sequences gracefully
        try:
            # This should work without any issues
            next(text_dataset)
            assert False, "Should have raised StopIteration since we exhausted the dataset"
        except StopIteration:
            # This is expected behavior
            pass

        print("✓ All short sequences were properly padded and masked")
        print("✓ No errors raised during processing")
        print("✓ Roundtrip tokenization successful for all short sequences")
