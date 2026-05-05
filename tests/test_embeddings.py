from types import SimpleNamespace

from rag.embeddings import EmbeddingConfig, OpenAIEmbedder


class DummyEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, list(input)))
        data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
        return SimpleNamespace(data=data)


class DummyClient:
    def __init__(self):
        self.embeddings = DummyEmbeddingsAPI()


def test_embed_texts_batches_by_10():
    config = EmbeddingConfig(api_key="k", base_url="http://x", model="m")
    embedder = OpenAIEmbedder(config, batch_size=10)
    dummy = DummyClient()
    embedder.client = dummy

    texts = [f"t{i}" for i in range(23)]
    out = embedder.embed_texts(texts)

    assert len(out) == 23
    assert len(dummy.embeddings.calls) == 3
    assert len(dummy.embeddings.calls[0][1]) == 10
    assert len(dummy.embeddings.calls[1][1]) == 10
    assert len(dummy.embeddings.calls[2][1]) == 3
