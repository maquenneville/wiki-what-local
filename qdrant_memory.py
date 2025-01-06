import multiprocessing
from qdrant_client import QdrantClient, models


class QdrantMemory:

    def __init__(self, args):
        """
        Initialize Qdrant memory with optional collection name and configuration.

        Args:
            collection_name (str, optional): Name of the Qdrant collection.
                                             If not provided, read from 'config.ini'.
        """
        self.collection_name = args.collection
        self.storage_path = args.storage_path

        self.num_threads = multiprocessing.cpu_count()

        if self.storage_path:
            print(f"Using persistent Qdrant storage at: {self.storage_path}")
            self.client = QdrantClient(path=self.storage_path)
        else:
            print("Using in-memory Qdrant storage")
            self.client = QdrantClient(":memory:")

        self.client.set_model(
            embedding_model_name="BAAI/bge-base-en", threads=self.num_threads
        )

        self._create_collection()

    def _create_collection(self):

        fastembed_params = self.client.get_fastembed_vector_params()
        vector_name, vector_config = list(fastembed_params.items())[0]

        if self.client.collection_exists(self.collection_name):
            existing_collection = self.client.get_collection(self.collection_name)
            existing_vectors = existing_collection.config.params.vectors

            if (
                vector_name not in existing_vectors
                or existing_vectors[vector_name].size != vector_config.size
                or existing_vectors[vector_name].distance != vector_config.distance
            ):
                print(
                    f"Existing collection '{self.collection_name}' is incompatible. "
                    "Recreating with correct configuration..."
                )
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Using existing compatible collection: {self.collection_name}")
                return

        if not self.client.collection_exists(self.collection_name):
            cpu_cores = multiprocessing.cpu_count()
            vector_params = fastembed_params

            print(
                f"Creating collection '{self.collection_name}' with {cpu_cores} segments."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    )
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=cpu_cores
                ),
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def store_single(self, document: str, metadata: dict = None, doc_id: str = None):

        self.client.add(
            collection_name=self.collection_name,
            documents=[document],
            metadata=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None,
            parallel=self.num_threads,
        )
        print(f"Stored document with ID: {doc_id or 'auto-generated'}")

    def store(self, documents: list, metadatas: list = None, ids: list = None):

        self.client.add(
            collection_name=self.collection_name,
            documents=documents,
            metadata=metadatas,
            ids=ids,
            parallel=self.num_threads,
        )
        print(f"Stored {len(documents)} documents in '{self.collection_name}'.")

    def fetch_context(self, query: str, top_n: int = 2) -> list:

        results = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=top_n,
        )
        return [res.document for res in results]

    def __str__(self) -> str:
        return f"QdrantMemory(collection='{self.collection_name}')"
