import json
import os
import random

from bertopic import BERTopic
from bertopic.backend._utils import select_backend
from bertopic.representation import KeyBERTInspired
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
import matplotlib.pyplot as plt
import numpy as np

class TopicModel:
    def __init__(
            self, 
            model, 
            model_name, 
            num_docs,
            data_dir_path,
            sample_frac=None,
            min_topic_frac=0.001,
            n_components=5,
            n_neighbours=15,
            run_dir_name=None,
            num_topics=None
        ):
        self.sample_frac = sample_frac

        if sample_frac:
            min_topic_size = int(num_docs * sample_frac * min_topic_frac)
        else:
            min_topic_size = int(num_docs * min_topic_frac)

        self.model = model

        if not run_dir_name:
            sample_text = str(sample_frac).replace('.', '_') if sample_frac else "all"
            run_dir_name = f'topics_{model_name}_{sample_text}'

        this_embedding_name = f'embeddings_{model_name}.npy'

        run_dir_path = os.path.join(data_dir_path, run_dir_name)
        if not os.path.exists(run_dir_path):
            os.mkdir(run_dir_path)

        self.embedding_path = os.path.join(data_dir_path, this_embedding_name)

        self.run_dir_path = run_dir_path

        gpu_accelerate = False
        low_memory = True

        if gpu_accelerate:
            import cuml

            # Create instances of GPU-accelerated UMAP and HDBSCAN
            umap_model = cuml.manifold.UMAP(
                n_components=n_components, 
                n_neighbors=n_neighbours, 
                min_dist=0.0
            )

            hdbscan_model = cuml.cluster.HDBSCAN(
                min_cluster_size=min_topic_size, 
                min_samples=10, 
                gen_min_span_tree=True, 
                prediction_data=True,
            )
        else:
            import umap
            umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbours,
                min_dist=0.0,
                metric='cosine',
                low_memory=low_memory,
                random_state=42,
                verbose=True
            )
            hdbscan_model = None

        representation_model = KeyBERTInspired()

        self.topic_model = BERTopic(
            embedding_model=model,
            min_topic_size=min_topic_size,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            verbose=True, 
            low_memory=low_memory,
            nr_topics=num_topics
        )

    def change_dir(self, new_dir_path):
        self.run_dir_path = new_dir_path
        self.embedding_path = os.path.join(new_dir_path, os.path.basename(self.embedding_path))

    def get_embeddings(self, docs):
        if not os.path.exists(self.embedding_path):
            self.topic_model.embedding_model = select_backend(self.model,
                                                        language=self.topic_model.language)
            embeddings = self.topic_model._extract_embeddings(docs,
                                                    method="document",
                                                    verbose=self.topic_model.verbose)
            
            with open(self.embedding_path, 'wb') as f:
                np.save(f, embeddings)
        else:
            with open(self.embedding_path, 'rb') as f:
                embeddings = np.load(f)

        return embeddings
    
    def fit_transform(self, docs, embeddings=None, write_to_file=True):
        docs = list(docs)

        if embeddings is None:
            embeddings = self.get_embeddings(docs)

        if self.sample_frac:
            # sample both docs and embeddings
            sample_size = int(self.sample_frac * len(docs))
            sample_indices = random.sample(range(len(docs)), sample_size)
            docs = [docs[i] for i in sample_indices]
            embeddings = embeddings[sample_indices]

        topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)

        if write_to_file:
            if not self.sample_frac:
                with open(os.path.join(self.run_dir_path, 'topics.json'), 'w') as f:
                    json.dump([int(topic) for topic in topics], f)

                np.save(os.path.join(self.run_dir_path, 'probs.npy'), probs)
            self.write_topics_info(docs, topics, probs)
            

    def write_topics_info(self, docs=None, write_hierarchy=True):
        topic_df = self.topic_model.get_topic_info()
        topic_df.to_csv(os.path.join(self.run_dir_path, 'topic_info.csv'), index=False)
        
        # only if there are some topics 
        if write_hierarchy and docs and len(topic_df) > 1:
            hierarchical_topics = self.topic_model.hierarchical_topics(docs)
            hierarchical_topics.to_csv(os.path.join(self.run_dir_path, 'hierarchical_topics.csv'), index=False)

            tree = self.topic_model.get_topic_tree(hierarchical_topics)
            with open(os.path.join(self.run_dir_path, f'cluster_tree.txt'), 'w', encoding='utf-8') as f:
                f.write(tree)

    def transform(self, docs, embeddings=None):
        if embeddings is None:
            embeddings = self.get_embeddings(docs)

        topics, probs = self.topic_model.transform(list(docs), embeddings=embeddings)

        with open(os.path.join(self.run_dir_path, 'topics.json'), 'w') as f:
            json.dump([int(topic) for topic in topics], f)

        np.save(os.path.join(self.run_dir_path, 'probs.npy'), probs)


    def visualize_topics(self, embeddings):

        umap_embeddings = self.topic_model.umap_model.transform(embeddings)

        # Extract predictions and probabilities if it is a HDBSCAN-like model
        predictions, probabilities = hdbscan_delegator(self.topic_model.hdbscan_model, "approximate_predict", umap_embeddings)

        # Map probabilities and predictions
        predictions = self.topic_model._map_predictions(predictions)

        fig, ax = plt.subplots(figsize=(20, 20))
        scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=predictions, cmap="hsv", s=0.1)
        legend = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Topics")
        ax.add_artist(legend)
        fig.savefig(os.path.join(self.run_dir_path, 'embedding_visualization.png'))
        fig.close()

        fig = self.topic_model.visualize_topics()
        fig.write_html(os.path.join(self.run_dir_path, 'topic_visualization.html'))
        fig.write_image(os.path.join(self.run_dir_path, 'topic_visualization.png'))