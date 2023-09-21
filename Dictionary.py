import os
import torch

class Dictionary(object):
    def __init__(self):

        # Get the current directory
        current_dir = os.getcwd()

        # Get the path to the model weights file
        glove_embeddings_path = os.path.join("model", "glove.6B.50d.txt")
        
        # Define label-to-index mapping
        self.label2idx = {
         'O': 0,
         'B-ACTION':1,
         'B-CHANGE':2,
         'B-SCENARIO':3,
         'B-POSSESSION':4,
         'B-SENTIMENT':5,
         'I-SCENARIO':6,
         'I-ACTION':7,
         'I-CHANGE':8,
         'I-SENTIMENT':9,
         'I-POSSESSION':10
          }

        # Load glove embeddings
        self.glove_tokens,self.glove_embeddings = self.process_glove_embeddings(glove_embeddings_path)
        
        # Build word-to-index and index-to-label mappings
        self.word2idx = {word: index for index, word in enumerate(self.glove_tokens)}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def process_glove_embeddings(self, path):
        """
        This function processes the glove embeddings by loading them from a file and adding <unk> and <pad> tokens.
        """
        glove_tokens = []
        glove_embeddings = []

        with open(path, 'rt') as f:
            full_content = f.read().strip().split('\n')
            for i in range(len(full_content)):
                i_word = full_content[i].split(' ')[0]
                i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
                glove_tokens.append(i_word)
                glove_embeddings.append(i_embeddings)

        glove_embeddings = torch.tensor(glove_embeddings)

        # Add <unk> token embedding as the mean of all other embeddings
        glove_tokens.append("<unk>")
        unk_emb = torch.mean(glove_embeddings, dim=0, keepdim=True)
        glove_embeddings = torch.cat((glove_embeddings,unk_emb))

        # Add <pad> token embedding as zeros
        glove_tokens.append("<pad>")
        pad_emb = torch.zeros((1, glove_embeddings.shape[1]))
        glove_embeddings = torch.cat((glove_embeddings,pad_emb))

        return (glove_tokens,glove_embeddings)

    def get_embed_glove(self):
        """
        This function returns the glove embeddings
        """
        return self.glove_embeddings

    def __len__(self):
        """
        This function returns the length of the word-to-index mapping
        """
        return len(self.word2idx)
