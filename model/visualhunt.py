import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel
import timm
from sentence_transformers import SentenceTransformer, models

class ImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", nn_arch="TRN", pretrained=True, trainable=True, device=None):
        super().__init__()
        self.device = device
        self.nn_arch = nn_arch
        if self.nn_arch == "TRN":
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = timm.create_model(model_name, pretrained, num_classes=0)

        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, image_inputs):
        if self.nn_arch == "CNN":
            image_inputs = image_inputs.to(self.device)
            image_outputs = self.model(image_inputs)
            image_embedding = image_outputs
        else:
            image_inputs = {k:v.to(self.device) for k,v in image_inputs.items()}
            image_outputs = self.model(**image_inputs)
            image_embedding = image_outputs.pooler_output # I am using pooled CLS,
            #image_embedding = reduce(image_outputs.last_hidden_state, 'b p d -> b d', 'mean') # you could use the mean pooling with last_hidden_layers try it out

        return image_embedding

class TextEncoder(nn.Module):

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", trainable=True, device=None):
        super().__init__()
        self.device=device
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'mean')
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, text):
        attr_embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
        return attr_embedding

class VisualHuntNetwork(nn.Module):

    def __init__(self, hyperparms=None):

        super(VisualHuntNetwork, self).__init__()
        self.vision_projection = nn.Linear(hyperparms["vision_dim"], hyperparms["proj_dim"])
        self.text_projection = nn.Linear(hyperparms["text_dim"], hyperparms["proj_dim"])
        self.dropout = nn.Dropout(0.1)

        device = hyperparms["device"]
        nn_arch = hyperparms["nn_arch"]
        if  nn_arch == "TRN":
            model_name = hyperparms["model_name"]
            self.image_encoder = ImageEncoder(model_name, nn_arch, device=device)
        else:
            self.image_encoder = ImageEncoder(device=device)

        self.text_encoder = TextEncoder(device=device)
        EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
        #COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.criterion = nn.TripletMarginWithDistanceLoss(margin=0.1, distance_function=EUCLIDEAN, reduction="mean")
        self.relu_f = nn.ReLU()

    def forward(self, batch):

        # Getting Image and Text Features
        image_emb = self.image_encoder(batch["anchor"])
        image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)

        # Getting Image and Text Embeddings (with same dimension)
        image_emb = self.relu_f(self.vision_projection(image_emb))
        image_emb = self.dropout(image_emb)

        pos_attr_emb = self.text_encoder(batch["pos_attr"])
        pos_attr_emb = torch.nn.functional.normalize(pos_attr_emb, p=2, dim=1)
        pos_attr_emb = self.relu_f(self.text_projection(pos_attr_emb))
        pos_attr_emb = self.dropout(pos_attr_emb)

        neg_attr_emb = self.text_encoder(batch["neg_attr"])
        neg_attr_emb = torch.nn.functional.normalize(neg_attr_emb, p=2, dim=1)
        neg_attr_emb = self.relu_f(self.text_projection(neg_attr_emb))
        neg_attr_emb = self.dropout(neg_attr_emb)

        # Calculating the Loss (Note: Triplet Loss calculation)
        loss = self.criterion(image_emb, pos_attr_emb, neg_attr_emb)      
        return loss
