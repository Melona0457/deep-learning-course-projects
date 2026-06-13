import torch

import torch.nn as nn


class PatchEmbed(nn.Module):

    """ Image to Patch Embedding

    """



    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):

        super().__init__()

        num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.img_size = img_size

        self.patch_size = patch_size

        self.num_patches = num_patches



        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        # Conv2d로 패치 단위 임베딩을 한 번에 수행

        # (kernel_size = patch_size, stride = patch_size)

        self.proj = nn.Conv2d(

            in_chans,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size

        )

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################

    def forward(self, x):

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        # x: (B, C, H, W)

        B, C, H, W = x.shape

        # Conv를 통해 (B, embed_dim, H', W') 로 변환

        x = self.proj(x)

        # (B, embed_dim, H', W') -> (B, N, embed_dim)

        x = x.flatten(2).transpose(1, 2)

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################

        return x # output dimension must be: (batch size, number of patches, embed_dim)



class Attention(nn.Module):

    def __init__(self, dim, num_heads=8):

        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5



        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)



    def forward(self, x):

        B, N, C = x.shape

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        # qkv: (B, N, 3C)

        qkv = self.qkv(x)

        # (B, N, 3C) -> (3, B, num_heads, N, head_dim)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]   # 각각 (B, heads, N, head_dim)



        # Scaled dot-product attention

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        attn = attn.softmax(dim=-1)

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################



        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)



        return x # output dimension must be: (batch size, number of patches, embed_dim)



class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):

        super().__init__()

        out_features = out_features or in_features

        hidden_features = hidden_features or in_features



        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################



    def forward(self, x):

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        x = self.fc1(x)

        x = self.act(x)

        x = self.fc2(x)

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################

        return x # output dimension must be: (batch size, number of patches, out_features)



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads)

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,

                       act_layer=act_layer)



    def forward(self, x):

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        # Pre-LN 구조: x + Attention(LN(x)), x + MLP(LN(x))

        x = x + self.attn(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################

        return x



# example

x = torch.randn(2, 3)

x


exam1 = torch.cat((x, x, x), 0)

exam1


exam2 = torch.cat((x, x, x), 1)

exam2


class VisionTransformer(nn.Module):

    """ Vision Transformer """



    def __init__(self, img_size=28, patch_size=4, in_chans=1, num_classes=10, embed_dim=768, depth=12,

                 num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, ):

        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.depth = depth



        self.patch_embed = PatchEmbed(

            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches



        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        # similarly to cls_token, define a learnable positional embedding that matches the patchified input token size.

        # (1, num_patches + 1, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################



        self.blocks = nn.ModuleList([

            Block(

                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,  norm_layer=norm_layer)

            for i in range(depth)])

        self.norm = norm_layer(embed_dim)



        # Classifier head

        self.head = nn.Linear(

            embed_dim, num_classes) if num_classes > 0 else nn.Identity()



    def forward(self, x):

        ##############################################################################

        #                           IMPLEMENT YOUR CODE                              #

        ##############################################################################

        B = x.shape[0]



        # Patch Embedding: (B, C, H, W) -> (B, N, D)

        x = self.patch_embed(x)



        # Concatenate class tokens to patch embedding

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        x = torch.cat((cls_tokens, x), dim=1)          # (B, N+1, D)



        # Add positional embedding to patches

        x = x + self.pos_embed



        # Forward through encoder blocks

        for blk in self.blocks:

            x = blk(x)



        # Layer normalization

        x = self.norm(x)



        # Use class token for classification

        x = x[:, 0]  # (B, D)



        # Classifier head

        x = self.head(x)

        ##############################################################################

        #                              END YOUR CODE                                 #

        ##############################################################################

        return x



import numpy as np



from tqdm import tqdm, trange



import torch

import torch.nn as nn

from torch.optim import Adam

from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader



from torchvision.transforms import ToTensor

from torchvision.datasets.mnist import FashionMNIST


def Train():

    ##############################################################################

    #                           IMPLEMENT YOUR CODE                              #

    ##############################################################################



    patch_size = 4      # 28x28 -> 7x7 = 49 patches

    embed_dim = 256     # 적당한 크기의 임베딩 차원

    depth = 4           # Transformer encoder block 개수

    num_heads = 16       # 256 / 8 = 32 (정수로 나눠떨어짐)

    mlp_ratio = 4.0     # MLP hidden dim = embed_dim * mlp_ratio



    ##############################################################################

    #                              END YOUR CODE                                 #

    ##############################################################################



    # Loading data

    transform = ToTensor()



    train_set = FashionMNIST(root='./data', train=True, download=True, transform=transform)

    test_set = FashionMNIST(root='./data', train=False, download=True, transform=transform)



    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)

    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)



    # Defining model and training options

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")



    model = VisionTransformer(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio).to(device)

    model_path = './vit.pth'

    N_EPOCHS = 10

    LR = 0.001



    # Training loop

    optimizer = Adam(model.parameters(), lr=LR)

    criterion = CrossEntropyLoss()

    for epoch in trange(N_EPOCHS, desc="Training"):

        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            x, y = batch

            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            loss = criterion(y_hat, y)



            train_loss += loss.detach().cpu().item() / len(train_loader)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")



    # Test loop

    with torch.no_grad():

        correct, total = 0, 0

        test_loss = 0.0

        for batch in tqdm(test_loader, desc="Testing"):

            x, y = batch

            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            loss = criterion(y_hat, y)

            test_loss += loss.detach().cpu().item() / len(test_loader)



            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()

            total += len(x)

        print(f"Test loss: {test_loss:.2f}")

        print(f"Test accuracy: {correct / total * 100:.2f}%")



    torch.save(model.state_dict(), model_path)

    print('Saved Trained Model.')



Train()


