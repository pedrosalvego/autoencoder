#!/usr/bin/env python3
# Gerado a partir de SimCLR.ipynb — treino SimCLR (ConvNeXt) com AMP.
# Rode de dentro de ~/autoencoder/ :  nohup python SimCLR.py > treino.log 2>&1 &


# ===== Célula 0 =====
import torch
import glob
import os
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import Image


base_dir = r'/home/psalvego/enoe_organizado'
source_dir = os.path.join(base_dir, 'normais') 

test_high_dir = os.path.join(base_dir, 'high')
test_flood_dir = os.path.join(base_dir, 'flood')

print(f"Carregando TODAS as imagens normais de: {source_dir}")

simclr_transform = T.Compose([
    T.RandomResizedCrop(128, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:

    full_normal_dataset = ImageFolder(root=source_dir, transform=simclr_transform)
    total_count = len(full_normal_dataset)
    
    # Definindo tamanhos: 90% Treino, 10% Validação
    valid_size = int(total_count * 0.10) 
    train_size = total_count - valid_size
    
    # Gerando semente para reprodutibilidade (sempre dividir igual)
    generator = torch.Generator().manual_seed(42)
    
    # A MÁGICA ACONTECE AQUI: Dividindo virtualmente
    train_dataset, valid_dataset = random_split(
        full_normal_dataset, 
        [train_size, valid_size],
        generator=generator
    )

    # Criando os DataLoaders divididos
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    print(f"-> SUCESSO! Divisão realizada:")
    print(f"   - Total Imagens: {total_count}")
    print(f"   - Treino: {len(train_dataset)} imagens")
    print(f"   - Validação: {len(valid_dataset)} imagens")

except FileNotFoundError:
    print(f"-> ERRO CRÍTICO: Pasta '{source_dir}' não encontrada.")
except Exception as e:
    print(f"-> ERRO: {e}")

# --- 4. Carregando Testes de Anomalia (High e Flood) ---
# Estes continuam iguais, pois são para teste final

class FlatDirectoryDataset(Dataset):
    def __init__(self, directory, transform=None):
        # Busca imagens (ajuste as extensões se precisar, ex: *.png)
        self.filepaths = glob.glob(os.path.join(directory, '*.*'))
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Retorna a imagem e um '0' como label genérica
        return image, 0 

# --- 4. Carregando Testes de Anomalia (High e Flood) ---

# Teste High
try:
    # Usando a nossa nova classe em vez do ImageFolder
    test_high_dataset = FlatDirectoryDataset(directory=test_high_dir, transform=simclr_transform)
    
    if len(test_high_dataset) == 0:
        raise ValueError(f"Nenhuma imagem encontrada na pasta: {test_high_dir}")
        
    test_high_loader = DataLoader(test_high_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"-> Teste High carregado: {len(test_high_dataset)} imagens.")
except Exception as e:
    print(f"-> Aviso ou Erro ao carregar High: {e}")

# Teste Flood
try:
    # Usando a nossa nova classe em vez do ImageFolder
    test_flood_dataset = FlatDirectoryDataset(directory=test_flood_dir, transform=simclr_transform)
    
    if len(test_flood_dataset) == 0:
        raise ValueError(f"Nenhuma imagem encontrada na pasta: {test_flood_dir}")
        
    test_flood_loader = DataLoader(test_flood_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"-> Teste Flood carregado: {len(test_flood_dataset)} imagens.")
except Exception as e:
    print(f"-> Aviso ou Erro ao carregar Flood: {e}")


# 2. O Dataset Customizado
class SimCLRDataset(Dataset):
    def __init__(self, dados_base, transform):
        self.dados = dados_base
        self.transform = transform
        # Ferramenta para converter o seu Tensor de volta para Imagem antes de distorcer
        self.to_pil = T.ToPILImage() 

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        item = self.dados[idx]
        
        # 1. Isola a imagem (ignora a label se for uma tupla/lista)
        if isinstance(item, tuple) or isinstance(item, list):
            img_bruta = item[0]
        else:
            img_bruta = item
            
        # 2. Descobre o formato atual e força virar uma Imagem PIL
        if isinstance(img_bruta, torch.Tensor):
            # É aqui que o seu erro estava acontecendo!
            # Convertendo o Tensor de volta para imagem
            img = self.to_pil(img_bruta).convert('RGB')
            
        elif isinstance(img_bruta, str):
            # Se for o caminho do arquivo no PC
            img = Image.open(img_bruta).convert('RGB')
            
        elif isinstance(img_bruta, Image.Image):
            # Se já for uma Imagem PIL
            img = img_bruta.convert('RGB')
            
        else:
            raise TypeError(f"Formato desconhecido recebido: {type(img_bruta)}")
            
        # 3. Agora a mágica acontece com segurança
        visao_A = self.transform(img)
        visao_B = self.transform(img)
        
        return visao_A, visao_B

# 3. Criar o DataLoader (USE O MAIOR BATCH SIZE POSSÍVEL)
# Junte suas fotos normais e de enchente. O SimCLR não precisa saber quem é quem.
meu_dataset = SimCLRDataset(full_normal_dataset, simclr_transform)
loader_treino = DataLoader(meu_dataset, batch_size=256, shuffle=True, drop_last=True)

# ===== Célula 1 =====
import torch.nn as nn
import torchvision.models as models

class ModeloSimCLR(nn.Module):
    def __init__(self, dim_latente_final=128):
        super().__init__()
        # Carrega o ConvNeXt (vamos ignorar a camada de classificação final dele)
        convnext = models.convnext_tiny(pretrained=True)
        self.encoder = nn.Sequential(*list(convnext.children())[:-1]) # Remove a última camada
        
        # Descubra qual é o tamanho da saída do seu ConvNeXt (geralmente 768 no tiny)
        dim_saida_encoder = 768 
        
        # O Projection Head
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_saida_encoder, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, dim_latente_final) # A saída que vai para a NT-Xent
        )

    def forward(self, x):
        h = self.encoder(x) # Representação Rica (Vamos usar isso depois do treino)
        z = self.projector(h) # Projeção (Vamos usar isso SÓ durante o treino)
        return h, z

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = ModeloSimCLR().to(device)

# ===== Célula 2 =====
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    
    # 1. Junta os dois lotes (Visão A e Visão B)
    # Se o batch for 256, 'z' terá 512 imagens
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1) # Normalização essencial para cosseno
    
    # 2. Calcula a similaridade de TODO MUNDO contra TODO MUNDO
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # 3. Ignora a diagonal principal (A imagem comparada com ela mesma)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, -1e4) # -10000 é perfeitamente seguro para o float16
    
    # 4. Encontra quem é o "Gêmeo"
    # O gêmeo da imagem 'k' está na posição 'k + batch_size'
    labels = torch.cat([torch.arange(batch_size) + batch_size, 
                        torch.arange(batch_size)], dim=0).to(z.device)
    
    # 5. Cross Entropy faz o trabalho de puxar o gêmeo (target) e empurrar o resto
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

# ===== Célula 3 =====
from torch.cuda.amp import GradScaler, autocast

usar_amp = torch.cuda.is_available()  # só usa AMP se tiver GPU

optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler(enabled=usar_amp)
epochs = 50
best_loss = float('inf')

print(f"Iniciando Treinamento SimCLR (AMP={'ligado' if usar_amp else 'desligado'})...")
for epoch in range(epochs):
    modelo.train()
    loss_total = 0

    for visao_A, visao_B in loader_treino:
        visao_A, visao_B = visao_A.to(device), visao_B.to(device)

        optimizer.zero_grad()

        with autocast(enabled=usar_amp):
            _, z_i = modelo(visao_A)
            _, z_j = modelo(visao_B)
            loss = nt_xent_loss(z_i, z_j, temperature=0.5)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_total += loss.item()

    avg_loss = loss_total / len(loader_treino)
    print(f"Época [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(modelo.state_dict(), 'melhor_modelo_simclr.pth')
        print("   --> Modelo Salvo (Melhor Loss) 💾")

