import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class ManualAttention(nn.Module):
    def __init__(self, d_model):
        super(ManualAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        d_k = Q.size(-1) #Tamanho do vetor de consulta
        
        valores = torch.matmul(Q, K.transpose(-2, -1))
        valores_escalares = valores / (d_k ** 0.5)
        pesos_finais = F.softmax(valores_escalares, dim=-1)
        representacao_vetorial = torch.matmul(pesos_finais, V)
        
        return representacao_vetorial, pesos_finais

d_model = 16 #tamanho do vetor
seq_len = 10 #quantidade de elementos
input_data = torch.randn(1, seq_len, d_model) #criação dos dados de teste 

model = ManualAttention(d_model)
representacao_vetorial, pesos = model(input_data)

plt.figure(figsize=(10, 8))
sns.heatmap(pesos[0].detach().numpy(), annot=True, cmap='viridis')
plt.title("Heatmap de Pesos")
plt.xlabel("Posição das Chaves")
plt.ylabel("Posição das Consultas")
plt.show()