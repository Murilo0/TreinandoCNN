# TreinandoCNN
# 📄 Documentação do Projeto de Classificação de Dígitos Manuscritos com CNN

Este projeto utiliza redes neurais convolucionais (CNNs) com TensorFlow/Keras para classificar imagens de dígitos manuscritos. A base de dados utilizada está no arquivo `digitosmanuscritos.csv`.

---

## 📦 1. Carregamento e Pré-processamento dos Dados

```python
data = pd.read_csv("digitosmanuscritos.csv")
X = data.iloc[:, 1:].values.astype('float32')
y = data.iloc[:, 0].values
X /= 255.0
X = X.reshape(-1, 28, 28, 1)
y = tf.keras.utils.to_categorical(y, 10)
```

### Explicação:

- **Leitura do CSV**: O arquivo `digitosmanuscritos.csv` é lido usando `pandas`.
- **Separação de atributos e rótulos**:
  - `X`: características das imagens (pixels), colunas de 1 em diante.
  - `y`: rótulos (dígitos), primeira coluna.
- **Normalização**: os valores dos pixels são normalizados para o intervalo `[0, 1]`.
- **Redimensionamento**: as imagens são transformadas para o formato `[amostras, 28, 28, 1]` (formato adequado para CNNs).
- **One-hot encoding**: os rótulos são convertidos para o formato vetorial binário, adequado para `categorical_crossentropy`.

---

## 🧠 2. Definição da Arquitetura da CNN

```python
def criar_modelo():
	model = tf.keras.models.Sequential([  
    tf.keras.Input(shape=(28, 28, 1)),  
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  
    tf.keras.layers.MaxPooling2D((2, 2)),  
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  
    tf.keras.layers.MaxPooling2D((2, 2)),  
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(10, activation='softmax')  
])  
	
	model.compile(optimizer='adam',  
    loss='categorical_crossentropy',  
    metrics=['accuracy'])  
	
	return model
```

### Estrutura da CNN:

1. **Entrada**: imagens 28x28 em escala de cinza (1 canal).
2. **Camadas convolucionais e de pooling**:
   - `Conv2D(32) + MaxPooling2D`
   - `Conv2D(64) + MaxPooling2D`
3. **Camada Flatten**: transforma os dados em vetor 1D.
4. **Camada densa (128 neurônios) com Dropout (0.5)**: regularização para evitar overfitting.
5. **Saída**: 10 neurônios com ativação `softmax` (classificação multiclasse).
6. **Compilação**: otimizador `adam`, função de perda `categorical_crossentropy`, métrica `accuracy`.

---

## 🏃 3. Treinamento, Avaliação e Visualização

```python
def treinar_e_testar(X, y, proporcao_treino):
	X_train, X_test, y_train, y_test = train_test_split(  
	X, y, train_size=proporcao_treino, random_state=42, stratify=y.argmax(axis=1)  
)  
  
model = criar_modelo()  
  
early_stop = EarlyStopping(  
    monitor='val_loss',  
    patience=3,  
    restore_best_weights=True  
)  
  
history = model.fit(  
    X_train, y_train,  
    epochs=50,  
    batch_size=128,  
    verbose=1,  
    validation_split=0.1,  
    callbacks=[early_stop]  
)  
  
# Plotar gráficos de perda e acurácia  
plt.figure(figsize=(12, 5))  
  
plt.subplot(1, 2, 1)  
plt.plot(history.history['loss'], label='Treino')  
plt.plot(history.history['val_loss'], label='Validação')  
plt.title('Perda durante o treino')  
plt.xlabel('Época')  
plt.ylabel('Loss')  
plt.legend()  
  
plt.subplot(1, 2, 2)  
plt.plot(history.history['accuracy'], label='Treino')  
plt.plot(history.history['val_accuracy'], label='Validação')  
plt.title('Acurácia durante o treino')  
plt.xlabel('Época')  
plt.ylabel('Accuracy')  
plt.legend()  
  
filename_plot = f"training_plot_{int(proporcao_treino * 100)}_{int((1 - proporcao_treino) * 100)}.png"  
plt.savefig(filename_plot)  
plt.close()  
print(f"Gráficos de treino salvos em: {filename_plot}")  
  
# Avaliação  
loss, acc = model.evaluate(X_test, y_test, verbose=0)  
divisao = f"{round(proporcao_treino * 100)}_{round((1 - proporcao_treino) * 100)}"  
print(f"Acurácia com {divisao.replace('_', '/')}: {acc:.4f}")  
  
y_pred = model.predict(X_test)  
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = np.argmax(y_test, axis=1)  
  
print(classification_report(y_true, y_pred_classes))  
cm = confusion_matrix(y_true, y_pred_classes)  
  
# Criar a matriz em markdown  
markdown_lines = []  
header = "| |" + "|".join(str(i) for i in range(10)) + "|\n"  
separator = "|---" * 10 + "|---|\n"  
markdown_lines.append(header)  
markdown_lines.append(separator)  
for i, row in enumerate(cm):  
    row_str = f"|{i}|" + "|".join(str(cell) for cell in row) + "|\n"  
    markdown_lines.append(row_str)  
  
filename = f"confusion_matrix_{divisao}.md"  
with open(filename, "w") as f:  
    f.write(f"# Confusion Matrix ({divisao.replace('_', '/')})\n\n")  
    f.write(f"**Accuracy:** {acc:.4f}\n\n")  
    f.writelines(markdown_lines)  
  
print(f"\nMatriz de confusão salva como: {filename}")
```

### Etapas executadas:

#### a) Separação dos dados
- Utiliza `train_test_split` com estratificação para manter a proporção de classes.

#### b) Treinamento
- Modelo criado com `criar_modelo()`.
- `EarlyStopping` interrompe o treino se a `val_loss` não melhorar por 3 épocas.
- Utiliza `validation_split=0.1` durante o treino.

#### c) Visualização
- Gera gráficos de perda (`loss`) e acurácia (`accuracy`) ao longo das épocas.
- Salva como imagem com nome baseado na proporção treino/teste, por exemplo: `training_plot_80_20.png`.

#### d) Avaliação
- Calcula e exibe a acurácia final no conjunto de teste.
- Gera predições e imprime:
  - `classification_report`: precisão, recall e f1-score por classe.
  - `confusion_matrix`: matriz de confusão.

#### e) Salvamento da matriz de confusão
- Formata a matriz em Markdown.
- Salva o conteúdo em um arquivo `.md` com nome como `confusion_matrix_80_20.md`.

---

## 🧪 4. Testes com Diferentes Proporções de Treinamento/Teste

```python
proporcoes = [0.8, 0.9, 0.7]
for prop in proporcoes:
    treinar_e_testar(X, y, prop)
```

### O que acontece:
- Para cada proporção definida (`80/20`, `90/10`, `70/30`), a função `treinar_e_testar` é executada.
- Isso permite comparar o desempenho da CNN em diferentes cenários de disponibilidade de dados.

---

## 📁 Arquivos Gerados

- **Imagens de gráficos de treino**: `training_plot_XX_YY.png`
- **Matrizes de confusão em Markdown**: `confusion_matrix_XX_YY.md`

---

## 🧰 Bibliotecas Utilizadas

- `numpy`, `pandas`: manipulação de dados.
- `tensorflow.keras`: criação e treino do modelo CNN.
- `sklearn`: divisão dos dados e avaliação do modelo.
- `matplotlib`: visualização de desempenho.
- `EarlyStopping`: parada antecipada para evitar overfitting.

